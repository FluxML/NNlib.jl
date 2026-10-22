@testset "Normalization (instance/group/layer)" begin
    # On the GPU `instancenorm`/`groupnorm` route their standardisation through the cuDNN
    # `batchnorm` fast path (a reshape trick) and apply the affine transform generically;
    # `layernorm` over leading `dims` runs the fused kernels. The result must match the
    # generic CPU implementation both forward and backward. `gputest` compares the two,
    # including Zygote gradients.

    @testset "dispatch" begin
        # Guard against silently falling back to the generic path (which would still be
        # correct, just slower) for the supported cases.
        ext = Base.get_extension(NNlib, :NNlibCUDACUDNNExt)
        @test ext !== nothing
        xf = CUDA.zeros(Float32, 4, 5, 3, 8)
        @test parentmodule(which(instancenorm, typeof.((nothing, nothing, xf)))) === ext
        @test parentmodule(which(groupnorm, typeof.((nothing, nothing, xf, 3)))) === ext
        @testset for T in (Float32, Float16, BFloat16)
            x = CUDA.zeros(T, 768, 4); g = CUDA.ones(Float32, 768)
            @test NNlib.use_fused_norm(x, 1, g, g)
            @test NNlib.use_fused_norm(x, 1:1, nothing, nothing)
        end
    end

    @testset "instancenorm" begin
        @testset for sz in ((5, 4, 8), (4, 5, 3, 8), (3, 4, 2, 6, 5))
            C = sz[end-1]
            g = randn(Float32, C); b = randn(Float32, C); x = randn(Float32, sz)
            gputest((g, b, x) -> instancenorm(g, b, x), g, b, x; rtol=1e-3, atol=1e-4)
            gputest(x -> instancenorm(nothing, nothing, x), x; rtol=1e-3, atol=1e-4)
        end
    end

    @testset "groupnorm" begin
        # G=1 (all channels one group) and G=C (per-channel, ≡ instancenorm) are edge cases.
        @testset for (sz, G) in (((4, 5, 6, 2), 3), ((4, 5, 6, 2), 6), ((4, 5, 6, 2), 1),
                                 ((3, 3, 4, 8, 2), 4))
            C = sz[end-1]
            g = randn(Float32, C); b = randn(Float32, C); x = randn(Float32, sz)
            gputest((g, b, x) -> groupnorm(g, b, x, G), g, b, x; rtol=1e-3, atol=1e-4)
            gputest(x -> groupnorm(nothing, nothing, x, G), x; rtol=1e-3, atol=1e-4)
        end
    end

    @testset "layernorm" begin
        # Leading `dims` take the fused kernels; columns of 256 elements or more are shared
        # by several lanes.
        @testset for (sz, dims) in (((6, 4), 1), ((4, 5, 3, 8), (1, 2)), ((4, 5, 3, 8), (1, 2, 3)),
                                    ((1000, 7), 1), ((300, 5, 3), (1, 2)))
            ds = dims isa Integer ? (dims,) : dims
            gsz = ntuple(i -> i in ds ? sz[i] : 1, length(sz))
            g = randn(Float32, gsz); b = randn(Float32, gsz); x = randn(Float32, sz)
            gputest((g, b, x) -> layernorm(g, b, x; dims), g, b, x; rtol=1e-3, atol=1e-4)
            gputest((g, x) -> rmsnorm(g, x; dims), g, x; rtol=1e-3, atol=1e-4)
            gputest(x -> NNlib.normalise(x; dims), x; rtol=1e-3, atol=1e-4)
        end
        # Non-leading `dims` fall back to the generic path but must stay correct.
        let x = randn(Float32, 4, 5, 3, 8), g = randn(Float32, 1, 1, 3, 1), b = randn(Float32, 1, 1, 3, 1)
            gputest((g, b, x) -> layernorm(g, b, x; dims=3), g, b, x; rtol=1e-3, atol=1e-4)
        end
    end

    @testset "instancenorm running stats" begin
        # Tracking running statistics is not cuDNN-eligible; it falls back to the generic
        # path, which updates the per-channel stats in place.
        x = randn(Float32, 4, 5, 3, 8); g = randn(Float32, 3); b = randn(Float32, 3); mom = 0.1f0
        rm = CUDA.zeros(Float32, 3); rv = CUDA.ones(Float32, 3)
        instancenorm(CuArray(g), CuArray(b), CuArray(x), rm, rv, mom; training=true, track_stats=true)
        μi = vec(mean(mean(x; dims=(1, 2)); dims=4)); mi = 4 * 5
        σ²i = vec(mean(Statistics.var(x; dims=(1, 2), corrected=false); dims=4))
        @test Array(rm) ≈ mom .* μi rtol=1e-4
        @test Array(rv) ≈ (1 - mom) .* ones(Float32, 3) .+ mom .* (mi / (mi - 1)) .* σ²i rtol=1e-4
    end

    @testset "half precision fallback ($T)" for T in (Float16, BFloat16)
        # Half precision routes to the generic path (cuDNN's parameter type must match the
        # value type there); Float32 affine parameters are required, as on the CPU.
        x = randn(Float32, 4, 5, 6, 2); g = randn(Float32, 6); b = randn(Float32, 6)
        xd, gd, bd = CuArray(T.(x)), CuArray(g), CuArray(b)
        @test Array(Float32.(instancenorm(gd, bd, xd))) ≈ instancenorm(g, b, x) rtol=1e-1 atol=5e-2
        @test Array(Float32.(groupnorm(gd, bd, xd, 3))) ≈ groupnorm(g, b, x, 3) rtol=1e-1 atol=5e-2
    end

    @testset "fused layernorm and rmsnorm in $T" for T in (Float16, BFloat16)
        # half-precision inputs accumulate in Float32, as on the CPU
        x = randn(Float32, 768, 9); g = randn(Float32, 768); b = randn(Float32, 768); dy = randn(Float32, 768, 9)
        xd, gd, bd, dyd = CuArray(T.(x)), CuArray(g), CuArray(b), CuArray(T.(dy))
        xr = Float32.(T.(x)); dyr = Float32.(T.(dy))
        y = layernorm(gd, bd, xd)
        @test eltype(y) == T
        @test Float32.(Array(y)) ≈ layernorm(g, b, xr) rtol=2e-2 atol=2e-2
        @test Float32.(Array(rmsnorm(gd, xd))) ≈ rmsnorm(g, xr) rtol=2e-2 atol=2e-2
        for (dk, dr) in zip(NNlib.∇layernorm(gd, bd, xd, dyd), NNlib.∇layernorm(g, b, xr, dyr))
            @test Float32.(Array(dk)) ≈ dr rtol=2e-2 atol=2e-2 * maximum(abs, dr)
        end
        for (dk, dr) in zip(NNlib.∇rmsnorm(gd, xd, dyd), NNlib.∇rmsnorm(g, xr, dyr))
            @test Float32.(Array(dk)) ≈ dr rtol=2e-2 atol=2e-2 * maximum(abs, dr)
        end
    end
end
