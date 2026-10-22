@testset "Batchnorm" begin
    @testset "Mooncake" begin
        rng = Random.MersenneTwister(42)
        N, B = 3, 4
        g  = CUDA.ones(Float32, N)
        b  = CUDA.zeros(Float32, N)
        x4 = CUDA.randn(Float32, 1, 1, N, B)
        # Use nothing for running stats to avoid in-place mutation across test_rule calls.
        _bn(g, b, x) = sum(batchnorm(g, b, x, nothing, nothing, 0.1f0; training=true))
        test_rule(rng, _bn, g, b, x4; is_primitive=false, mode=Mooncake.ReverseMode)
    end
    v = CUDA.rand(Float32, 2)
    m = CUDA.rand(Float32, 2, 5)

    @testset for training in (true, false), track_stats in (true, false)
        kws = (training=training, track_stats=track_stats)

        # Normal
        batchnorm(v, v, m, v, v, 1.0; kws...)
        ∇batchnorm(v, v, m, m, v, v, 1.0; kws...)

        # No affine
        batchnorm(nothing, nothing, m, v, v, 1.0; kws...)
        ∇batchnorm(nothing, nothing, m, m, v, v, 1.0; kws...)

        # No tracking
        batchnorm(v, v, m, nothing, nothing, 1.0; kws...)
        ∇batchnorm(v, v, m, m, nothing, nothing, 1.0; kws...)

        # Both or neither tracked or affine params must be set
        for (α, β) in ((v, nothing), (nothing, v))
            @test_throws ArgumentError batchnorm(α, β, m, v, v, 1.0; kws...)
            @test_throws ArgumentError ∇batchnorm(α, β, m, m, v, v, 1.0; kws...)
            @test_throws ArgumentError batchnorm(v, v, m, α, β, 1.0; kws...)
        end
    end
    @testset "2D input uses the generic code" begin
        # cuDNN is slower than the generic code on a (1, 1, C, N) tensor.
        @test parentmodule(which(batchnorm, typeof.((v, v, m, v, v, 1.0)))) === NNlib
        @test parentmodule(which(batchnorm, typeof.((nothing, nothing, m, v, v, 1.0)))) === NNlib
        @test parentmodule(which(∇batchnorm, typeof.((v, v, m, m, v, v, 1.0)))) === NNlib
        @test parentmodule(which(rrule, typeof.((batchnorm, v, v, m, v, v, 1.0)))) === NNlib
        # the keywords of the cuDNN methods are accepted (Flux ≤ 0.16.10 passes them)
        @test batchnorm(v, v, m, nothing, nothing, 1.0; cache=nothing, alpha=1, beta=0) ≈ batchnorm(v, v, m)
    end
    @testset "3D input (issue #753)" begin
        # cuDNN batchnorm supports only 4D/5D descriptors; a 3D (W, C, N) input is
        # reshaped to 4D (1, W, C, N) on both the forward and backward paths. Guards
        # against the regression where the backward path errored with BAD_PARAM.
        g3 = rand(Float32, 4)
        b3 = rand(Float32, 4)
        x3 = rand(Float32, 5, 4, 8)
        _bn3(g, b, x) = batchnorm(g, b, x, nothing, nothing, 0.1f0; training=true)
        gputest(_bn3, g3, b3, x3; rtol=1e-3, atol=1e-4)
    end
    @testset "test mode" begin
        y_no_track_stats = batchnorm(v, v, m, nothing, nothing, 1.0; training=false, track_stats=false)
        running_mean = mean(m, dims=[2])
        running_var = Statistics.var(m, mean=running_mean, dims=[2], corrected=false)
        y_track_stats = batchnorm(v, v, m, running_mean, running_var, 1.0; training=false, track_stats=true)
        # batchnorm without tracked stats should equal bathnorm with tracked stats where the
        # stats are calculated only on the input.
        @test y_no_track_stats ≈ y_track_stats
    end
    @testset "test mode gradient (Flux.jl#2179)" begin
        # In inference mode with tracked statistics the forward pass normalises by the
        # *fixed* running mean/variance, so the gradient is a per-channel affine rescaling
        # — not the training-mode gradient that `cudnnBatchNormalizationBackward` computes.
        # This regressed silently because only the forward pass was checked above. Compare
        # the cuDNN path against the generic (CPU) reference for the input and, where
        # present, the affine-parameter gradients. Running stats differ from the batch
        # statistics so the inference gradient is genuinely distinct from the training one.
        @testset for sz in ((4, 8), (3, 3, 4, 8))
            C = sz[end-1]
            g = rand(Float32, C); b = rand(Float32, C)
            rm = randn(Float32, C); rv = rand(Float32, C) .+ 0.5f0
            x = randn(Float32, sz)
            gputest((g, b, x, rm, rv) -> batchnorm(g, b, x, rm, rv, 0.1f0; training=false, track_stats=true),
                    g, b, x, rm, rv; rtol=1e-3, atol=1e-4)
            gputest((x, rm, rv) -> batchnorm(nothing, nothing, x, rm, rv, 0.1f0; training=false, track_stats=true),
                    x, rm, rv; rtol=1e-3, atol=1e-4)
        end
    end
    @testset "second order (Flux.jl#2154)" begin
        # BatchNorm must be twice differentiable on the GPU. First-order gradients use
        # the fast cuDNN backward (dispatched on `x::CuArray{<:CUDNNFloat}`); nesting AD
        # seeds `ForwardDiff.Dual` eltypes, which fall through that dispatch to the
        # generic differentiable `batchnorm`/`∇batchnorm` in NNlib core. We check the
        # Hessian–vector product both ways — forward-over-reverse and reverse-over-reverse
        # — against the (finite-difference-validated) CPU reference. The loss multiplies
        # by a fixed array `a` to break batchnorm's normalisation invariance, so the
        # second-order term is genuinely non-zero.
        hvp(loss, X, V) =  # (reverse-over-reverse, forward-over-reverse)
            (Zygote.gradient(x -> sum(Zygote.gradient(loss, x)[1] .* V), X)[1],
             ForwardDiff.derivative(ε -> Zygote.gradient(loss, X .+ ε .* V)[1], 0f0))
        @testset for sz in ((3, 6), (4, 5, 3, 6))
            C = sz[end-1]
            gc = randn(Float32, C); bc = randn(Float32, C); ac = randn(Float32, sz)
            xc = randn(Float32, sz); vc = randn(Float32, sz)
            rmc = randn(Float32, C); rvc = rand(Float32, C) .+ 0.5f0
            @testset for (training, rm, rv) in ((true, nothing, nothing), (false, rmc, rvc))
                dev(a) = a === nothing ? nothing : CuArray(a)
                mkloss(g, b, a, rm, rv) =
                    x -> sum(abs2, a .* batchnorm(g, b, x, rm, rv, 0.1f0; training))
                rev_c, fwd_c = hvp(mkloss(gc, bc, ac, rm, rv), xc, vc)
                rev_g, fwd_g = hvp(mkloss(dev(gc), dev(bc), dev(ac), dev(rm), dev(rv)),
                                   CuArray(xc), CuArray(vc))
                @test collect(rev_g) ≈ rev_c rtol=1e-3 atol=1e-4
                @test collect(fwd_g) ≈ fwd_c rtol=1e-3 atol=1e-4
            end
        end
    end

    @testset "Float16" begin
        # cuDNN needs Float32 parameters and statistics for half-precision feature maps.
        x = CUDA.rand(Float16, 1, 1, 3, 4)
        g = CUDA.rand(Float32, 3)
        b = CUDA.rand(Float32, 3)
        running_mean = CUDA.zeros(Float32, 3)
        running_var = CUDA.ones(Float32, 3)
        ext = Base.get_extension(NNlib, :NNlibCUDACUDNNExt)
        cache = ext.BNCache()
        y = batchnorm(g, b, x, running_mean, running_var, 0.1f0; training=true,
                      track_stats=true, cache)

        x32 = Float32.(Array(x))
        mean = sum(x32; dims=(1, 2, 4)) ./ 4
        var = sum(abs2, x32 .- mean; dims=(1, 2, 4)) ./ 4
        g32 = reshape(Array(g), 1, 1, 3, 1)
        b32 = reshape(Array(b), 1, 1, 3, 1)
        ref = @. g32 * (x32 - mean) / sqrt(var + 1f-5) + b32
        @test eltype(y) == Float16
        @test Float32.(Array(y)) ≈ ref rtol=2f-3 atol=2f-3
        @test eltype(cache.mean) == Float32
        @test eltype(cache.ivar) == Float32
        # the running statistics are updated in place, with the unbiased variance
        @test Array(running_mean) ≈ 0.1f0 .* vec(mean) rtol=2f-3 atol=2f-3
        @test Array(running_var) ≈ 0.9f0 .+ 0.1f0 .* (4 / 3) .* vec(var) rtol=2f-3 atol=2f-3

        yi = batchnorm(g, b, x, running_mean, running_var, 0.1f0; training=false,
                       track_stats=true)
        running_mean32 = reshape(Array(running_mean), 1, 1, 3, 1)
        running_var32 = reshape(Array(running_var), 1, 1, 3, 1)
        inference_ref = @. g32 * (x32 - running_mean32) /
                           sqrt(running_var32 + 1f-5) + b32
        @test Float32.(Array(yi)) ≈ inference_ref rtol=2f-3 atol=2f-3

        dy = CUDA.rand(Float16, size(x))
        dy32 = Float32.(Array(dy))
        invvar = @. inv(sqrt(var + 1f-5))
        xhat = @. (x32 - mean) * invvar
        db_ref = sum(dy32; dims=(1, 2, 4))
        dg_ref = sum(dy32 .* xhat; dims=(1, 2, 4))
        dx_ref = @. g32 * invvar / 4 * (4 * dy32 - db_ref - xhat * dg_ref)
        # with the statistics saved by the forward pass, and recomputing them
        @testset "cache = $(c === nothing ? "nothing" : "saved")" for c in (cache, nothing)
            dg, db, dx = ∇batchnorm(g, b, x, dy, running_mean, running_var, 0.1f0;
                                    training=true, track_stats=true, cache=c)
            @test eltype(dg) == Float32
            @test eltype(db) == Float32
            @test eltype(dx) == Float16
            @test Array(dg) ≈ vec(dg_ref) rtol=3f-3 atol=3f-3
            @test Array(db) ≈ vec(db_ref) rtol=3f-3 atol=3f-3
            @test Float32.(Array(dx)) ≈ dx_ref rtol=3f-3 atol=3f-3
        end
    end
end
