@testset "Normalization: Enzyme reverse mode" begin
    # The Enzyme rules of `NNlibEnzymeCoreExt` run the `rrule`s: the fused kernels, cuDNN
    # for `batchnorm`, and the running statistics updated once. Their vector-Jacobian
    # products must match Zygote's. The shadows are fresh `zero`s: re-zeroing them with
    # `remake_zero!` corrupts a `CuArray` when Enzyme is loaded before CUDA, as it is here
    # (https://github.com/EnzymeAD/Enzyme.jl/issues/3806).
    function enzyme_pullback(f, dy, args...; consts=())
        ann = ntuple(i -> i in consts ? Const(args[i]) : Duplicated(args[i], zero(args[i])), length(args))
        fwd, rev = Enzyme.autodiff_thunk(ReverseSplitWithPrimal, Const{typeof(f)}, Duplicated, map(typeof, ann)...)
        tape, y, shadow = fwd(Const(f), ann...)
        copyto!(shadow, dy)
        rev(Const(f), ann..., tape)
        return y, map(a -> a isa Const ? nothing : a.dval, ann)
    end

    function test_enzyme(f, args...; consts=(), rtol=1e-4, atol=1e-4)
        y, back = Zygote.pullback(f, args...)
        dy = CUDA.randn(Float32, size(y))
        dy = eltype(y) == Float32 ? dy : eltype(y).(dy)
        ye, grads = enzyme_pullback(f, dy, args...; consts)
        @test collect(ye) ≈ collect(y) rtol=rtol atol=atol
        for (ge, gz, a) in zip(grads, back(dy), args)
            ge === nothing && continue
            @test collect(ge) ≈ collect(something(gz, zero(a))) rtol=rtol atol=atol
        end
    end

    @testset "$name" for (name, f, gsz, xsz) in (
            ("layernorm", (g, b, x) -> layernorm(g, b, x), (300,), (300, 5)),
            ("layernorm dims", (g, b, x) -> layernorm(g, b, x; dims=1:2, eps=1f-3), (4, 5), (4, 5, 3, 8)),
            ("groupnorm", (g, b, x) -> groupnorm(g, b, x, 3), (6,), (4, 5, 6, 2)),
            ("instancenorm", (g, b, x) -> instancenorm(g, b, x), (6,), (4, 5, 6, 2)),
            ("batchnorm", (g, b, x) -> batchnorm(g, b, x), (6,), (4, 5, 6, 2)),
            ("batchnorm 2-D", (g, b, x) -> batchnorm(g, b, x), (6,), (6, 16)))
        g, b, x = CUDA.randn(Float32, gsz), CUDA.randn(Float32, gsz), CUDA.randn(Float32, xsz)
        test_enzyme(f, g, b, x)
        # each activity compiles anew: constant arguments for the fused kernels and cuDNN
        if name in ("layernorm", "batchnorm")
            test_enzyme(f, g, b, x; consts=(1, 2))
            test_enzyme(f, g, b, x; consts=(3,))
        end
    end

    @testset "rmsnorm" begin
        g, x = CUDA.randn(Float32, 300), CUDA.randn(Float32, 300, 5)
        test_enzyme((g, x) -> rmsnorm(g, x), g, x)
        test_enzyme(x -> rmsnorm(nothing, x; dims=1:2), x)
    end

    @testset "no affine transform" begin
        x4 = CUDA.randn(Float32, 4, 5, 6, 2)
        test_enzyme(x -> NNlib.normalise(x; dims=1:3), x4)
        test_enzyme(x -> groupnorm(nothing, nothing, x, 2), x4)
        test_enzyme(x -> instancenorm(nothing, nothing, x), x4)
        test_enzyme(x -> batchnorm(nothing, nothing, x), x4)
        test_enzyme(x -> batchnorm(nothing, nothing, x), CUDA.randn(Float32, 6, 16))
    end

    @testset "running statistics, $(ndims(x))-D" for x in (CUDA.randn(Float32, 4, 5, 6, 8), CUDA.randn(Float32, 6, 16))
        g, b = CUDA.randn(Float32, 6), CUDA.randn(Float32, 6)
        rm, rv = CUDA.randn(Float32, 6), CUDA.rand(Float32, 6) .+ 0.5f0
        norms = ndims(x) > 2 ? (batchnorm, instancenorm) : (batchnorm,)
        @testset "$norm" for norm in norms
            # inference: normalising with the running statistics
            test_enzyme((g, b, x) -> norm(g, b, x, rm, rv; training=false), g, b, x)
            test_enzyme(x -> norm(nothing, nothing, x, rm, rv, 0.1f0; training=false), x)
            # training: updated once per gradient, as by one plain call, with no gradient
            rm1, rv1 = copy(rm), copy(rv)
            y1 = norm(g, b, x, rm1, rv1, 0.1f0; training=true, track_stats=true)
            f(g, b, x, rm, rv) = norm(g, b, x, rm, rv, 0.1f0; training=true, track_stats=true)
            _, back = Zygote.pullback(f, g, b, x, copy(rm), copy(rv))
            dy = CUDA.randn(Float32, size(x))
            rm2, rv2 = copy(rm), copy(rv)
            y, (dg, db, dx, drm, drv) = enzyme_pullback(f, dy, g, b, x, rm2, rv2)
            @test collect(y) ≈ collect(y1) rtol=1e-4 atol=1e-4
            @test collect(rm2) ≈ collect(rm1) rtol=1e-5
            @test collect(rv2) ≈ collect(rv1) rtol=1e-5
            @test all(iszero, collect(drm)) && all(iszero, collect(drv))
            for (ge, gz) in zip((dg, db, dx), back(dy))
                @test collect(ge) ≈ collect(gz) rtol=1e-4 atol=1e-4
            end
        end
    end

    # cuDNN's BFloat16 needs compute capability 8.0
    halfs = CUDA.capability(CUDA.device()) >= v"8.0" ? (Float16, BFloat16) : (Float16,)
    @testset "$T" for T in halfs
        # half-precision feature maps with Float32 scale and bias
        x2 = T.(CUDA.randn(Float32, 300, 5)); x4 = T.(CUDA.randn(Float32, 4, 5, 6, 2))
        test_enzyme((g, b, x) -> layernorm(g, b, x), CUDA.randn(Float32, 300), CUDA.randn(Float32, 300), x2;
                    rtol=1e-2, atol=1e-2)
        test_enzyme((g, b, x) -> groupnorm(g, b, x, 3), CUDA.randn(Float32, 6), CUDA.randn(Float32, 6), x4;
                    rtol=1e-2, atol=1e-2)
        test_enzyme((g, b, x) -> batchnorm(g, b, x), CUDA.randn(Float32, 6), CUDA.randn(Float32, 6), x4;
                    rtol=1e-2, atol=1e-2)
    end
end
