using NNlib: batchnorm, instancenorm, groupnorm, layernorm, rmsnorm, normalise

function normalization_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    gpu = Backend != CPU
    T = Float32
    atol = 1e-3
    var(a; kws...) = Statistics.var(a; kws...)
    # Enzyme's reverse pass over GPU reductions (mean/var) currently segfaults
    # upstream, so on a GPU backend we compare only against Zygote.
    cmp = gpu ? AutoZygote() : [AutoZygote(), AutoEnzyme()]

    # Reference forward pass (matches Flux's `_norm_layer_forward`) for the
    # channel-wise layers, computed on the CPU.
    function ref_norm(g, b, x, reduce_dims; eps=1f-5)
        μ = mean(x; dims=reduce_dims)
        σ² = var(x; mean=μ, dims=reduce_dims, corrected=false)
        N = ndims(x)
        as = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
        g === nothing && return (x .- μ) ./ sqrt.(σ² .+ eps)
        gr = reshape(g, as); br = reshape(b, as)
        s = gr ./ sqrt.(σ² .+ eps)
        return s .* x .- s .* μ .+ br
    end

    @testset "batchnorm" begin
        x = randn(T, 4, 5, 3, 8); g = randn(T, 3); b = randn(T, 3)
        rd = (1, 2, 4)
        @test cpu(batchnorm(device(g), device(b), device(x))) ≈ ref_norm(g, b, x, rd) atol=atol
        @test cpu(batchnorm(nothing, nothing, device(x))) ≈ ref_norm(nothing, nothing, x, rd) atol=atol
        # 2D (feature-vector) input
        x2 = randn(T, 3, 8)
        @test cpu(batchnorm(device(g), device(b), device(x2))) ≈ ref_norm(g, b, x2, (2,)) atol=atol

        @test test_gradients(batchnorm, g, b, x; test_gpu=gpu, atol, compare=cmp)

        @test_throws ArgumentError batchnorm(nothing, nothing, device(randn(T, 5)))
        # the scale and bias must be given together or not at all
        @test_throws ArgumentError batchnorm(device(g), nothing, device(x))
        @test_throws ArgumentError batchnorm(nothing, device(b), device(x))
        @test_throws ArgumentError NNlib.∇batchnorm(device(g), nothing, device(x), device(x))
    end

    @testset "batchnorm running stats" begin
        x = randn(T, 4, 5, 3, 8); g = randn(T, 3); b = randn(T, 3); mom = 0.1f0
        rm = device(zeros(T, 3)); rv = device(ones(T, 3))
        batchnorm(device(g), device(b), device(x), rm, rv, mom; training=true)
        μ = vec(mean(x; dims=(1,2,4))); σ² = vec(var(x; dims=(1,2,4), corrected=false)); m = 4*5*8
        @test cpu(rm) ≈ mom .* μ atol=atol
        @test cpu(rv) ≈ (1-mom) .* ones(T, 3) .+ mom .* (m/(m-1)) .* σ² atol=atol
        # inference: normalise with the stored running stats
        yinf = cpu(batchnorm(device(g), device(b), device(x), rm, rv, mom; training=false))
        rm4 = reshape(cpu(rm),1,1,3,1); rv4 = reshape(cpu(rv),1,1,3,1)
        @test yinf ≈ reshape(g,1,1,3,1)./sqrt.(rv4.+1f-5).*(x.-rm4).+reshape(b,1,1,3,1) atol=atol
        # track_stats=false leaves the running stats untouched
        rm2 = device(zeros(T, 3)); rv2 = device(ones(T, 3))
        batchnorm(device(g), device(b), device(x), rm2, rv2, mom; training=true, track_stats=false)
        @test cpu(rm2) == zeros(T, 3) && cpu(rv2) == ones(T, 3)
    end

    # A gradient call that evaluates the function once must leave the running statistics
    # updated exactly as one plain call does. Mooncake undoes in-place writes on its
    # reverse pass unless the update is a primitive (`NNlibMooncakeExt`).
    Backend == CPU && @testset "running stats under AD" begin
        x = randn(T, 4, 5, 3, 8); g = randn(T, 3); b = randn(T, 3); mom = 0.1f0
        w = randn(T, size(x))  # keeps the loss from being invariant to the normalisation
        adtypes = AbstractADType[AutoZygote(), AutoMooncake()]
        NNLIB_TEST_ENZYME && push!(adtypes, AutoEnzyme())
        grad_once(loss, adtype, args...) = withgradient(loss, adtype, args...).grad
        # `withgradient` prepares a Mooncake cache, and preparing one evaluates the function
        # a further time, which updates the statistics twice. A bare rule evaluates it once.
        grad_once(loss, ::AutoMooncake, args...) = Base.tail(Mooncake.value_and_gradient!!(
            Mooncake.build_rrule(loss, args...), loss, args...)[2])
        @testset "$norm" for norm in (batchnorm, instancenorm)
            loss(g, b, x, rm, rv) = sum(abs2, w .* norm(g, b, x, rm, rv, mom; training=true, track_stats=true))
            rm0 = zeros(T, 3); rv0 = ones(T, 3)
            loss(g, b, x, rm0, rv0)
            @test rm0 != zeros(T, 3) && rv0 != ones(T, 3)
            ref = Zygote.gradient(loss, g, b, x, zeros(T, 3), ones(T, 3))
            @testset "$(nameof(typeof(adtype)))" for adtype in adtypes
                rm = zeros(T, 3); rv = ones(T, 3)
                grad = grad_once(loss, adtype, g, b, x, rm, rv)
                @test rm ≈ rm0
                @test rv ≈ rv0
                @test all(i -> isapprox(grad[i], ref[i]; rtol=1e-3, atol), 1:3)
            end
        end
    end

    @testset "instancenorm" begin
        x = randn(T, 4, 5, 3, 8); g = randn(T, 3); b = randn(T, 3)
        @test cpu(instancenorm(device(g), device(b), device(x))) ≈ ref_norm(g, b, x, (1,2)) atol=atol
        @test cpu(instancenorm(nothing, nothing, device(x))) ≈ ref_norm(nothing, nothing, x, (1,2)) atol=atol
        @test test_gradients(instancenorm, g, b, x; test_gpu=gpu, atol, compare=cmp)
        @test_throws ArgumentError instancenorm(nothing, nothing, device(randn(T, 3, 8)))

        # running stats accumulate the per-channel average across the batch
        rm = device(zeros(T, 3)); rv = device(ones(T, 3)); mom = 0.1f0
        instancenorm(device(g), device(b), device(x), rm, rv, mom; training=true, track_stats=true)
        μi = vec(mean(mean(x; dims=(1,2)); dims=4)); σ²i = vec(mean(var(x; dims=(1,2), corrected=false); dims=4)); mi = 4*5
        @test cpu(rm) ≈ mom .* μi atol=atol
        @test cpu(rv) ≈ (1-mom) .* ones(T, 3) .+ mom .* (mi/(mi-1)) .* σ²i atol=atol
    end

    @testset "groupnorm" begin
        x = randn(T, 4, 5, 6, 2); g = randn(T, 6); b = randn(T, 6)
        function ref_gn(g, b, x, G; eps=1f-5)
            sz = size(x); N = ndims(x); C = sz[N-1]
            x2 = reshape(x, sz[1:N-2]..., C÷G, G, sz[N])
            rd = ntuple(identity, N-1)
            μ = mean(x2; dims=rd); σ² = var(x2; mean=μ, dims=rd, corrected=false)
            as = (ntuple(_->1, N-2)..., C÷G, G, 1)
            reshape(reshape(g,as)./sqrt.(σ².+eps).*(x2.-μ).+reshape(b,as), sz)
        end
        @test cpu(groupnorm(device(g), device(b), device(x), 3)) ≈ ref_gn(g, b, x, 3) atol=atol
        @test cpu(groupnorm(nothing, nothing, device(x), 2)) ≈ ref_gn(ones(T,6), zeros(T,6), x, 2) atol=atol
        @test test_gradients((g,b,x) -> groupnorm(g, b, x, 3), g, b, x; test_gpu=gpu, atol, compare=cmp)
        @test_throws ArgumentError groupnorm(device(g), device(b), device(x), 4)  # 4 ∤ 6
    end

    @testset "layernorm / normalise" begin
        x = randn(T, 6, 4)
        μ = mean(x; dims=1); σ² = var(x; mean=μ, dims=1, corrected=false)
        @test cpu(layernorm(nothing, nothing, device(x); dims=1)) ≈ (x .- μ) ./ sqrt.(σ² .+ 1f-5) atol=atol
        g = randn(T, 6); b = randn(T, 6)
        @test cpu(layernorm(device(g), device(b), device(x); dims=1)) ≈
            g ./ sqrt.(σ².+1f-5) .* (x .- μ) .+ b atol=atol
        @test test_gradients((g,b,x) -> layernorm(g, b, x; dims=1), g, b, x; test_gpu=gpu, atol, compare=cmp)

        # normalise: zero mean, unit std over `dims`
        z = cpu(normalise(device(x); dims=1))
        @test all(isapprox.(vec(Statistics.std(z; dims=1, corrected=false)), 1; atol=1e-3))
    end

    @testset "rmsnorm" begin
        x = randn(T, 6, 4)
        rms = sqrt.(mean(abs2, x; dims=1) .+ 1f-5)
        @test cpu(rmsnorm(nothing, device(x))) ≈ x ./ rms atol=atol
        g = randn(T, 6)
        @test cpu(rmsnorm(device(g), device(x))) ≈ g .* x ./ rms atol=atol
        @test test_gradients(rmsnorm, g, x; test_gpu=gpu, atol, compare=cmp)
        @test test_gradients(x -> rmsnorm(nothing, x), x; test_gpu=gpu, atol, compare=cmp)

        # several `dims`, with a scale that broadcasts against them
        x4 = randn(T, 4, 5, 3, 8); g4 = randn(T, 4, 5)
        rms4 = sqrt.(mean(abs2, x4; dims=(1,2)) .+ 1f-5)
        @test cpu(rmsnorm(device(g4), device(x4); dims=(1,2))) ≈ g4 .* x4 ./ rms4 atol=atol
        @test test_gradients((g,x) -> rmsnorm(g, x; dims=(1,2)), g4, x4; test_gpu=gpu, atol, compare=cmp)

        # the mean is not subtracted, so a shift of the input is not normalised away
        @test cpu(rmsnorm(nothing, device(x .+ 10))) ≉ cpu(rmsnorm(nothing, device(x)))
        # `eps` keeps an all-zero input finite
        @test all(iszero, cpu(rmsnorm(nothing, device(zeros(T, 6, 4)))))

        # half precision: the mean square accumulates in Float32, so the scale must be Float32
        xh = device(Float16.(x))
        yh = rmsnorm(device(g), xh)
        @test eltype(yh) == Float16
        @test Float32.(cpu(yh)) ≈ g .* x ./ rms rtol=1e-2 atol=1e-2
        @test_throws ArgumentError rmsnorm(device(Float16.(g)), xh)
    end

    @testset "gradient operators (∇)" begin
        # The explicit VJPs must agree with the pullback returned by the rrule
        # (which is what `test_gradients` above checks against finite differences).
        x = randn(T, 4, 5, 3, 8); g = randn(T, 3); b = randn(T, 3)
        dy = randn(T, size(x))
        xd, gd, bd, dyd = device(x), device(g), device(b), device(dy)
        for (∇op, f) in (
                (NNlib.∇batchnorm,    (g,b,x) -> batchnorm(g, b, x)),
                (NNlib.∇instancenorm, (g,b,x) -> instancenorm(g, b, x)),
            )
            dg, db, dx = ∇op(gd, bd, xd, dyd)
            _, back = Zygote.pullback(f, gd, bd, xd)
            dgz, dbz, dxz = back(dyd)
            @test cpu(dg) ≈ cpu(dgz) atol=atol
            @test cpu(db) ≈ cpu(dbz) atol=atol
            @test cpu(dx) ≈ cpu(dxz) atol=atol
        end
        dg, db, dx = NNlib.∇groupnorm(gd, bd, xd, dyd, 3)
        _, back = Zygote.pullback((g,b,x) -> groupnorm(g, b, x, 3), gd, bd, xd)
        dgz, dbz, dxz = back(dyd)
        @test cpu(dg) ≈ cpu(dgz) atol=atol
        @test cpu(dx) ≈ cpu(dxz) atol=atol

        gl = randn(T, 4, 5); gld = device(gl)
        dg, dx = NNlib.∇rmsnorm(gld, xd, dyd; dims=(1,2))
        _, back = Zygote.pullback((g,x) -> rmsnorm(g, x; dims=(1,2)), gld, xd)
        dgz, dxz = back(dyd)
        @test cpu(dg) ≈ cpu(dgz) atol=atol
        @test cpu(dx) ≈ cpu(dxz) atol=atol
        @test NNlib.∇rmsnorm(nothing, xd, dyd)[1] === nothing
    end

    # Second-order differentiation. The reference HVP below builds a full Jacobian with
    # `ForwardDiff.gradient`, which is impractical on the GPU, so this generic check runs
    # on the CPU only. GPU second-order coverage lives in `test/gpu/cuda/batchnorm.jl`
    # (Flux.jl#2154): nesting AD seeds `ForwardDiff.Dual` eltypes, which route away from
    # the cuDNN kernels to this same differentiable path.
    Backend == CPU && @testset "second order" begin
        x = randn(T, 4, 5, 3, 4); g = randn(T, 3); b = randn(T, 3)
        gl = randn(T, 4, 5, 1, 1); bl = randn(T, 4, 5, 1, 1)
        function hvp_match(loss, x0)
            v = randn(T, size(x0)...)
            ref = ForwardDiff.derivative(ε -> ForwardDiff.gradient(loss, x0 .+ ε .* v), zero(T))
            grad(x) = Zygote.gradient(loss, x)[1]
            zz = Zygote.gradient(x -> sum(grad(x) .* v), x0)[1]
            return isapprox(zz, ref; rtol=1e-2, atol=1e-3)
        end
        @test hvp_match(x -> sum(abs2, layernorm(gl, bl, x; dims=(1,2))), x)
        @test hvp_match(x -> sum(abs2, rmsnorm(gl, x; dims=(1,2))), x)
        @test hvp_match(x -> sum(abs2, groupnorm(g, b, x, 3)), x)
        @test hvp_match(x -> sum(abs2, batchnorm(g, b, x, nothing, nothing, 0.1f0)), x)
        @test hvp_match(x -> sum(abs2, instancenorm(g, b, x, nothing, nothing, 0.1f0)), x)
    end
end
