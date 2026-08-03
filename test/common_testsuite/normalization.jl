using NNlib: batchnorm, instancenorm, groupnorm, layernorm, normalise

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
end
