@testset "softmax integer input" begin
    @test softmax(Int[0, 0]) == [0.5, 0.5]
end

@testset "softmax on different dims" begin
    xs = rand(fill(2, 5)...)
    out = similar(xs)
    for (fn!, fn) in [(softmax!, softmax), (logsoftmax!, logsoftmax)], i = 1:ndims(xs)
        @test fn!(out, xs; dims = i) == fn(xs; dims = i)
    end
end

@testset "softmax" begin
    xs = rand(5, 5)
    @test all(sum(softmax(xs), dims = 1) .≈ 1)
    @test all(sum(softmax(xs; dims = 2), dims = 2) .≈ 1)
    @test sum(softmax(vec(xs))) ≈ 1
    @test log.(softmax(xs; dims = 2)) ≈ logsoftmax(xs; dims = 2)

    xs = [-100_000.0, -100_000.0]
    @test softmax(xs) ≈ [0.5, 0.5]
    @test logsoftmax(xs) ≈ log.([0.5, 0.5])

    xs = rand(5)
    @test softmax(xs) ≈ exp.(xs) ./ sum(exp.(xs))
    @test logsoftmax(xs) ≈ log.(softmax(xs))

    xs = Float32[1, 2, 3000.0]
    @test logsoftmax(xs) ≈ [-2999, -2998, 0]

    xs = Float32[1 2 3; 1000 2000 3000]
    @test logsoftmax(xs) ≈ [-999 -1998 -2997; 0 0 0.0]

    y = logsoftmax(xs)
    @test ∇logsoftmax(ones(Float32, size(xs)), y) ≈ Float32[1 1 1; -1 -1 -1]
    
    y = softmax(xs)
    @test ∇softmax(ones(Float32, size(xs)), y) ≈ zeros(Float32, size(xs))

    # These values precalculated using PyTorch's nn.LogSoftmax
    xs = [
        -0.238639 0.748142 -0.283194 -0.525461 -1.5348 -0.797842
        0.690384 0.211427 0.254794 -0.213572 -0.314174 -0.372663
        -1.146370 -0.577988 0.718952 0.919720 -0.620773 0.929977
    ]
    ys = [
        0.237703 -0.621474 0.448193 0.546047 0.564185 0.632273
        -0.930163 0.0519798 0.0549979 0.3799 -0.477112 0.437428
        0.69246 0.569494 -0.503191 -0.925947 -0.0870738 -1.0697
    ]
    
    y = logsoftmax(xs)
    @test ∇logsoftmax(ones(size(xs)), y) ≈ ys rtol = 1e-6
    
    y = softmax(xs)
    @test ∇softmax(ones(size(xs)), y) ≈ zeros(size(xs)) atol = 1e-6
end

@testset "softmax with Inf, NaN" begin
    @test softmax(Float32[1 2; 3 Inf]) ≈    Float32[0.11920292 0.0; 0.880797 1.0]
    @test softmax(Float32[1 -Inf; 3 Inf]) ≈ Float32[0.11920292 0.0; 0.880797 1.0]
    @test softmax(Float32[1 Inf; 3 Inf]) ≈  Float32[0.11920292 0.5; 0.880797 0.5]

    @test softmax(Float32[1 2; 3 NaN]) ≈    Float32[0.11920292 NaN; 0.880797 NaN] nans=true
    @test softmax(Float32[1 2; 3 Inf]; dims=2) ≈ Float32[0.26894143 0.7310586; 0.0 1.0]
    @test softmax(Float32[1 2; 3 Inf]; dims=(:)) ≈ Float32[0.0 0.0; 0.0 1.0]
    @test softmax(Float32[1 2; 3 Inf]; dims=(1,2)) ≈ Float32[0.0 0.0; 0.0 1.0]

    @test exp.(logsoftmax(Float32[1 2; 3 Inf])) ≈ softmax(Float32[1 2; 3 Inf])
    @test exp.(logsoftmax(Float32[1 -Inf; 3 Inf])) ≈ softmax(Float32[1 -Inf; 3 Inf])
    @test exp.(logsoftmax(Float32[1 Inf; 3 Inf])) ≈ softmax(Float32[1 Inf; 3 Inf])
end

@testset "mutating softmax" begin
    map([
        Float64[1 2 3; 5 6 7],
        Float64[
            -0.238639 0.748142 -0.283194 -0.525461 -1.5348 -0.797842
            0.690384 0.211427 0.254794 -0.213572 -0.314174 -0.372663
            -1.146370 -0.577988 0.718952 0.919720 -0.620773 0.929977
        ],
    ]) do xs
        out = similar(xs)
        softmax!(out, xs)
        @test out ≈ softmax(xs) rtol = 1e-6
        logsoftmax!(out, xs)
        @test out ≈ logsoftmax(xs) rtol = 1e-6

        @testset "$fn(Float64, $(size(xs)))" for fn in [zeros, ones, rand]
            Δ = fn(Float64, size(xs))
            y = softmax(xs) 
            ∇softmax!(out, Δ, y)
            @test out ≈ ∇softmax(Δ, y)  rtol = 1e-6
            
            y = logsoftmax(xs)
            ∇logsoftmax!(out, Δ, y)
            @test out ≈ ∇logsoftmax(Δ, y)  rtol = 1e-6
        end
    end
end

@testset "logsumexp" begin
    flogsoft(x; dims) = mean(x .- logsoftmax(x; dims = dims), dims = dims)

    x = rand(3, 4)
    @test logsumexp(x) ≈ flogsoft(x, dims = :)
    @test logsumexp(x; dims = 1) ≈ flogsoft(x, dims = 1)
end

# Sums are accumulated in Float32 (#813). Near-uniform logits over a large vocabulary
# are the hard case: summed in their own type, BFloat16 keeps no correct digit.
@testset "half precision: $T" for T in (Float16, BFloat16)
    maxerr(a, b) = maximum(abs, Float64.(a) .- b)
    ulp(ref) = eps(T(maximum(abs, ref)))  # of the largest entry
    hrng = StableRNG(813)
    V = 50257
    x2 = T.(0.01f0 .* randn(hrng, Float32, V, 2))
    dy2 = T.(randn(hrng, Float32, V, 2))
    tol = eps(T(log(V)))  # an ulp of the outputs of `logsoftmax` and `logsumexp`
    @testset "dims = $d" for (x, dy, d) in ((x2, dy2, 1), (permutedims(x2), permutedims(dy2), 2),
                                            (x2[:, 1], dy2[:, 1], :))
        xd, dyd = Float64.(x), Float64.(dy)
        y, ly, lse = softmax(x; dims = d), logsoftmax(x; dims = d), logsumexp(x; dims = d)
        @test eltype(y) == eltype(ly) == T
        @test eltype(lse) == Float32  # not rounded to T (#826)
        @test NNlib.fast_maximum(x; dims = d) == maximum(Float32.(x); dims = d)
        @test maxerr(sum(Float64, y; dims = d), 1) < eps(T) / 2
        @test maxerr(ly, logsoftmax(xd; dims = d)) <= tol
        @test maxerr(lse, logsumexp(xd; dims = d)) <= 64eps(Float32(log(V)))
        @test softmax!(copy(x); dims = d) == y
        @test logsoftmax!(copy(x); dims = d) == ly

        dx, ref = ∇softmax(dy, y; dims = d), ∇softmax(dyd, Float64.(y); dims = d)
        @test eltype(dx) == T
        @test maxerr(dx, ref) <= 2ulp(ref)
        dx, ref = ∇logsoftmax(dy, ly; dims = d), ∇logsoftmax(dyd, Float64.(ly); dims = d)
        @test eltype(dx) == T
        @test maxerr(dx, ref) <= 2ulp(ref)

        g = gradient(x -> sum(logsumexp(x; dims = d)), x)[1]
        @test eltype(g) == T
        @test maxerr(g, softmax(xd; dims = d)) <= ulp(softmax(xd; dims = d))
    end

    if T == Float16  # past 65504, a Float16 sum overflows
        V = 128256
        x = zeros(T, V, 1)
        @test logsumexp(x; dims = 1) == fill(log(Float32(V)), 1, 1)
        @test logsumexp(x) == log(Float32(V))
        @test logsoftmax(x) == fill(T(-log(V)), V, 1)
        @test sum(Float64, softmax(x)) ≈ 1 rtol = 3e-3  # 1/V is subnormal
        @test maxerr(∇logsoftmax(ones(T, V, 1), logsoftmax(x)), 0) < 1e-2
    end
end

@testset "AutoDiff" begin
    for f in (softmax, logsoftmax), d in (:, 1, 2)
        @test test_gradients(x -> f(x; dims = d), randn(rng, 3, 4))
    end
    @test test_gradients(x -> softmax(x) .* (1:3), randn(rng, 3))
    @test test_gradients(x -> softmax(x) .* (1:3), randn(rng, 3, 5))
    @test test_gradients(x -> softmax(x, dims = 2) .* (1:3), randn(rng, 3, 5))

    @test test_gradients(x -> logsoftmax(x) .* (1:3), randn(rng, 3))
    @test test_gradients(x -> logsoftmax(x) .* (1:3), randn(rng, 3, 5))
    @test test_gradients(x -> logsoftmax(x, dims = 2) .* (1:3), randn(rng, 3, 5))

    for d  in (:, 1, 2)
        @test test_gradients(x -> logsumexp(x; dims = d), randn(rng, 3, 4))
    end
end

@testset "Second derivatives" begin
    x = [1 2 3; 6 5 4]
    H = Zygote.hessian_dual(x -> sum(sin, softmax(x)), x)
    @test H ≈ Zygote.hessian_reverse(x -> sum(sin, softmax(x)), x)

    H2 = Zygote.hessian_dual(x -> sum(sin, logsoftmax(x)), x)
    @test H2 ≈ Zygote.hessian_reverse(x -> sum(sin, logsoftmax(x)), x)

    H3 = Zygote.hessian_dual(x -> sum(sin, logsumexp(x)), x)
    @test H3 ≈ Zygote.hessian_reverse(x -> sum(sin, logsumexp(x)), x)

    # Reverse-over-reverse goes through the rules for the gradients themselves.
    for d in (:, 1, 2, (1, 2))
        y, dy = softmax(randn(rng, 3, 4); dims = d), randn(rng, 3, 4)
        ChainRulesTestUtils.test_rrule(NNlib.∇softmax, dy, y; fkwargs = (; dims = d))
        ChainRulesTestUtils.test_rrule(NNlib.∇logsoftmax, dy, log.(y); fkwargs = (; dims = d))
    end
end

@testset "second-order rules, half precision: $T" for T in (Float16, BFloat16)
    # `dy ≈ 1`, as from `sum(sin, softmax(x))`, so the sums cancel against `dy`.
    hrng = StableRNG(815)
    V = 128256
    x = T.(0.01f0 .* randn(hrng, Float32, V, 2))
    dy = T.(1 .+ 0.01f0 .* randn(hrng, Float32, V, 2))
    Δ = T.(randn(hrng, Float32, V, 2))
    ulps(a, ref) = maximum(abs, Float64.(a) .- ref) / eps(T(maximum(abs, ref)))
    @testset "$∇f" for (∇f, y) in ((∇softmax, softmax(x)), (∇logsoftmax, logsoftmax(x)))
        _, ∂dy, ∂y = rrule(∇f, dy, y)[2](Δ)
        _, ref_dy, ref_y = rrule(∇f, Float64.(dy), Float64.(y))[2](Float64.(Δ))
        @test eltype(∂dy) == eltype(∂y) == T
        @test ulps(∂dy, ref_dy) <= 1
        @test ulps(∂y, ref_y) <= 1
    end
end
