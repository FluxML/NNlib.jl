using Statistics: mean
using NNlib: ∇softmax_data, ∇logsoftmax_data

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
    @test ∇logsoftmax_data(ones(Float32, size(xs)), y) ≈ Float32[1 1 1; -1 -1 -1]
    
    y = softmax(xs)
    @test ∇softmax_data(ones(Float32, size(xs)), y) ≈ zeros(Float32, size(xs))

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
    @test ∇logsoftmax_data(ones(size(xs)), y) ≈ ys rtol = 1e-6
    
    y = softmax(xs)
    @test ∇softmax_data(ones(size(xs)), y) ≈ zeros(size(xs)) atol = 1e-6
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
            ∇softmax!(out, Δ, xs, y)  # deprecated
            @test out ≈ ∇softmax_data(Δ, y)  rtol = 1e-6
            
            y = logsoftmax(xs)
            ∇logsoftmax!(out, Δ, xs, y)  # deprecated
            @test out ≈ ∇logsoftmax_data(Δ, y)  rtol = 1e-6
        end
    end
end

@testset "logsumexp" begin
    flogsoft(x; dims) = mean(x .- logsoftmax(x; dims = dims), dims = dims)

    x = rand(3, 4)
    @test logsumexp(x) ≈ flogsoft(x, dims = :)
    @test logsumexp(x; dims = 1) ≈ flogsoft(x, dims = 1)
end

@testset "AutoDiff" begin
    for f in (softmax, logsoftmax), d in (:, 1, 2)
        gradtest(f, (3,4); fkwargs = (dims = d,), check_rrule = true)
    end
    gradtest(x -> softmax(x) .* (1:3), 3)
    gradtest(x -> softmax(x) .* (1:3), (3,5), atol = 1e-4)
    gradtest(x -> softmax(x, dims = 2) .* (1:3), (3,5), atol = 1e-4)

    gradtest(x -> logsoftmax(x) .* (1:3), 3)
    gradtest(x -> logsoftmax(x) .* (1:3), (3,5))
    gradtest(x -> logsoftmax(x, dims = 2) .* (1:3), (3,5))

    for d  in (:, 1, 2)
        gradtest(logsumexp, (3,4), fkwargs = (dims = d,))
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
end
