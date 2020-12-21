using Zygote
using Statistics: mean

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
    @test ∇logsoftmax(ones(Float32, size(xs)), xs, y) ≈ Float32[1 1 1; -1 -1 -1]
    
    y = softmax(xs)
    @test ∇softmax(ones(Float32, size(xs)), xs, y) ≈ zeros(Float32, size(xs))

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
    @test ∇logsoftmax(ones(size(xs)), xs) ≈ ys rtol = 1e-6
    @test ∇softmax(ones(size(xs)), xs) ≈ zeros(size(xs)) atol = 1e-6
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

        map([zeros, ones]) do fn
            Δ = fn(Float64, size(xs))
            y = softmax(xs) 
            ∇softmax!(out, Δ, xs, y)
            @test out ≈ ∇softmax(Δ, xs, y)  rtol = 1e-6
            
            y = logsoftmax(xs)
            ∇logsoftmax!(out, Δ, xs, y)
            @test out ≈ ∇logsoftmax(Δ, xs, y)  rtol = 1e-6
        end
    end
end

@testset "logsumexp" begin
    flogsoft(x; dims) = mean(x .- logsoftmax(x; dims = dims), dims = dims)

    x = rand(3, 4)
    @test logsumexp(x) ≈ flogsoft(x, dims = :)
    @test logsumexp(x; dims = 1) ≈ flogsoft(x, dims = 1)
    @test gradient(x -> logsumexp(x), x)[1] ≈ gradient(x -> flogsoft(x, dims = :), x)[1]
end
