 using Zygote

@testset "softmax" begin
    xs = rand(5,5)
    @test all(sum(softmax(xs), dims = 1) .≈ 1)
    @test all(sum(softmax(xs; dims=2), dims = 2) .≈ 1)
    @test sum(softmax(vec(xs))) ≈ 1
    @test log.(softmax(xs; dims=2)) ≈ logsoftmax(xs; dims=2)

    xs = [-100_000, -100_000.]
    @test softmax(xs) ≈ [0.5, 0.5]
    @test logsoftmax(xs) ≈ log.([0.5, 0.5])

    xs = rand(5)
    @test softmax(xs) ≈ exp.(xs) ./ sum(exp.(xs))
    @test logsoftmax(xs) ≈ log.(softmax(xs))

    xs = Float32[1, 2, 3000.]
    @test logsoftmax(xs) ≈ [-2999, -2998, 0]

    xs = Float32[1 2 3; 1000 2000 3000]
    @test logsoftmax(xs) ≈ [-999 -1998 -2997; 0 0 0.]

    @test NNlib.∇logsoftmax(ones(size(xs)), xs) ≈ Float32[1 1 1; -1 -1 -1]
    @test NNlib.∇softmax(ones(size(xs)), xs) ≈ zeros(Float32, size(xs))

    # These values precalculated using PyTorch's nn.LogSoftmax
    xs = [
        -0.238639  0.748142 -0.283194 -0.525461 -1.5348   -0.797842;
            0.690384  0.211427  0.254794 -0.213572 -0.314174 -0.372663;
        -1.146370 -0.577988  0.718952  0.919720 -0.620773  0.929977
    ]
    ys = [
        0.237703 -0.621474 0.448193 0.546047 0.564185 0.632273;
        -0.930163 0.0519798 0.0549979 0.3799 -0.477112 0.437428;
        0.69246 0.569494 -0.503191 -0.925947 -0.0870738 -1.0697
    ]
    @test isapprox(NNlib.∇logsoftmax(ones(size(xs)), xs), ys; rtol = 1e-6)
    @test isapprox(NNlib.∇softmax(ones(size(xs)), xs), zeros(size(xs)); atol = 1e-6)
end

@testset "mutating softmax" begin
    xs = Float64[1 2 3; 5 6 7]

    out = zeros(Float64, size(xs))
    NNlib.softmax!(out, xs)
    @test isapprox(out, softmax(xs); rtol=1e-6)
    NNlib.logsoftmax!(out, xs)
    @test isapprox(out, logsoftmax(xs); rtol=1e-6)

    out = ones(Float64, size(xs))
    NNlib.softmax!(out, xs)
    @test isapprox(out, softmax(xs); rtol=1e-6)
    NNlib.logsoftmax!(out, xs)
    @test isapprox(out, logsoftmax(xs); rtol=1e-6)

    out = zeros(Float64, size(xs))
    NNlib.∇softmax!(out, xs)
    @test isapprox(out, NNlib.∇softmax(zeros(size(xs)), xs); rtol=1e-6)
    out = zeros(Float64, size(xs))
    NNlib.∇logsoftmax!(out, xs)
    @test isapprox(out, NNlib.∇logsoftmax(zeros(size(xs)), xs); rtol=1e-6)

    out = ones(Float64, size(xs))
    NNlib.∇softmax!(out, xs)
    @test isapprox(out, NNlib.∇softmax(ones(size(xs)), xs); rtol=1e-6)
    out = ones(Float64, size(xs))
    NNlib.∇logsoftmax!(out, xs)
    @test isapprox(out, NNlib.∇logsoftmax(ones(size(xs)), xs); rtol=1e-6)

    xs = [
        -0.238639  0.748142 -0.283194 -0.525461 -1.5348   -0.797842;
            0.690384  0.211427  0.254794 -0.213572 -0.314174 -0.372663;
        -1.146370 -0.577988  0.718952  0.919720 -0.620773  0.929977
    ]

    out = zeros(Float64, size(xs))
    NNlib.softmax!(out, xs)
    @test isapprox(out, softmax(xs); rtol=1e-6)
    NNlib.logsoftmax!(out, xs)
    @test isapprox(out, logsoftmax(xs); rtol=1e-6)

    out = ones(Float64, size(xs))
    NNlib.softmax!(out, xs)
    @test isapprox(out, softmax(xs); rtol=1e-6)
    NNlib.logsoftmax!(out, xs)
    @test isapprox(out, logsoftmax(xs); rtol=1e-6)

    out = zeros(Float64, size(xs))
    NNlib.∇softmax!(out, xs)
    @test isapprox(out, NNlib.∇softmax(zeros(size(xs)), xs); rtol=1e-6)
    out = zeros(Float64, size(xs))
    NNlib.∇logsoftmax!(out, xs)
    @test isapprox(out, NNlib.∇logsoftmax(zeros(size(xs)), xs); rtol=1e-6)

    out = ones(Float64, size(xs))
    NNlib.∇softmax!(out, xs)
    @test isapprox(out, NNlib.∇softmax(ones(size(xs)), xs); rtol=1e-6)
    out = ones(Float64, size(xs))
    NNlib.∇logsoftmax!(out, xs)
    @test isapprox(out, NNlib.∇logsoftmax(ones(size(xs)), xs); rtol=1e-6)
end

@testset "logsumexp" begin
    flogsoft(x; dims) = mean(x .- logsoftmax(x; dims=dims), dims=dims) 
    
    x = rand(3,4)
    @test logsumexp(x) ≈ flogsoft(x, dims=:)
    @test logsumexp(x; dims=1) ≈ flogsoft(x, dims=1)
    @test gradient(x -> logsumexp(x), x)[1] ≈ gradient(x -> flogsoft(x, dims=:), x)[1]
end