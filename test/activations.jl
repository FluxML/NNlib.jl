
ACTIVATION_FUNCTIONS = [@eval($a) for a in NNlib.ACTIVATIONS]

BINARY_ACTIVATIONS = filter(f -> hasmethod(f, Tuple{Float64, Float64}), ACTIVATION_FUNCTIONS)

@test sigmoid(0.0) == 0.5
@test hardsigmoid(0.0) == 0.5
@test hardtanh(0.0) == 0.0
@test relu(0.0) == 0.0
@test leakyrelu(0.0) == 0.0
@test relu6(0.0) == 0.0
@test rrelu(0.0) == 0.0
@test elu(0.0) == 0.0
@test gelu(0.0) == 0.0
@test gelu_tanh(0.0) == 0.0
@test gelu_sigmoid(0.0) == 0.0
@test gelu_erf(0.0) == 0.0
@test swish(0.0) == 0.0
@test hardswish(0.0) == 0.0
@test lisht(0.0) == 0.0
@test softplus(0.0) ≈ log(2.0)
@test softplus(1e8) ≈ 1e8
@test softplus(-1e8) ≈ 0.0
@test softsign(0.0) == 0.0
@test selu(0.0) == 0.0
@test celu(0.0) == 0.0
@test trelu(0.0) == 0.0
@test logcosh(0.0) == log(cosh(0.0))
@test mish(0.0) == 0.0
@test tanhshrink(0.0) == 0.0
@test softshrink(0.0) == 0.0

@test sigmoid(1.0) == 1.0 / (1.0 + exp(-1.0))
@test hardsigmoid(1.0) == max(0,min(1, (1 + 3)/6))
@test hardtanh(1.0) == 1.0
@test relu(1.0) == 1.0
@test leakyrelu(1.0) == 1.0
@test relu6(1.0) == 1.0
@test rrelu(1.0) == 1.0
@test elu(1.0) == 1.0
@test gelu(1.0) ≈ 0.8411919906082768
@test gelu_tanh(1.0) ≈ 0.8411919906082768
@test gelu_sigmoid(1.0) ≈ 0.8411919906082768
@test gelu_erf(1.0) == 0.8413447460685429
@test swish(1.0) == sigmoid(1.0)
@test hardswish(1.0) == hardsigmoid(1.0)
@test lisht(1.0) ≈ 1.0 * tanh(1.0)
@test softplus(1.0) ≈ log(exp(1.0) + 1.0)
@test softsign(1.0) == 0.5
@test selu(1.0) == 1.0507009873554804934193349852946
@test celu(1.0) == 1.0
@test trelu(1.0) == 0.0
@test logcosh(1.0) ≈ log(cosh(1.0))
@test mish(1.0) ≈ tanh(log(1.0 + exp(1.0)))
@test tanhshrink(1.0) ≈ 0.23840584404423515
@test softshrink(1.0) == 0.5

@test sigmoid(-1.0) == exp(-1.0) / (1.0 + exp(-1.0))
@test hardsigmoid(-1.0) == max(0,min(1,(-1+3)/6 ))
@test hardtanh(-1.0) == -1.0
@test relu(-1.0) == 0.0
@test leakyrelu(-1.0) == -0.01
@test relu6(-1.0) == 0.0
@test -1/3.0 <= rrelu(-1.0) <= -1/8.0
@test elu(-1.0) == exp(-1.0) - 1.0
@test gelu(-1.0) ≈ -0.15880800939172324
@test gelu_tanh(-1.0) ≈ -0.15880800939172324
@test gelu_sigmoid(-1.0) ≈ -0.15880800939172324
@test gelu_erf(-1.0) == -0.15865525393145707
@test swish(-1.0) == -sigmoid(-1.0)
@test hardswish(-1.0) == -hardsigmoid(-1.0)
@test lisht(-1.0) ≈ -1.0 * tanh(-1.0)
@test softplus(-1.0) ≈ log(exp(-1.0) + 1.0)
@test softsign(-1.0) == -0.5
@test selu(-1.0) ≈ 1.0507009873554804934193349852946 * 1.6732632423543772848170429916717 * (exp(-1.0) - 1.0)
@test celu(-1.0) == exp(-1.0) - 1
@test trelu(-1.0) == 0.0
@test log(cosh(-1.0)) ≈ log(cosh(-1.0))
@test mish(-1.0) ≈ -tanh(log(1.0 + exp(-1.0)))
@test tanhshrink(-1.0) ≈ -0.23840584404423515
@test softshrink(-1.0) == -0.5

@testset "Float inference" begin
    @testset "$(a): " for a in ACTIVATION_FUNCTIONS
        for T in [Float16, Float32, Float64]
            for val in [-10, -1, 0, 1, 10]
                out = @inferred a(T(val))
                @test typeof(out) == T
            end
        end
    end
    @testset "binary $a: " for a in BINARY_ACTIVATIONS
        for T in [Float16, Float32, Float64]
            for val in [-10, -1, 0, 1, 10], beta in Any[0.1, 0.5f0, 1]
                out = @inferred a(T(val), beta)
                @test typeof(out) == T
            end
        end
    end
end

@testset "Array input -> error" begin
    x = rand(5)
    for a in ACTIVATION_FUNCTIONS
        @test size(a(x)) == size(x)
        grad = Zygote.gradient(p -> sum(a(p)), x)
        @test size(grad[1]) == size(x)
    end
    for a in BINARY_ACTIVATIONS
        @test size(a(x, 0.1)) == size(x)
        grad = Zygote.gradient(p -> sum(a(p, 0.1)), x)
        @test size(grad[1]) == size(x)
    end
end

@testset "NaN propagation" begin
    @testset "$a" for a in ACTIVATION_FUNCTIONS
        # With NaN input, all should produce NaN output:
        @test isnan(a(NaN32))

        # Ideally +-Inf would not lead to NaN, but perhaps
        # these aren't worth the complication of fixing:
        a == softsign && continue
        @test !isnan(a(Inf32))

        a in [gelu, gelu_tanh, gelu_sigmoid, gelu_erf, swish, hardswish, logcosh, mish] && continue
        @test !isnan(a(-Inf32))
    end
end

@testset "Integer inputs" begin
    # These should work without error, for e.g. readme examples,
    # but no serious use will involve integers, no need for performance.
    @testset "$a" for a in ACTIVATION_FUNCTIONS
        @test typeof(a(Int64(1))) <: Real
        @test typeof(a(Int32(1))) <: Real
    end

    # The following ones can pass integers through. But it's not very important.
    @testset "relu: Int -> Int" begin
        @test typeof(relu(Int64(1))) == Int64
        @test typeof(relu(Int32(1))) == Int32
    end
    @testset "relu6: Int -> Int" begin
        @test typeof(relu6(Int64(1))) == Int64
        @test typeof(relu6(Int32(1))) == Int32
    end
    @testset "hardtanh: Int -> Int" begin
        @test typeof(hardtanh(Int64(1))) == Int64
        @test typeof(hardtanh(Int32(1))) == Int32
    end
    @testset "trelu: Int -> Int" begin
        @test typeof(trelu(Int64(1))) == Int64
        @test typeof(trelu(Int32(1))) == Int32
    end
end

@testset "elu" begin
    @test elu(42) == 42
    @test elu(42.) == 42.

    @test elu(-4) ≈ (exp(-4) - 1)
end

@testset "mish" begin
    @test mish(-5) ≈ -0.033576237730161704
    @test mish(9) == 9*tanh(log(1 + exp(9)))
    xs = Float32[1 2 3; 1000 2000 3000]
    @test typeof(mish.(xs)) == typeof(xs)
end

@test leakyrelu( 0.4,0.3) ≈  0.4
@test leakyrelu(-0.4,0.3) ≈ -0.12

@test relu6(10.0) == 6.0

@test -0.2 <= rrelu(-0.4,0.25,0.5) <= -0.1

@testset "celu" begin
    @test celu(42) == 42
    @test celu(42.) == 42.

    @test celu(-4, 0.5) ≈ 0.5*(exp(-4.0/0.5) - 1)
end

@testset "softshrink" begin
    @test softshrink(15., 5.) == 10.
    @test softshrink(4., 5.) == 0.
    @test softshrink(-15., 5.) == -10.
end

@testset "logsigmoid" begin
    xs = randn(10,10)
    @test logsigmoid.(xs) ≈ log.(sigmoid.(xs))
    for T in [:Float32, :Float64]
        @eval @test logsigmoid.($T[-100_000, 100_000.]) ≈ $T[-100_000, 0.]
    end
end

@test logcosh(1_000.0) + log(2) == 1_000.0

@testset "hardsigmoid" begin
    @test hardsigmoid(0.3) == max(0,min(1,(0.3+3)/6))
    @test hardsigmoid(-0.3) == max(0,min(1,(-0.3+3)/6))
    for T in [:Float32, :Float64]
        @eval @test hardsigmoid.($T[-100_000, 100_000.]) ≈ $T[0., 1.]
    end
end

@test hardtanh(10.0) == 1.0

@test lisht(2.5) == 2.5*tanh(2.5)

@testset "trelu" begin
    @test trelu(0.5) == 0.0
    @test trelu(1.0) == 0.0
    @test trelu(1.1) == 1.1
    @test trelu(0.9,0.5) == 0.9
end

## Faster variants

using NNlib: tanh_fast, sigmoid_fast

function countepsfrom(x::T, xtrue) where {T<:AbstractFloat}
    target = T(xtrue)
    for n in Iterators.flatten(zip(0:100, -1:-1:-100))
        nextfloat(x, n) === target && return n
    end
    return round(Int, (target - x) / eps(x))
end

mean_eps(f, g, xs) = mean(x -> abs(countepsfrom(f(x), g(big(x)))), xs)
worst_eps(f, g, xs) = maximum(x -> abs(countepsfrom(f(x), g(big(x)))), xs)
function find_worst(f, g, xs)
    c, i = findmax(x -> abs(countepsfrom(f(x), g(big(x)))), xs)
    c, xs[i]
end

@testset "tanh_fast & sigmoid_fast: Float64" begin
    
    x64 = 1e-6:1e-4:5
    xbig = vcat(6:3:200.0, 1000, 10^6, typemax(Float64))
    
    @testset "tanh" begin
        mean_eps(tanh, tanh, x64)  # 0.06582
        worst_eps(tanh, tanh, x64) # 2

        @test mean_eps(tanh_fast, tanh, x64) < 0.2  # 0.13164
        @test worst_eps(tanh_fast, tanh, x64) <= 5  # 5

        @test mean_eps(tanh_fast, tanh, -x64) < 0.6 # 0.5248
        @test worst_eps(tanh_fast, tanh, -x64) <= 5 # 5

        @test tanh_fast.(xbig) ≈ tanh.(xbig)
        @test tanh_fast.(-xbig) ≈ tanh.(-xbig)
    end
    @testset "sigmoid" begin
        mean_eps(sigmoid, sigmoid, x64)  # 0.39246
        worst_eps(sigmoid, sigmoid, x64) # 1

        @test mean_eps(sigmoid_fast, sigmoid, x64) < 0.5  # 0.40432
        @test worst_eps(sigmoid_fast, sigmoid, x64) <= 5  # 2

        mean_eps(sigmoid, sigmoid, -x64)  # 0.37672
        worst_eps(sigmoid, sigmoid, -x64) # 2

        @test mean_eps(sigmoid_fast, sigmoid, -x64) < 0.6  # 0.56478
        @test worst_eps(sigmoid_fast, sigmoid, -x64) <= 5  # 4

        @test sigmoid_fast.(xbig) ≈ sigmoid.(xbig)
        @test sigmoid_fast.(-xbig) ≈ sigmoid.(-xbig)
    end
end

@testset "tanh_fast & sigmoid_fast: Float32" begin
    
    x32 = 1f-6:1f-4:5
    xbig32 = vcat(6:3:200f0, 1000, typemax(Float32))

    @testset "tanh" begin
        mean_eps(tanh, tanh, x32)  # 0.065
        worst_eps(tanh, tanh, x32) # 1

        @test mean_eps(tanh_fast, tanh, x32) < 0.8  # 0.65414
        @test worst_eps(tanh_fast, tanh, x32) <= 5  # 5

        @test mean_eps(tanh_fast, tanh, -x32) < 0.8 # 0.65414
        @test worst_eps(tanh_fast, tanh, -x32) <= 5 # 5

        @test tanh_fast.(xbig32) ≈ tanh.(xbig32)
        @test tanh_fast.(-xbig32) ≈ tanh.(-xbig32)
    end
    @testset "sigmoid" begin
        mean_eps(sigmoid, sigmoid, x32)  # 0.38896
        worst_eps(sigmoid, sigmoid, x32) # 1

        @test mean_eps(sigmoid_fast, sigmoid, x32) < 0.5  # 0.38896
        @test worst_eps(sigmoid_fast, sigmoid, x32) <= 2  # 2

        mean_eps(sigmoid, sigmoid, -x32)  # 0.38088
        worst_eps(sigmoid, sigmoid, -x32) # 2

        @test mean_eps(sigmoid_fast, sigmoid, -x32) < 0.5  # 0.38088
        @test worst_eps(sigmoid_fast, sigmoid, -x32) <= 2  # 2

        @test sigmoid_fast.(xbig32) ≈ sigmoid.(xbig32)
        @test sigmoid_fast.(-xbig32) ≈ sigmoid.(-xbig32)
    end
end

## Autodiff tests

WITH_UNARY_RULE = [@eval($a) for (a, _) in NNlib.UNARY_ACTS]

WITH_BINARY_RULE = [@eval($a) for (a, _, _) in NNlib.BINARY_ACTS]

has_rule(a) = rrule(a, 1f0) === nothing ? "(no rule)" : ""

@testset "Gradient inference" begin
    @testset "$(a): $(has_rule(a))" for a in ACTIVATION_FUNCTIONS
        @testset "$T" for T in [Float16, Float32, Float64]
            for val in [-10, -1, 0, 1, 10]
                grad = @inferred gradient(a, T(val))
                @test typeof(grad[1]) == T
            end
        end
    end
end

using Base.Broadcast: broadcasted

@testset "lazy broadcasting" begin
    # ChainRules returns a Broadcasted, check these rules accept it
    @test rrule(broadcasted, relu, rrule(broadcasted, +, [1,2], 3)[1]) != nothing
    @test rrule(broadcasted, leakyrelu, rrule(broadcasted, +, [1,2], 3)[1], 0.2) != nothing
end

@testset "Gradient correctness" begin
    
    local rng = StableRNG(17)

    @testset "$(f): $(has_rule(f))" for f in ACTIVATION_FUNCTIONS
        f == rrelu && continue # stocastich output
        
        ## Avoid singular points of some activations
        ## problematic for finite diff methods
        gradtest(f, +2 + rand(rng))
        gradtest(f, -2 - rand(rng))
        gradtest(f, +2 .+ rand(rng, 2, 2), check_broadcast=true)
        gradtest(f, -2 .- rand(rng, 2, 2), check_broadcast=true)

        if f in BINARY_ACTIVATIONS
            gradtest(x -> f(x, 0.2), 1 + rand(rng))
            gradtest(x -> f(x, 0.7), 1 + rand(rng))

            gradtest(x -> f(x, 0.2), -2 + rand(rng))
            gradtest(x -> f(x, 0.7), -2 + rand(rng))
        end

        ## Check that rules, including broadcast rules, are defined:
        if f in WITH_UNARY_RULE
            @test rrule(f, rand()) !== nothing
            @test rrule(broadcasted, f, rand(2)) !== nothing
        end
        if f in WITH_BINARY_RULE
            @test rrule(f, rand(), rand()) !== nothing
            @test rrule(broadcasted, f, rand(2), rand()) !== nothing
        end
    end 
    
    @testset "Flux-like usage" begin
        ## This checks some broadcast rules for correctness:
        gradtest((x, W, b) -> σ.(W*x .+ b), 5, (2,5), 2)
        gradtest((x, W, b) -> σ.(W*x .+ b), (5,3), (2,5), 2)
        gradtest((x, W, b) -> relu.(W*x .+ b), 5, (2,5), 2)
        gradtest((x, W, b) -> relu.(W*x .+ b), (5,3), (2,5), 2)
        gradtest((x, W, b) -> selu.(W*x .+ b), 5, (2,5), 2)
        gradtest((x, W, b) -> selu.(W*x .+ b), (5,3), (2,5), 2, atol=1e-4)
        gradtest((x, W, b) -> elu.(W*x .+ b, 2), 5, (2,5), 2)
        gradtest((x, W, b) -> elu.(W*x .+ b, 2), (5,3), (2,5), 2, atol=1e-4)

        gradtest((x, W, b) -> logσ.(W*x .+ b), 5, (2,5), 2)
        gradtest((x, W, b) -> logσ.(W*x .+ b), (5,3), (2,5), 2)

        ## Binary functions have their own broadcast rules:
        gradtest((x, W, b) -> leakyrelu.(W*x .+ b, 0.2), 5, (2,5), 2)
        gradtest((x, W, b) -> leakyrelu.(W*x .+ b, 0.7), (5,3), (2,5), 2)
    end

    @testset "Zygote issue 758" begin
        ## Tests for https://github.com/FluxML/Zygote.jl/issues/758
        @test gradient(xs -> sum(selu.(xs)), [1_000, 10_000])[1] ≈ [1.0507009873554805, 1.0507009873554805] rtol=1e-8
        @test gradient(x -> selu(x), 1_000) == (1.0507009873554805,)
        @test gradient(xs -> sum(elu.(xs, 2)), [1_000, 10_000]) == ([1., 1.],)
        @test gradient(x -> elu(x, 2), 1_000) == (1.,)
        @test gradient(x -> elu(x, 2), -1) == (2*exp(-1),)
        gradtest(x-> selu.(x),[100., 1_000.])
        gradtest(x -> elu.(x, 3.5),[100., 1_000.])
        gradtest(x -> elu.(x, 3.5),[1_000., 10_000.])
        gradtest(x -> selu.(x), [1_000., 10_000.])
        gradtest(x -> selu.(x), 10, atol=1e-4)
    end

end

@testset "Second derivatives" begin
    ## Not extensive, but a start!
    ## More careful tests could look for `nothing` gradients of piecewise functions
    @testset "$(f): $(has_rule(f))" for f in ACTIVATION_FUNCTIONS
        f == rrelu && continue

        ## Scalar
        h = Zygote.hessian_dual(x -> sin(f(x)), 0.23)
        @test h ≈ Zygote.hessian_reverse(x -> sin(f(x)), 0.23)

        ## Broadcasting
        x = [-0.9, -0.2, 0.1, 0.3, 1.2]
        H = Zygote.hessian_dual(x -> sum(abs2, f.(x .+ 0.1)), x)
        @test H ≈ Zygote.hessian_reverse(x -> sum(abs2, f.(x .+ 0.1)), x)
    end
    @testset "$(f): $(has_rule(f))" for f in BINARY_ACTIVATIONS
        f == rrelu && continue

        ## Scalar
        h = Zygote.hessian_dual(x -> sin(f(x, 0.3)), 0.45)
        @test h ≈ Zygote.hessian_reverse(x -> sin(f(x, 0.3)), 0.45)

        ## Broadcasting
        x = [-0.9, -0.2, 0.1, 0.3, 1.2]
        H = Zygote.hessian_dual(x -> sum(abs2, f.(x .+ 0.1, 0.3)), x)
        @test H ≈ Zygote.hessian_reverse(x -> sum(abs2, f.(x .+ 0.1, 0.3)), x)
    end
end
