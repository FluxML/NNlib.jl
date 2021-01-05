using NNlib, Test, Zygote
using NNlib: ACTIVATIONS

ACTIVATION_FUNCTIONS = 
    [@eval($a) for a in ACTIVATIONS]

function test_value_float_precision_preserving(a)
    @testset "$(a): " begin
        for T in [Float16, Float32, Float64]
            for val in [-10, -1, 0, 1, 10]
                val = @inferred a(T(val))
                @test typeof(val) == T
            end
        end
    end
end

function test_value_int_input_forces_float64(a)
    @testset "$(a): " begin
        for T in [Int32, Int64]
            for val in [-10, -1, 0, 1, 10]
                val = @inferred a(T(val))
                @test typeof(val) == Float64
            end
        end
    end
end

function test_gradient_float_precision_preserving(a)
    @testset "$(a): " begin
        for T in [Float16, Float32, Float64]
            for val in [-10, -1, 0, 1, 10]
                val = @inferred a'(T(val))
                @test typeof(val) == T
            end
        end
    end
end

@test sigmoid(0.0) == 0.5
@test hardsigmoid(0.0) == 0.5
@test hardtanh(0.0) == 0.0
@test relu(0.0) == 0.0
@test leakyrelu(0.0) == 0.0
@test relu6(0.0) == 0.0
@test rrelu(0.0) == 0.0
@test elu(0.0) == 0.0
@test gelu(0.0) == 0.0
@test swish(0.0) == 0.0
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
@test gelu(1.0) == 0.8411919906082768
@test swish(1.0) == sigmoid(1.0)
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
@test gelu(-1.0) == -0.15880800939172324
@test swish(-1.0) == -sigmoid(-1.0)
@test lisht(-1.0) ≈ -1.0 * tanh(-1.0)
@test softplus(-1.0) ≈ log(exp(-1.0) + 1.0)
@test softsign(-1.0) == -0.5
@test selu(-1.0) == 1.0507009873554804934193349852946 * 1.6732632423543772848170429916717 * (exp(-1.0) - 1.0)
@test celu(-1.0) == exp(-1.0) - 1
@test trelu(-1.0) == 0.0
@test log(cosh(-1.0)) ≈ log(cosh(-1.0))
@test mish(-1.0) ≈ -tanh(log(1.0 + exp(-1.0)))
@test tanhshrink(-1.0) ≈ -0.23840584404423515
@test softshrink(-1.0) == -0.5

@testset "Float inference" begin
    test_value_float_precision_preserving.(ACTIVATION_FUNCTIONS)
end

@testset "Array input" begin
    x = rand(5)
    for a in ACTIVATION_FUNCTIONS
        @test_throws ErrorException a(x)
    end
end

@testset "Test Integer64 and Integer32 inputs will force Float64 outputs" begin
    test_value_int_input_forces_float64.(filter(x -> (x != relu && x != relu6 && x != hardtanh && x != trelu), ACTIVATION_FUNCTIONS))

    @testset "relu: " begin
        # relu doesn't have to force floating point outputs
        @test typeof(relu(Int64(1))) == Int64
        @test typeof(relu(Int32(1))) == Int32
    end

    @testset "relu6: " begin
        # relu6 doesn't have to force floating point outputs
        @test typeof(relu6(Int64(1))) == Int64
        @test typeof(relu6(Int32(1))) == Int32
    end

    @testset "hardtanh: " begin
        # hardtanh doesn't have to force floating point outputs
        @test typeof(hardtanh(Int64(1))) == Int64
        @test typeof(hardtanh(Int32(1))) == Int32
    end

    @testset "trelu: " begin
        # trelu doesn't have to force floating point outputs
        @test typeof(trelu(Int64(1))) == Int64
        @test typeof(trelu(Int32(1))) == Int32
    end
end

@testset "Float gradient inference" begin
    test_gradient_float_precision_preserving.(ACTIVATION_FUNCTIONS)
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

@testset "AutoDiff" begin
    
    local rng = StableRNG(17)

    for f in ACTIVATION_FUNCTIONS
        f == rrelu && continue # stocastich output
        
        # gradtest(f, rand(rng))
        # gradtest(f, (2, 3), check_broadcast=true)
        
        ## Avoid singular points of some activations
        ## problematic for finite diff methods
        gradtest(f, +2 + rand(rng))
        gradtest(f, -2 - rand(rng))
        gradtest(f, +2 .+ rand(rng, 2, 2), check_broadcast=true)
        gradtest(f, -2 .- rand(rng, 2, 2), check_broadcast=true)

        if Symbol(f) in NNlib.UNARY_ACTS
            @test rrule(f, rand()) !== nothing
            @test rrule(broadcasted, f, rand(2)) !== nothing
        end
        if Symbol(f) in NNlib.BINARY_ACTS
            @test rrule(f, rand(), rand()) !== nothing
            @test rrule(broadcasted, f, rand(2), rand(2)) !== nothing
        end
    end 
    
    gradtest((x, W, b) -> σ.(W*x .+ b), 5, (2,5), 2)
    gradtest((x, W, b) -> σ.(W*x .+ b), (5,3), (2,5), 2)
    gradtest((x, W, b) -> relu.(W*x .+ b), 5, (2,5), 2)
    gradtest((x, W, b) -> relu.(W*x .+ b), (5,3), (2,5), 2)
    gradtest((x, W, b) -> selu.(W*x .+ b), 5, (2,5), 2)
    gradtest((x, W, b) -> selu.(W*x .+ b), (5,3), (2,5), 2, atol=1e-4)
    gradtest((x, W, b) -> elu.(W*x .+ b, 2), 5, (2,5), 2)
    gradtest((x, W, b) -> elu.(W*x .+ b, 2), (5,3), (2,5), 2, atol=1e-4)

    # tests for https://github.com/FluxML/Zygote.jl/issues/758
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

    gradtest((x, W, b) -> σ.(W*x .+ b), 5, (2,5), 2)
    gradtest((x, W, b) -> σ.(W*x .+ b), (5,3), (2,5), 2)
    gradtest((x, W, b) -> logσ.(W*x .+ b), 5, (2,5), 2)
    gradtest((x, W, b) -> logσ.(W*x .+ b), (5,3), (2,5), 2)
end
