
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
# https://github.com/FluxML/NNlib.jl/issues/671: softplus must be smooth at the
# origin so that derivatives (computed e.g. by ForwardDiff for Hessians) are correct there.
@test ForwardDiff.derivative(softplus, 0.0) ≈ 0.5
@test ForwardDiff.derivative(x -> ForwardDiff.derivative(softplus, x), 0.0) ≈ 0.25
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
        for T in [Float16, BFloat16, Float32, Float64]
            for val in [-10, -1, 0, 1, 10]
                out = @inferred a(T(val))
                @test typeof(out) == T
            end
        end
    end
    @testset "binary $a: " for a in BINARY_ACTIVATIONS
        for T in [Float16, BFloat16, Float32, Float64]
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

## Vectorizable kernels

@testset "_exp, _expm1 & _log1p_unit: $T" for T in [Float32, Float64]
    _exp, _expm1, _log1p_unit = NNlib._exp, NNlib._expm1, NNlib._log1p_unit

    @testset "_exp" begin
        xs = T.(-40:0.0137:40)
        @test worst_eps(_exp, exp, xs) <= 1  # 1, as for Base's exp
        @test mean_eps(_exp, exp, xs) < 0.2  # 0.061 for Float32, 0.099 for Float64

        # Subnormal results, down to where they round to zero
        xsub = T.(range(log(nextfloat(zero(T))) - 2, log(floatmin(T)), length=2001))
        @test worst_eps(_exp, exp, xsub) <= 1
        @test issubnormal(_exp(log(floatmin(T)) - 3))

        # Across the overflow threshold
        xtop = T.(range(log(floatmax(T)) - 2, log(floatmax(T)) + 2, length=2001))
        @test worst_eps(_exp, exp, xtop) <= 1

        @test _exp(zero(T)) === one(T)
        @test _exp(T(Inf)) === T(Inf)
        @test _exp(floatmax(T)) === T(Inf)
        @test _exp(T(-Inf)) === zero(T)
        @test _exp(-floatmax(T)) === zero(T)
        @test isnan(_exp(T(NaN)))
    end

    @testset "_expm1" begin
        # Down to 4e-18, where `exp(x) - 1` is zero
        xs = vcat(T.(-30:0.0137:0), -T.(exp.(-(0:0.01:40))))
        @test worst_eps(_expm1, expm1, xs) <= 1   # 1, as for Base's expm1
        @test worst_eps(_expm1, expm1, -xs) <= 2  # 1, but 2 for some x in 0.35..0.7

        xtop = T.(range(log(floatmax(T)) - 2, log(floatmax(T)) + 2, length=2001))
        @test worst_eps(_expm1, expm1, xtop) <= 1

        @test _expm1(zero(T)) === zero(T)
        @test _expm1(nextfloat(zero(T))) === nextfloat(zero(T))
        @test _expm1(T(Inf)) === T(Inf)
        @test _expm1(floatmax(T)) === T(Inf)
        @test _expm1(T(-Inf)) === -one(T)
        @test _expm1(-floatmax(T)) === -one(T)
        @test isnan(_expm1(T(NaN)))
    end

    @testset "_log1p_unit" begin
        ts = vcat(T.(0:1e-4:1), T.(exp.(-(0:0.01:40))))
        @test worst_eps(_log1p_unit, log1p, ts) <= 1  # 1, as for Base's log1p
        @test mean_eps(_log1p_unit, log1p, ts) < 0.2  # 0.059

        @test _log1p_unit(zero(T)) === zero(T)
        @test _log1p_unit(one(T)) === log(T(2))
        @test isnan(_log1p_unit(T(NaN)))
    end

    @testset "activations" begin
        xs = T.(-30:0.0173:30)
        @test worst_eps(softplus, softplus, xs) <= 2  # 1
        @test mean_eps(softplus, softplus, xs) < 0.3  # 0.097 for Float32, 0.143 for Float64
        @test worst_eps(mish, mish, xs) <= 4          # 3
        @test mean_eps(mish, mish, xs) < 0.6          # 0.284 for Float32, 0.448 for Float64

        # Small negative inputs, for which `exp(x) - 1` has no correct digit
        xs = vcat(T.(-30:0.0173:0), -T.(exp.(-(0:0.01:40))))
        @test worst_eps(elu, elu, xs) <= 1    # 1
        @test worst_eps(celu, celu, xs) <= 1  # 1
        @test worst_eps(selu, selu, xs) <= 3  # 2
        @test elu(-eps(T) / 4) === -eps(T) / 4  # was zero
    end
end

@testset "kernels fall back to Base" begin
    @test NNlib._exp(1) === exp(1)
    @test NNlib._exp(Float16(1)) === exp(Float16(1))
    @test NNlib._exp(big(1.0)) == exp(big(1.0))
    @test NNlib._exp(1.0 + 2.0im) === exp(1.0 + 2.0im)
    @test NNlib._expm1(1) === expm1(1)
    @test NNlib._expm1(Float16(1)) === expm1(Float16(1))
    @test NNlib._expm1(big(1.0)) == expm1(big(1.0))
end

# A loop over `f` is compiled to SIMD code only if `f` is inlined and free of branches.
# Pointers keep bounds checks out of the loop, whatever `--check-bounds` says.
function simd_loop!(f::F, y::Ptr{T}, x::Ptr{T}, n::Int) where {F,T}
    @simd for i in 1:n
        unsafe_store!(y, f(unsafe_load(x, i)), i)
    end
end

function is_vectorized(f, T)
    ir = sprint() do io
        InteractiveUtils.code_llvm(io, simd_loop!, Tuple{typeof(f),Ptr{T},Ptr{T},Int}; debuginfo=:none)
    end
    # Float16 is computed in Float32
    return occursin(Regex("<\\d+ x $(T === Float64 ? "double" : "float")>"), ir)
end

@testset "SIMD" begin
    # The counters of code coverage sit in the loop, which is then not vectorized.
    if Sys.ARCH in (:x86_64, :aarch64) && Base.JLOptions().code_coverage == 0
        @testset "$f" for f in [σ, sigmoid_fast, tanh_fast, elu, selu, celu, gelu_tanh, gelu_sigmoid,
                                swish, softplus, logσ, logcosh, mish, lisht, tanhshrink]
            @test is_vectorized(f, Float32)
            @test is_vectorized(f, Float64)
            @test is_vectorized(f, Float16)
        end
        @test !is_vectorized(exp, Float32)  # the reason for `_exp`
    end
end

# `act ∘ +` is what `bias_act!` broadcasts. A call to it that is not inlined boxes its arguments.
composed_allocs(f, y, xs...) = (broadcast!(f, y, xs...); @allocated broadcast!(f, y, xs...))

@testset "composition with ∘" begin
    x, b = randn(Float32, 100), randn(Float32, 100)
    y = similar(x)
    @testset "$f" for f in ACTIVATION_FUNCTIONS
        @test composed_allocs(f ∘ +, y, x, b) < 100  # a box per element would be 1600 bytes or more
        f == rrelu || @test y == f.(x .+ b)
        @test composed_allocs(abs ∘ f, y, x) < 100
        f == rrelu || @test y == abs.(f.(x))
    end
    @test (σ ∘ relu)(1.0) === σ(1.0)
    @test (σ ∘ +)(1, 2) === σ(3)
    @test (σ ∘ (x -> sum(x; dims=1)))([1.0 2.0; 3.0 4.0]) == σ.([4.0 6.0])  # not only numbers
end

## Half precision

# The activations that are computed in Float32 for Float16 and BFloat16 inputs
HALF_ACTIVATIONS = [σ, hardσ, logσ, leakyrelu, elu, gelu_tanh, gelu_sigmoid, gelu_erf, swish, hardswish,
                    lisht, selu, celu, softsign, softplus, logcosh, mish, tanhshrink, tanh_fast, sigmoid_fast]

@testset "half precision: $T" for T in [Float16, BFloat16]
    # The Float32 result rounded once, for every input there is. One number at a time:
    # a vectorized rounding to BFloat16 is not safe on all CPUs (JuliaMath/BFloat16s.jl#107).
    @testset "$f" for f in HALF_ACTIVATIONS
        ok = all(0x0000:0xffff) do u
            x = reinterpret(T, u)
            isequal(f(x), T(f(Float32(x))))
        end
        @test ok
    end

    @testset "two arguments" begin
        for x in (T(-2), T(-0.5f0), T(0), T(3)), α in Any[0.3, 0.3f0, 2, T(0.5f0)]
            @test elu(x, α) === T(elu(Float32(x), α))
            @test celu(x, α) === T(celu(Float32(x), α))
            @test leakyrelu(x, α) === T(leakyrelu(Float32(x), α))
        end
    end
end

@testset "half precision accuracy" begin
    # One in seven of the Float16s. A rounded Float32 result is nearly always the correctly
    # rounded one, except where the Float32 formula itself cancels.
    xs = filter(isfinite, reinterpret.(Float16, 0x0000:0x0007:0xffff))
    @testset "$f" for f in HALF_ACTIVATIONS
        errs = [abs(countepsfrom(f(x), f(big(x)))) for x in xs]
        @test maximum(errs) <= (f == gelu_tanh ? 8 : f == gelu_erf ? 3 : 1)  # 6, 2 and 1
        f in (gelu_tanh, gelu_erf, logcosh, tanhshrink) || @test mean(errs) < 1e-3  # 3e-4 at most
    end

    # The derivatives that are a function of the input are computed in Float32 as well.
    # Those written in terms of the output (`σ`, `tanh_fast`, `elu`, `selu`, `celu`) are not.
    @testset "derivative of $f" for f in [hardσ, logσ, leakyrelu, gelu_sigmoid, gelu_erf, swish, hardswish,
                                          lisht, softsign, softplus, logcosh, mish, tanhshrink]
        xd = filter(x -> x != 0 && abs(x) != 3, xs)  # not at the kinks
        @test maximum(x -> abs(countepsfrom(gradient(f, x)[1], ForwardDiff.derivative(f, big(x)))), xd) <= 1
    end
end

## Autodiff tests

WITH_UNARY_RULE = [@eval($a) for (a, _) in NNlib.UNARY_ACTS]

WITH_BINARY_RULE = [@eval($a) for (a, _, _) in NNlib.BINARY_ACTS]

has_rule(a) = rrule(a, 1f0) === nothing ? "(no rule)" : ""

@testset "Gradient inference" begin
    @testset "$(a): $(has_rule(a))" for a in ACTIVATION_FUNCTIONS
        @testset "$T" for T in [Float16, BFloat16, Float32, Float64]
            for val in [-10, -1, 0, 1, 10]
                grad = @inferred gradient(a, T(val))
                @test typeof(grad[1]) == T
            end
        end
    end
end

@testset "AD through the vectorizable kernels: $T" for T in [Float32, Float64]
    # Enzyme and Mooncake have no rules for the activations: they differentiate their code,
    # and so the kernels for `Float32` and `Float64`. ForwardDiff takes the generic methods.
    adtypes = AbstractADType[AutoZygote(), AutoMooncake()]
    NNLIB_TEST_ENZYME && push!(adtypes, AutoEnzyme())
    rtol = T == Float32 ? 1e-3 : 1e-6

    @testset "$f with $(nameof(typeof(ad)))" for ad in adtypes,
            f in [σ, sigmoid_fast, tanh_fast, elu, selu, celu, gelu_tanh, gelu_sigmoid,
                  swish, softplus, logσ, logcosh, mish, lisht, tanhshrink]
        for x in T[-3, -0.5, 0.5, 3]
            @test withgradient(f, ad, x).grad[1] ≈ ForwardDiff.derivative(f, x) rtol=rtol
        end
        xs = T[-3, -0.5, 0.5, 3]
        @test withgradient(x -> sum(f.(x)), ad, xs).grad[1] ≈ ForwardDiff.derivative.(f, xs) rtol=rtol
    end

    # `softplus` and `mish` are assembled from two pieces that meet at zero
    @testset "at zero with $(nameof(typeof(ad)))" for ad in adtypes
        @test withgradient(softplus, ad, zero(T)).grad[1] ≈ 0.5
        @test withgradient(softplus, ad, -zero(T)).grad[1] ≈ 0.5
        @test withgradient(logσ, ad, zero(T)).grad[1] ≈ 0.5
        @test withgradient(mish, ad, zero(T)).grad[1] ≈ 0.6
        @test withgradient(σ, ad, -zero(T)).grad[1] ≈ 0.25
        @test withgradient(sigmoid_fast, ad, -zero(T)).grad[1] ≈ 0.25
    end
    @test ForwardDiff.derivative(σ, -zero(T)) ≈ 0.25
    @test ForwardDiff.derivative(sigmoid_fast, -zero(T)) ≈ 0.25
end

@testset "AD in half precision" begin
    # Enzyme and Mooncake differentiate the conversions around the Float32 code as well
    adtypes = AbstractADType[AutoZygote(), AutoMooncake()]
    NNLIB_TEST_ENZYME && push!(adtypes, AutoEnzyme())
    xs = Float16[-1.5, -0.5, 0.5, 1.5]

    @testset "$f with $(nameof(typeof(ad)))" for ad in adtypes, f in HALF_ACTIVATIONS
        ref = ForwardDiff.derivative.(f, Float32.(xs))
        @test [withgradient(f, ad, x).grad[1] for x in xs] ≈ ref rtol=2e-2
        @test withgradient(x -> sum(f.(x)), ad, xs).grad[1] ≈ ref rtol=2e-2
    end
end

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
        @test test_gradients(f, +2 + rand(rng))
        @test test_gradients(f, -2 - rand(rng))
        @test test_gradients(x -> f.(x), +2 .+ rand(rng, 2, 2))
        @test test_gradients(x -> f.(x), -2 .- rand(rng, 2, 2))

        if f in BINARY_ACTIVATIONS
            @test test_gradients(x -> f(x, 0.2), 1 + rand(rng))
            @test test_gradients(x -> f(x, 0.7), 1 + rand(rng))

            @test test_gradients(x -> f(x, 0.2), -2 + rand(rng))
            @test test_gradients(x -> f(x, 0.7), -2 + rand(rng))
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
        @test test_gradients((x, W, b) -> σ.(W*x .+ b), randn(rng, 5), randn(rng, 2, 5), randn(rng, 2))
        @test test_gradients((x, W, b) -> σ.(W*x .+ b), randn(rng, 5, 3), randn(rng, 2, 5), randn(rng, 2))
        @test test_gradients((x, W, b) -> relu.(W*x .+ b), randn(rng, 5), randn(rng, 2, 5), randn(rng, 2))
        @test test_gradients((x, W, b) -> relu.(W*x .+ b), randn(rng, 5, 3), randn(rng, 2, 5), randn(rng, 2))
        @test test_gradients((x, W, b) -> selu.(W*x .+ b), randn(rng, 5), randn(rng, 2, 5), randn(rng, 2))
        @test test_gradients((x, W, b) -> selu.(W*x .+ b), randn(rng, 5, 3), randn(rng, 2, 5), randn(rng, 2))
        @test test_gradients((x, W, b) -> elu.(W*x .+ b, 2), randn(rng, 5), randn(rng, 2, 5), randn(rng, 2))
        @test test_gradients((x, W, b) -> elu.(W*x .+ b, 2), randn(rng, 5, 3), randn(rng, 2, 5), randn(rng, 2))

        @test test_gradients((x, W, b) -> logσ.(W*x .+ b), randn(rng, 5), randn(rng, 2, 5), randn(rng, 2))
        @test test_gradients((x, W, b) -> logσ.(W*x .+ b), randn(rng, 5, 3), randn(rng, 2, 5), randn(rng, 2))

        ## Binary functions have their own broadcast rules:
        @test test_gradients((x, W, b) -> leakyrelu.(W*x .+ b, 0.2), randn(rng, 5), randn(rng, 2, 5), randn(rng, 2))
        @test test_gradients((x, W, b) -> leakyrelu.(W*x .+ b, 0.7), randn(rng, 5, 3), randn(rng, 2, 5), randn(rng, 2))
    end

    @testset "Zygote issue 758" begin
        ## Tests for https://github.com/FluxML/Zygote.jl/issues/758
        @test gradient(xs -> sum(selu.(xs)), [1_000, 10_000])[1] ≈ [1.0507009873554805, 1.0507009873554805] rtol=1e-8
        @test gradient(x -> selu(x), 1_000) == (1.0507009873554805,)
        @test gradient(xs -> sum(elu.(xs, 2)), [1_000, 10_000]) == ([1., 1.],)
        @test gradient(x -> elu(x, 2), 1_000) == (1.,)
        @test gradient(x -> elu(x, 2), -1) == (2*exp(-1),)
        # Enzyme differentiates through the raw `exp` and overflows to NaN on these
        # saturated inputs; the overflow-safe gradient is a Zygote/ChainRules property,
        # so compare against Zygote only here.
        @test test_gradients(x-> selu.(x),[100., 1_000.]; compare = AutoZygote())
        @test test_gradients(x -> elu.(x, 3.5),[100., 1_000.]; compare = AutoZygote())
        @test test_gradients(x -> elu.(x, 3.5),[1_000., 10_000.]; compare = AutoZygote())
        @test test_gradients(x -> selu.(x), [1_000., 10_000.]; compare = AutoZygote())
        @test test_gradients(x -> selu.(x), randn(rng, 10))
    end

    @testset "saturated gradients: $f" for f in (hardσ, hardtanh, relu6, hardswish)
        # The `gradtest`s above only probe near the origin (±2±rand), which sits inside
        # the linear region of the clamping activations and never exercises the saturated
        # branch of their derivative. Check that branch directly against ForwardDiff.
        for x in (-50.0, 50.0, -7.0, 7.0)
            ad = Zygote.gradient(z -> sum(f.(z)), [x])[1][1]
            fd = ForwardDiff.derivative(f, x)
            @test ad ≈ fd  atol=1e-6
        end
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

@testset "complex" begin
    ## https://github.com/FluxML/NNlib.jl/issues/132
    ## Holomorphic activations are well-defined on ℂ. Each is checked against a naive
    ## elementary formula for (a) the forward value and (b) the gradient. Gradients of
    ## both `f` and the naive version are taken with Zygote, so whatever complex-gradient
    ## convention Zygote uses cancels, and the test isolates the correctness of NNlib's
    ## hand-written rules (in particular the `conj` needed for the ChainRules convention).
    holo = [
        (σ,          z -> 1 / (1 + exp(-z))),
        (swish,      z -> z / (1 + exp(-z))),
        (softplus,   z -> log(1 + exp(z))),
        (logσ,       z -> -log(1 + exp(-z))),
        (mish,       z -> z * tanh(log(1 + exp(z)))),
        (logcosh,    z -> log(cosh(z))),
        (tanh_fast,  z -> tanh(z)),
        (lisht,      z -> z * tanh(z)),
        (tanhshrink, z -> z - tanh(z)),
    ]
    zs = ComplexF64[0.4 + 0.6im, -0.7 + 0.2im, 1.2 - 0.9im, -0.3 - 1.1im]

    @testset "$f" for (f, naive) in holo
        for z in zs
            @test f(z) ≈ naive(z)                                   # forward
            @test gradient(z -> abs2(f(z)), z)[1] ≈
                  gradient(z -> abs2(naive(z)), z)[1]               # scalar gradient
        end
        @test gradient(z -> sum(abs2, f.(z)), zs)[1] ≈
              gradient(z -> sum(abs2, naive.(z)), zs)[1]            # broadcasted gradient
    end

    @testset "numerical stability" begin
        ## must stay finite where the naive formula overflows / loses the phase
        for z in (600 + 2im, -600 + 2im, 80 - 50im, -80 + 50im)
            @test isfinite(σ(z))
            @test isfinite(softplus(z))
            @test isfinite(swish(z))
        end
        @test σ(-800 + 0im) ≈ 0 atol = 1e-12
        @test real(softplus(800 + 1im)) ≈ 800
    end

    @testset "piecewise activations reject complex" begin
        ## Order-defined activations have no canonical complex extension; they should
        ## throw rather than silently pick one (e.g. CReLU/modReLU/zReLU).
        for f in (relu, leakyrelu, relu6, elu, selu, celu,
                  hardσ, hardtanh, hardswish, trelu, softshrink)
            @test_throws MethodError f(1.0 + 1.0im)
        end
    end
end
