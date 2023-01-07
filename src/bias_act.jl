
using NNlib: fast_act, tanh_fast
using ChainRulesCore

const RCR = RuleConfig{>:HasReverseMode}

# This just saves typing `only.(only.(` many times:
@inline only_derivative(y,f::F,x) where F = only(only(ChainRulesCore.derivatives_given_output(y, f, x)))

# This has no methods, used for testing whether `derivatives_given_output(Ω, f, x)`
# is independent of `x`, as `_return_type` says `Union{}` when calling is an error.
struct NotaNumber <: Real end

"""
    bias_act!(σ, x, b)

This is equivalent to `x .= σ.(x .+ b)`, also replacing `sigmoid` & `tanh`
with `sigmoid_fast` & `tanh_fast`.
It will only overwrite `x` when `x isa StridedArray{<:AbstractFloat}`.

When used within a gradient, it will overwrite only when `σ` has
a method of `derivatives_given_output` which does not need the input at all.
Such methods are defined by e.g. `@scalar_rule relu(x) Ω > 0` where the derivative
contains only `Ω` (the output) not `x`.

!!! warning
    This is not safe to use if `x` is still needed for the gradient
    of some other function. Incorrect use will give silently wrong answers.
    It is intended mainly for Flux layers, in which the previous operation is
    known to be safe, e.g. `bias_act!(σ, weight * input, bias)` for a `Dense` layer.
"""
bias_act!(σ::Function, x::AbstractArray, b) = fast_act(σ, x).(x .+ b)  # fallback

bias_act!(σ::Function, x::StridedArray{<:AbstractFloat}, b::AbstractArray{<:Union{Bool, AbstractFloat}}) =
    _fast_broadcast!(fast_act(σ, x)∘(+), x, b)  # works around a SIMD bug

bias_act!(::typeof(identity), x::StridedArray{<:AbstractFloat}, b::Bool) =
    (@assert !b "bias=true is not accepted; layer constructors shoud guarantee this";  x)


function ChainRulesCore.rrule(cfg::RCR, ::typeof(bias_act!), σ::F, x::AbstractArray{T,N}, b::B) where {F,T,N,B}
    biasgrad = if eltype(B) !== Bool
        # Summing over ndims(x)+1 is a trick to make b_dims type-stable
        dims = ntuple(d -> size(b,d)==1 ? d : N+1, N)
        _biasgrad(dx) = reshape(sum(dx; dims), size(b))
    else
        Returns(NoTangent())
    end

    # Fast path: it is now safe to overwrite x, since this is not needed for gradient of σ
    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        Ω = bias_act!(σ, x, b)  # now x === Ω, when x isa StridedArray{<:AbstractFloat}
        function bias_act!_fastback(Δ)
            # Tempting to overwrite x again, but only safe if you call pullback at most once,
            # TODO with e.g. https://github.com/FluxML/Zygote.jl/pull/1340
            # https://github.com/JuliaDiff/ChainRulesCore.jl/pull/592
            dx = only_derivative.(Ω, σ, NotaNumber()) .* unthunk(Δ)
            return (NoTangent(), NoTangent(), dx, biasgrad(dx))
        end
        return Ω, bias_act!_fastback

    # Slower path: can't overwrite x, but can use derivatives_given_output
    # This case is WRONG and tests fail, but not sure why
    elseif isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
        Ω2 = fast_act(σ, x).(x) .+ b
        @show σ b
        function bias_act!_back2(Δ)
            dx = only_derivative.(Ω2, σ, x .+ b) .* unthunk(Δ)
            return (NoTangent(), NoTangent(), dx, biasgrad(dx))
        end
        return Ω2, bias_act!_back2

    # Fallback path: let AD handle the broadcast
    else
        Ω3, back = rrule_via_ad(cfg, broadcast, fast_act(σ, x), bias_act!(identity, x, b))
        @inline function bias_act!_slowback(Δ)
            _, _, dx = back(Δ)
            return (NoTangent(), NoTangent(), dx, biasgrad(dx))
        end
        return Ω3, bias_act!_slowback
    end
end

# Two easy cases with identity
function rrule(cfg::RCR, ::typeof(bias_act!), ::typeof(identity), x::AbstractArray{T,N}, b::B) where {T,N,B}
    dims = ntuple(d -> size(b,d)==1 ? d : N+1, N)
    biasgrad(dx) = reshape(sum(dx; dims), size(b))
    function bias_act!_idback(Δ)
        dx = unthunk(Δ)
        return (NoTangent(), NoTangent(), dx,  biasgrad(dx))
    end
    return bias_act!(identity, x, b), bias_act!_idback
end
function rrule(cfg::RCR, ::typeof(bias_act!), ::typeof(identity), x::AbstractArray{T,N}, b::Bool) where {T,N}
    bias_act!_trivial(Δ) = (NoTangent(), NoTangent(), Δ, NoTangent())
    return x, bias_act!_trivial
end



# """
#     add_act(σ, x, y...)
#     add_act!(σ, x, y, z...)

# Equivalent to `σ.(x .+ y .+ z)`. The mutating method `add_act!`
# """
# add_act(σ::Function, x::AbstractArray, yz::AbstractArray...) = σ.(.+(x, yz...))  # fused


# function ChainRulesCore.rrule(::typeof(add_act), σ::F, x::AbstractArray, yz::AbstractArray...) where {F,T,N}
#     if isconcretetype(Core.Compiler._return_type(
#             derivatives_given_output, Tuple{T, F, NotaNumber}))

# end


# bias_act!(σ::Function, x::StridedArray{<:AbstractFloat}, b::Bool) =
#     # b ? (x .= fast_act(σ, x).(x .+ b)) : (x .= fast_act(σ, x).(x))
#     (@assert !b "bias=true is not accepted";  (x .= fast_act(σ, x).(x)))


# using NNlib, BenchmarkTools

#=

## M1 mac, 1.10

julia> w, b = rand(Float32, 100, 10000), rand(Float32, 100);

julia> @btime bias_act!(relu, $w, $b);
  min 19.500 μs, mean 21.375 μs (0 allocations)

julia> @btime relu.($w .+ $b);
  min 17.208 μs, mean 62.826 μs (2 allocations, 390.67 KiB)

julia> @btime bias_act!(tanh, $w, $b);
  min 63.792 μs, mean 65.052 μs (0 allocations)

julia> @btime tanh_fast.($w .+ $b);
  min 63.583 μs, mean 102.004 μs (2 allocations, 390.67 KiB)

julia> using Zygote

julia> @btime gradient((w,b) -> sum(bias_act!(relu, w, b)), $w, $b);
  min 145.166 μs, mean 150.785 μs (51 allocations, 2.18 KiB)

julia> @btime gradient((w,b) -> sum(relu.(w .+ b)), $w, $b);
  min 165.583 μs, mean 314.267 μs (32 allocations, 1.15 MiB)

julia> @btime gradient((w,b) -> sum(bias_act!(tanh, w, b)), $w, $b);
  min 191.917 μs, mean 195.956 μs (51 allocations, 2.18 KiB)

julia> @btime gradient((w,b) -> sum(tanh_fast.(w .+ b)), $w, $b);
  min 209.458 μs, mean 338.652 μs (32 allocations, 1.15 MiB)



## Cyclops

julia> using CUDA  # 10x bigger

julia> cw, cb = CUDA.rand(Float32, 100, 100_00), CUDA.rand(Float32, 100);

julia> @btime CUDA.@sync bias_act!(relu, $cw, $cb);
  22.546 μs (27 allocations: 1.45 KiB)

julia> @btime CUDA.@sync relu.($cw .+ $cb);  # faster, that's odd?
  31.282 μs (38 allocations: 1.81 KiB)

julia> @btime CUDA.@sync bias_act!(tanh, $cw, $cb);
  27.030 μs (27 allocations: 1.45 KiB)

julia> @btime CUDA.@sync tanh_fast.($cw .+ $cb);
  36.421 μs (38 allocations: 1.81 KiB)

julia> using Zygote

julia> @btime CUDA.@sync gradient((w,b) -> sum(bias_act!(relu, w, b)), $cw, $cb);
  204.507 μs (382 allocations: 18.15 KiB)

julia> @btime CUDA.@sync gradient((w,b) -> sum(relu.(w .+ b)), $cw, $cb);
  204.458 μs (409 allocations: 19.19 KiB)

julia> @btime CUDA.@sync gradient((w,b) -> sum(bias_act!(tanh, w, b)), $cw, $cb);
  224.545 μs (382 allocations: 18.15 KiB)

julia> @btime CUDA.@sync gradient((w,b) -> sum(tanh_fast.(w .+ b)), $cw, $cb);
  204.793 μs (411 allocations: 19.30 KiB)


=#

#=

(jl_fuwIi8) pkg> add https://github.com/mcabbott/NNlib.jl/tree/bias_act_23

julia> using NNlib, Zygote, BenchmarkTools

julia> w, b, x = rand(Float32, 50, 50), rand(Float32, 50), randn(Float32, 50, 100);

julia> @btime bias_act!(relu, $w * $x, $b);
  min 5.243 μs, mean 8.600 μs (2 allocations, 19.61 KiB)

julia> @btime relu.($w * $x .+ $b);
  min 5.160 μs, mean 10.863 μs (4 allocations, 39.22 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, bias_act!(relu, w*x, b)), $w, $x, $b);
  min 21.042 μs, mean 40.476 μs (43 allocations, 89.83 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, relu.(w*x .+ b)), $w, $x, $b);
  min 21.542 μs, mean 43.947 μs (41 allocations, 128.91 KiB)

julia> @btime gradient((w,x) -> sum(abs2, w*x), $w, $x);
  min 14.708 μs, mean 26.450 μs (28 allocations, 69.41 KiB)

julia> @btime gradient(x -> sum(abs2, x), $x);
  min 1.938 μs, mean 4.160 μs (2 allocations, 19.61 KiB)


# Cyclops

julia> @btime bias_act!(relu, $w * $x, $b);
  24.786 μs (2 allocations: 19.61 KiB)

julia> @btime relu.($w * $x .+ $b);
  25.501 μs (4 allocations: 39.22 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, bias_act!(relu, w*x, b)), $w, $x, $b);
  91.847 μs (43 allocations: 89.83 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, relu.(w*x .+ b)), $w, $x, $b);
  98.054 μs (41 allocations: 128.91 KiB)

julia> @btime gradient((w,x) -> sum(abs2, w*x), $w, $x);
  80.464 μs (28 allocations: 69.41 KiB)

julia> @btime gradient(x -> sum(abs2, x), $x);
  4.604 μs (2 allocations: 19.61 KiB)

julia> @time using CUDA; @time cu(ones(3)) .+ 1;

julia> w, b, x = CUDA.rand(Float32, 1000, 1000), CUDA.rand(Float32, 1000), CUDA.rand(Float32, 1000, 1000);



=#



