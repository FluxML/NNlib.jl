
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
    if eltype(B) !== Bool
        b_dims = ntuple(d -> size(b,d)==1 ? d : N+1, N)
        size_b = size(b)
    end

    # Fast path: it is now safe to overwrite x, since this is not needed for gradient of σ
    if isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, NotaNumber}))
        Ω = bias_act!(σ, x, b)  # now x === Ω, when x isa StridedArray{<:AbstractFloat}
        function bias_act!_fastback(Δ)
            # Tempting to overwrite x again, but only safe if you call pullback at most once,
            # TODO with e.g. https://github.com/FluxML/Zygote.jl/pull/1340
            # https://github.com/JuliaDiff/ChainRulesCore.jl/pull/592
            dx = only_derivative.(Ω, σ, NotaNumber()) .* unthunk(Δ)
            db = eltype(B) === Bool ? NoTangent() : reshape(sum(dx; dims = b_dims), size_b)
            return (NoTangent(), NoTangent(), dx, db)
        end
        return Ω, bias_act!_fastback

    # # Slower path: can't overwrite x, but can use derivatives_given_output
    # # This case is WRONG and tests fail, but not sure why
    # elseif isconcretetype(Core.Compiler._return_type(only_derivative, Tuple{T, F, T}))
    #     Ω2 = fast_act(σ, x).(x) .+ b
    #     @show σ b
    #     function bias_act!_back2(Δ)
    #         dx = only_derivative.(Ω2, σ, x .+ b) .* unthunk(Δ)
    #         db = eltype(B) === Bool ? NoTangent() : reshape(sum(dx; dims = b_dims), size_b)
    #         return (NoTangent(), NoTangent(), dx, db)
    #     end
    #     return Ω2, bias_act!_back2

    # Fallback path: let AD handle the broadcast
    else
        Ω3, back = rrule_via_ad(cfg, broadcast, fast_act(σ, x), bias_act!(identity, x, b))
        @inline function bias_act!_slowback(Δ)
            _, _, dx = back(Δ)
            db = eltype(B) === Bool ? NoTangent() : reshape(sum(dx; dims = b_dims), size_b)
            return (NoTangent(), NoTangent(), dx, db)
        end
        return Ω3, bias_act!_slowback
    end
end

# Two easy cases
function rrule(cfg::RCR, ::typeof(bias_act!), ::typeof(identity), x::AbstractArray{T,N}, b::B) where {T,N,B}
    b_dims = ntuple(d -> size(b,d)==1 ? d : N+1, N)
    size_b = size(b)
    function bias_act!_idback(Δ)
        dx = unthunk(Δ)
        db = reshape(sum(dx; dims = b_dims), size_b)
        return (NoTangent(), NoTangent(), dx, db)
    end
    return bias_act!(identity, x, b), bias_act!_idback
end

function rrule(cfg::RCR, ::typeof(bias_act!), ::typeof(identity), x::AbstractArray{T,N}, b::Bool) where {T,N}
    bias_act!_trivial(Δ) = (NoTangent(), NoTangent(), Δ, NoTangent())
    return x, bias_act!_trivial
end


