module NNlibMooncakeExt

using Mooncake: Mooncake, MinimalCtx, DefaultCtx, CoDual, Dual, NoRData, primal, tangent,
    zero_fcodual, @is_primitive, @zero_adjoint, @zero_derivative
using NNlib: NNlib
using Base.Broadcast: Broadcasted, DefaultArrayStyle, broadcasted, materialize

# Mooncake's reverse pass restores every value that was overwritten on the forward pass.
# Traced like ordinary code, the in-place update of the running statistics is therefore
# undone, and a gradient call hands back the statistics it started with. As a primitive
# the update is an opaque call instead, so its writes are kept, as under Zygote and Enzyme.
#
# The statistics are updated once per evaluation of the function. `prepare_gradient_cache`
# and `prepare_pullback_cache` evaluate it once themselves, so they update them as well.
@zero_adjoint MinimalCtx Tuple{typeof(NNlib._update_running_stats!),Vararg}

# `CUDA.default_rng()` reads the task-local storage, which Mooncake cannot differentiate
# (chalk-lab/Mooncake.jl#1368). Same as the `@non_differentiable` for ChainRules.
@zero_derivative MinimalCtx Tuple{typeof(NNlib._rng_from_array),AbstractArray}

# Mooncake has no predicate to query, so `within_gradient` is replaced in the code it differentiates.
Mooncake.@mooncake_overlay NNlib.within_gradient(x) = true

# `_exp` and `_expm1` build a power of two from its bits, which Mooncake refuses to
# differentiate (a `bitcast` from an integer to a float). As primitives they are called
# as they are, and their derivative comes from the value they return. `_log1p_unit` it
# can differentiate, but one operation at a time: its rule is there for speed only.
#
# The rules are inlined into the code Mooncake derives, like the kernels into NNlib's:
# left as a call, each costs some 2 ns more per element.
for (f, ctx, dydx) in ((:_exp, :MinimalCtx, :y), (:_expm1, :MinimalCtx, :(y + 1)),
                       (:_log1p_unit, :DefaultCtx, :(inv(1 + x))))
    @eval begin
        @is_primitive $ctx Tuple{typeof(NNlib.$f),P} where {P<:Union{Float32,Float64}}
        @inline function Mooncake.frule!!(::Dual{typeof(NNlib.$f)}, x_ẋ::Dual{P}) where {P<:Union{Float32,Float64}}
            x = primal(x_ẋ)
            y = NNlib.$f(x)
            return Dual(y, $dydx * tangent(x_ẋ))
        end
        @inline function Mooncake.rrule!!(::CoDual{typeof(NNlib.$f)}, x_x̄::CoDual{P}) where {P<:Union{Float32,Float64}}
            x = primal(x_x̄)
            y = NNlib.$f(x)
            dydx = $dydx
            pullback(ȳ::P) = NoRData(), dydx * ȳ
            return zero_fcodual(y), pullback
        end
    end
end

# A broadcast of an activation over an array is one primitive, as it is one `rrule` for
# ChainRules. Taken element by element, Mooncake has to record each of them for its reverse
# pass, and neither of the two loops is then compiled to SIMD code. Here both are, with the
# derivatives of the `@scalar_rule`s.
#
# As that `rrule` does, `broadcasted` computes the activation at once, in the code Mooncake
# differentiates: an array comes out of the fused broadcast `g.(f.(h.(x)))` after `f`, and
# another one goes into it, where `h.(x)` is computed.
#
# `CuArray`s are left to Mooncake, which differentiates a fused broadcast in one kernel:
# with the activation computed apart, `f.(x .+ b)` is 1.3 times slower there.
const Activation = Union{(typeof(getfield(NNlib, f)) for (f, _) in NNlib.UNARY_ACTS)...}
const FloatArray = Array{<:Union{Float32,Float64}}

_broadcast(f::Activation, x::FloatArray) = f.(x)

# Varargs, as Base's `broadcasted(f, arg1, args...)`: for a call dispatched at runtime,
# Mooncake takes the varargs layout from Base's method and runs the overlay's code (#799).
Mooncake.@mooncake_overlay Base.Broadcast.broadcasted(f::Activation, x::Vararg{FloatArray,1}) =
    _broadcast(f, only(x))
Mooncake.@mooncake_overlay Base.Broadcast.broadcasted(f::Activation, x::Vararg{Broadcasted{<:DefaultArrayStyle},1}) =
    broadcasted(f, materialize(only(x)))

@is_primitive DefaultCtx Tuple{typeof(_broadcast),Activation,FloatArray}

function Mooncake.frule!!(::Dual{typeof(_broadcast)}, f_ḟ::Dual{<:Activation}, x_ẋ::Dual{<:FloatArray})
    f, x, ẋ = primal(f_ḟ), primal(x_ẋ), tangent(x_ẋ)
    Ω = f.(x)
    return Dual(Ω, _add_derivative!(zero(Ω), ẋ, Ω, f, x))
end

function Mooncake.rrule!!(::CoDual{typeof(_broadcast)}, f_f̄::CoDual{<:Activation}, x_x̄::CoDual{<:FloatArray})
    f, x, x̄ = primal(f_f̄), primal(x_x̄), tangent(x_x̄)
    Ω = f.(x)
    Ω̄ = zero(Ω)
    function pullback(::NoRData)
        _add_derivative!(x̄, Ω̄, Ω, f, x)
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(Ω, Ω̄), pullback
end

# `acc .+= t .* f'.(x)`, with a tangent `t` in forward mode and a cotangent in reverse. A
# loop, as that broadcast is not compiled to SIMD code for a matrix when the derivative is
# as large as `sigmoid_fast(x)`.
function _add_derivative!(acc, t, Ω, f::F, x) where {F}
    @inbounds @simd for i in eachindex(acc, t, Ω, x)
        acc[i] += t[i] * NNlib.only_derivative(Ω[i], f, x[i])
    end
    return acc
end

end
