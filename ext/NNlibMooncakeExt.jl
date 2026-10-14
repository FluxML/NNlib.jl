module NNlibMooncakeExt

using Mooncake: Mooncake, MinimalCtx, DefaultCtx, CoDual, Dual, NoRData, primal, tangent,
    zero_fcodual, @is_primitive, @zero_adjoint, @zero_derivative
using NNlib: NNlib

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

end
