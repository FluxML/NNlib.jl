module NNlibMooncakeExt

using Mooncake: Mooncake, MinimalCtx, DefaultCtx, ReverseMode, CoDual, Dual, NoFData, NoRData,
    primal, tangent, zero_fcodual, zero_rdata, @is_primitive, @zero_adjoint
using NNlib: NNlib, batchnorm, instancenorm, groupnorm, layernorm, rmsnorm
using NNlib.GPUArraysCore: AbstractGPUArray
using ChainRulesCore: ChainRulesCore, unthunk
using Base: IEEEFloat
using Base.Broadcast: Broadcasted, DefaultArrayStyle, broadcasted, materialize

# BFloat16 is a primitive float to Mooncake only where `BFloat16s.BFloat16` is `Core.BFloat16`
# (Julia ≥ 1.11, as in Mooncake's own extension for it; `tangent_type` throws for it otherwise), and
# its arrays have an array tangent only on the CPU: a `CuArray{BFloat16}` has none, so no rule can
# return its gradient. The rules are for `RuleFloat` on the CPU and `IEEEFloat` on a device.
const MaybeBFloat16 = isdefined(Core, :BFloat16) && NNlib.BFloat16 === Core.BFloat16 ? NNlib.BFloat16 : Union{}
const RuleFloat = Union{IEEEFloat,MaybeBFloat16}

# Mooncake's reverse pass restores every value that was overwritten on the forward pass.
# Traced like ordinary code, the in-place update of the running statistics is therefore
# undone, and a gradient call hands back the statistics it started with. As a primitive
# the update is an opaque call instead, so its writes are kept, as under Zygote and Enzyme.
#
# The statistics are updated once per evaluation of the function. `prepare_gradient_cache`
# and `prepare_pullback_cache` evaluate it once themselves, so they update them as well.
@zero_adjoint MinimalCtx Tuple{typeof(NNlib._update_running_stats!),Vararg}

# Mooncake has no predicate to query, so `within_gradient` is replaced in the code it differentiates.
Mooncake.@mooncake_overlay NNlib.within_gradient(x) = true

# The code Mooncake differentiates runs serially: it can differentiate neither the
# `ScopedValue` read of `should_use_spawn` nor the tasks that a `true` would start. The rules
# call NNlib outside it, where the kernels keep their threads.
Mooncake.@mooncake_overlay NNlib.should_use_spawn() = false

# cuDNN's `batchnorm` for 3-5-D arrays cannot be traced, and an overlay method takes precedence
# over the more specific methods of the cuDNN extension: where Mooncake differentiates the call,
# the generic code runs, as `within_gradient` selects it for the other normalizations.
Mooncake.@mooncake_overlay function NNlib.batchnorm(
    g, b, x::AbstractGPUArray{T,N}, running_mean, running_var, momentum;
    eps=1f-5, training::Bool=true, track_stats::Bool=true, kws...,
) where {T,N}
    return NNlib._norm_layer(g, b, x, running_mean, running_var, (ntuple(identity, N - 2)..., N);
                             training, momentum, eps, track_stats)
end

# The backward pass of the softmax family calls cuDNN, and the reverse rules run it. A
# Hessian-vector product differentiates that pass, which the overlay keeps to the generic code.
for f in (:∇softmax!, :∇logsoftmax!)
    _f = Symbol(:_, f)
    @eval Mooncake.@mooncake_overlay NNlib.$f(
        dx::AbstractGPUArray, dy::AbstractGPUArray, y::AbstractGPUArray; dims=1
    ) = NNlib.$_f(dx, dy, y; dims)
end

# A normalization is one primitive, whose two passes are those of its ChainRules rule. That
# runs outside the code Mooncake differentiates, where `within_gradient` is false: the fused
# kernels compute the output and keep the statistics for the reverse pass, the running
# statistics are updated once, and cuDNN is called rather than traced (its `try`/`catch`
# cannot be). The gradients are then those of Zygote.
#
# Other arrays, and forward mode, differentiate the generic code.
const NormFunction = Union{typeof(batchnorm),typeof(instancenorm),typeof(groupnorm),
                           typeof(layernorm),typeof(rmsnorm)}
const NormArray = Union{Array{<:RuleFloat},AbstractGPUArray{<:IEEEFloat}}
const NormParam = Union{Nothing,NormArray}

for (f, args) in ((:batchnorm, :(NormParam, NormParam, NormArray, Vararg)),
                  (:instancenorm, :(NormParam, NormParam, NormArray, Vararg)),
                  (:groupnorm, :(NormParam, NormParam, NormArray, Integer)),
                  (:layernorm, :(NormParam, NormParam, NormArray)),
                  (:rmsnorm, :(NormParam, NormArray)))
    @eval begin
        @is_primitive MinimalCtx ReverseMode Tuple{typeof($f),$(args.args...)}
        @is_primitive MinimalCtx ReverseMode Tuple{typeof(Core.kwcall),NamedTuple,typeof($f),$(args.args...)}
    end
end

function Mooncake.rrule!!(f::CoDual{<:NormFunction}, args::Vararg{CoDual,N}) where {N}
    y, pullback = _chainrules_rrule(primal(f), (;), args)
    return y, ȳ -> (NoRData(), pullback(ȳ)...)
end

function Mooncake.rrule!!(::CoDual{typeof(Core.kwcall)}, kw::CoDual{<:NamedTuple},
                          f::CoDual{<:NormFunction}, args::Vararg{CoDual,N}) where {N}
    y, pullback = _chainrules_rrule(primal(f), primal(kw), args)
    kw_rdata = zero_rdata(primal(kw))
    return y, ȳ -> (NoRData(), kw_rdata, NoRData(), pullback(ȳ)...)
end

function _chainrules_rrule(f, kw, args::NTuple{N,CoDual}) where {N}
    y, cr_pullback = ChainRulesCore.rrule(f, map(primal, args)...; kw...)
    res = zero_fcodual(y)
    function pullback(::NoRData)
        Δ = cr_pullback(tangent(res))
        foreach(_accumulate!, map(_cotangent, args), ntuple(i -> unthunk(Δ[i + 1]), Val(N)))
        return map(zero_rdata ∘ primal, args)
    end
    return res, pullback
end

# The fdata of a view holds its parent, whole: the cotangent is added to the slice `arrayify` gives.
_cotangent(x::CoDual{<:SubArray}) = last(Mooncake.arrayify(x))
_cotangent(x::CoDual) = tangent(x)

# The running statistics have no cotangent, and a scale or bias that is `nothing` has
# nowhere to put the one cuDNN computes.
_accumulate!(x̄::AbstractArray, Δ::AbstractArray) = (x̄ .+= Δ; nothing)
_accumulate!(x̄::AbstractArray, ::ChainRulesCore.AbstractZero) = nothing
_accumulate!(::NoFData, Δ) = nothing

# Convolutions and pooling over views. The rules for arrays (`@from_rrule`) add ChainRules'
# cotangent to the fdata of an argument, which for a view holds its whole parent: they take no
# views, and Mooncake would trace one down to the BLAS calls. Here the cotangent goes into the
# slice of it that `arrayify` returns. A view is in at least one position of each signature,
# which keeps them apart from those of the rules for arrays. `fold` and `unfold` of BFloat16
# arrays take the same rules, as `@from_rrule` cannot convert BFloat16, and tracing them fails.
const ViewArray{P,N} = SubArray{P,N,<:Union{Array{P},AbstractGPUArray{P}}}
const PlainArray{P,N} = Union{Array{P,N},AbstractGPUArray{P,N}}
const ViewFunction = Union{typeof(NNlib.conv),typeof(NNlib.depthwiseconv),
                           typeof(NNlib.∇conv_data),typeof(NNlib.∇depthwiseconv_data),
                           typeof(NNlib.∇conv_filter),typeof(NNlib.maxpool),
                           typeof(NNlib.meanpool),typeof(NNlib.fold),typeof(NNlib.unfold)}

for f in (:conv, :depthwiseconv, :∇conv_data, :∇depthwiseconv_data, :∇conv_filter),
    (X, W) in ((:ViewArray, :PlainArray), (:PlainArray, :ViewArray), (:ViewArray, :ViewArray))
    @eval begin
        @is_primitive MinimalCtx ReverseMode Tuple{typeof(NNlib.$f),$X{P,N},$W{P,M},NNlib.ConvDims} where {P<:IEEEFloat,N,M}
        @is_primitive MinimalCtx ReverseMode Tuple{typeof(Core.kwcall),NamedTuple,typeof(NNlib.$f),$X{P,N},$W{P,M},NNlib.ConvDims} where {P<:IEEEFloat,N,M}
    end
end
for f in (:maxpool, :meanpool)
    @eval begin
        @is_primitive MinimalCtx ReverseMode Tuple{typeof(NNlib.$f),ViewArray{<:IEEEFloat,N},NNlib.PoolDims} where {N}
        @is_primitive MinimalCtx ReverseMode Tuple{typeof(Core.kwcall),NamedTuple,typeof(NNlib.$f),ViewArray{<:IEEEFloat,N},NNlib.PoolDims} where {N}
    end
end
@is_primitive MinimalCtx ReverseMode Tuple{typeof(NNlib.unfold),Array{<:MaybeBFloat16,N},NNlib.DenseConvDims} where {N}
@is_primitive MinimalCtx ReverseMode Tuple{typeof(NNlib.fold),Array{<:MaybeBFloat16,N},NTuple{M,Int},NNlib.DenseConvDims} where {N,M}

function Mooncake.rrule!!(f::CoDual{<:ViewFunction}, args::Vararg{CoDual,N}) where {N}
    y, pullback = _chainrules_rrule(primal(f), (;), args)
    return y, ȳ -> (NoRData(), pullback(ȳ)...)
end

function Mooncake.rrule!!(::CoDual{typeof(Core.kwcall)}, kw::CoDual{<:NamedTuple},
                          f::CoDual{<:ViewFunction}, args::Vararg{CoDual,N}) where {N}
    y, pullback = _chainrules_rrule(primal(f), primal(kw), args)
    kw_rdata = zero_rdata(primal(kw))
    return y, ȳ -> (NoRData(), kw_rdata, NoRData(), pullback(ȳ)...)
end

# `_exp` and `_expm1` build a power of two from its bits, which Mooncake refuses to
# differentiate (a `bitcast` from an integer to a float). As primitives they are called
# as they are, and their derivative comes from the value they return. `_log1p_unit` it
# can differentiate, but one operation at a time: its rule is there for speed only.
#
# The rules are inlined into the code Mooncake derives, like the kernels into NNlib's:
# left as a call, each costs some 2 ns more per element.
#
# A zero cotangent stays zero: `elu` and its relatives call `_expm1` on any input, large
# positive ones included, where it and its derivative overflow to `Inf`. The `ifelse` that
# discards it gives it a zero cotangent, and `0 * Inf` is `NaN`.
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
            pullback(ȳ::P) = NoRData(), ifelse(iszero(ȳ), ȳ, dydx * ȳ)
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
# Mooncake takes the varargs layout from Base's method and runs the overlay's code
# (chalk-lab/Mooncake.jl#1382).
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
# The rules in `rules.jl` come from Mooncake's own `MooncakeNNlibExt`, still shipped as of
# Mooncake 0.5.63. Its methods have the same signatures, so defining both would overwrite
# one with the other: ours are loaded only by a later Mooncake that no longer declares that
# extension.
const MOONCAKE_SHIPS_RULES = pkgversion(Mooncake) <= v"0.5.63" ||
    occursin(r"^MooncakeNNlibExt\b"m, read(joinpath(pkgdir(Mooncake), "Project.toml"), String))

MOONCAKE_SHIPS_RULES || include("rules.jl")

end
