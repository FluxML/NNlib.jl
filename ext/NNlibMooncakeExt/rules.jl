# Ported from `ext/MooncakeNNlibExt.jl` of Mooncake.jl 0.5.63 (MIT, © Will Tebbutt and Hong Ge),
# keeping its formatting so that the two stay easy to compare while both exist.

using NNlib, Random, Mooncake
import NNlib.GPUArraysCore: AbstractGPUArray
using Base: IEEEFloat
using LinearAlgebra
using Statistics: mean
using NNlib: conv, depthwiseconv, logsoftmax, softmax, logsumexp, dropout
using Mooncake.Nfwd: NDual
import ChainRulesCore as CRC

import Mooncake:
    @from_rrule,
    DefaultCtx,
    MinimalCtx,
    @is_primitive,
    rrule!!,
    CoDual,
    NoRData,
    zero_fcodual,
    primal,
    tangent,
    arrayify,
    frule!!,
    Dual,
    NoPullback,
    ForwardMode,
    ReverseMode

# Half precision is accumulated in `S = Float32`, as NNlib's `logsumexp` does, and returned in it.
@inline function _nf_logsumexp_accum(
    grad::NTuple{N,S}, w::S, partials::NTuple{N,IEEEFloat}
) where {N,S}
    return ntuple(k -> grad[k] + w * partials[k], Val(N))
end

@inline function _nf_logsumexp_scale(grad::NTuple{N,T}, inv_sw::T) where {N,T}
    return ntuple(k -> grad[k] * inv_sw, Val(N))
end

@inline function _nf_logsumexp_inf(x::AbstractVector{<:NDual{<:Any,N}}, u::S) where {N,S}
    count_u = 0
    grad = ntuple(_ -> zero(S), Val(N))
    @inbounds for xi in x
        if xi.value == u
            count_u += 1
            grad = _nf_logsumexp_accum(grad, one(S), xi.partials)
        end
    end
    return NDual{S,N}(u, _nf_logsumexp_scale(grad, inv(S(count_u))))
end

function NNlib.logsumexp(x::AbstractVector{NDual{T,N}}) where {T<:IEEEFloat,N}
    S = typeof(NNlib._acc(zero(T)))
    isempty(x) && return NDual{S,N}(typemin(S))
    u = S(@inbounds x[begin].value)
    @inbounds for i in (firstindex(x) + 1):lastindex(x)
        v = S(x[i].value)
        v > u && (u = v)
    end
    isinf(u) && return _nf_logsumexp_inf(x, u)
    sum_w = zero(S)
    grad = ntuple(_ -> zero(S), Val(N))
    @inbounds for xi in x
        w = exp(S(xi.value) - u)
        sum_w += w
        grad = _nf_logsumexp_accum(grad, w, xi.partials)
    end
    y_val = u + log(sum_w)
    return NDual{S,N}(y_val, _nf_logsumexp_scale(grad, inv(sum_w)))
end

# Array types which we test rules against, so are confident work.
# Parametric on both element type P and dimensionality N.
const SupportedArray{P,N} = Union{
    Array{P,N},
    AbstractGPUArray{P,N},
    Adjoint{P,<:Union{Array{P,N},AbstractGPUArray{P,N}}},
    Transpose{P,<:Union{Array{P,N},AbstractGPUArray{P,N}}},
}
# The GPU-backed members of `SupportedArray`. `Adjoint`/`Transpose` of a GPU array are
# `AnyGPUArray`, which is what NNlib dispatches its kernels on, so a bound of bare
# `AbstractGPUArray` does not cover everything that reaches a kernel.
const GPUBackedArray{P,N} = Union{
    AbstractGPUArray{P,N},
    Adjoint{P,<:AbstractGPUArray{P,N}},
    Transpose{P,<:AbstractGPUArray{P,N}},
}
# `@from_rrule` does not honour the wrapper members: it adds ChainRules' cotangent into a view's
# fdata, which is the parent's, so those signatures throw — hence `dropout`'s hand-written rule.

# Floats the softmax family is differentiated for. NNlib computes half precision in Float32; BFloat16
# is Mooncake's primitive float only where `BFloat16s.BFloat16` is `Core.BFloat16` (Julia ≥ 1.11).
const SoftmaxFloat = if Mooncake.tangent_type(NNlib.BFloat16) === NNlib.BFloat16
    Union{IEEEFloat,NNlib.BFloat16}
else
    IEEEFloat
end

# On Julia ≤ 1.11, `maximum(x::Adjoint/Transpose; dims, init)` routes through
# `LinearAlgebra.mapreducedim! → switch_dim12 → PermutedDimsArray`, leaving
# type parameters unresolved and causing JET type-stability failures.
# Collecting CPU-backed wrappers to a plain Array avoids that path.
@static if VERSION < v"1.12"
    function _maximum(
        x::Tx, dims, init
    ) where {T<:SoftmaxFloat,A<:Array{T},Tx<:Union{Adjoint{T,A},Transpose{T,A}}}
        return maximum(collect(x); dims, init)
    end
end
_maximum(x, dims, init) = maximum(x; dims, init)

const BatchedMulArray{P} = Union{Array{P,3},AbstractGPUArray{P,3}}

@is_primitive MinimalCtx Tuple{
    typeof(batched_mul),BatchedMulArray{P},BatchedMulArray{P}
} where {P<:IEEEFloat}

function frule!!(
    ::Dual{typeof(batched_mul)},
    A::Dual{<:BatchedMulArray{P}},
    B::Dual{<:BatchedMulArray{P}},
) where {P<:IEEEFloat}
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    y = batched_mul(pA, pB)
    dy = batched_mul(dA, pB)
    NNlib.batched_mul!(dy, pA, dB, one(P), one(P))
    return Dual(y, dy)
end

function rrule!!(
    ::CoDual{typeof(batched_mul)},
    A::CoDual{<:BatchedMulArray{P}},
    B::CoDual{<:BatchedMulArray{P}},
) where {P<:IEEEFloat}
    pA, dA = arrayify(A)
    pB, dB = arrayify(B)
    y = batched_mul(pA, pB)
    dy = zero(y)
    function batched_mul_pullback!!(::NoRData)
        if size(pA, 3) == 1 && size(dy, 3) != 1
            dA .+= sum(batched_mul(dy, NNlib.batched_adjoint(pB)); dims=3)
        else
            NNlib.batched_mul!(dA, dy, NNlib.batched_adjoint(pB), one(P), one(P))
        end
        if size(pB, 3) == 1 && size(dy, 3) != 1
            dB .+= sum(batched_mul(NNlib.batched_adjoint(pA), dy); dims=3)
        else
            NNlib.batched_mul!(dB, NNlib.batched_adjoint(pA), dy, one(P), one(P))
        end
        return NoRData(), NoRData(), NoRData()
    end
    return CoDual(y, dy), batched_mul_pullback!!
end

@is_primitive MinimalCtx Tuple{
    typeof(NNlib._affine_normalize),
    DenseArray{P},
    DenseArray{P},
    DenseArray{P},
    DenseArray{P},
    DenseArray{P},
    P,
} where {P<:IEEEFloat}

function frule!!(
    ::Dual{typeof(NNlib._affine_normalize)},
    x::Dual{<:DenseArray{P}},
    mean::Dual{<:DenseArray{P}},
    variance::Dual{<:DenseArray{P}},
    scale::Dual{<:DenseArray{P}},
    bias::Dual{<:DenseArray{P}},
    eps::Dual{P},
) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    pmean, dmean = arrayify(mean)
    pvariance, dvariance = arrayify(variance)
    pscale, dscale = arrayify(scale)
    pbias, dbias = arrayify(bias)
    peps, deps = primal(eps), tangent(eps)
    centered = px .- pmean
    inv_std = inv.(sqrt.(pvariance .+ peps))
    y = pscale .* centered .* inv_std .+ pbias
    dy =
        dscale .* centered .* inv_std .+ pscale .* (dx .- dmean) .* inv_std .-
        P(0.5) .* pscale .* centered .* inv_std .^ 3 .* (dvariance .+ deps) .+ dbias
    return Dual(y, dy)
end

function rrule!!(
    ::CoDual{typeof(NNlib._affine_normalize)},
    x::CoDual{<:DenseArray{P}},
    mean::CoDual{<:DenseArray{P}},
    variance::CoDual{<:DenseArray{P}},
    scale::CoDual{<:DenseArray{P}},
    bias::CoDual{<:DenseArray{P}},
    eps::CoDual{P},
) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    pmean, dmean = arrayify(mean)
    pvariance, dvariance = arrayify(variance)
    pscale, dscale = arrayify(scale)
    pbias, dbias = arrayify(bias)
    peps = primal(eps)
    centered = px .- pmean
    inv_std = inv.(sqrt.(pvariance .+ peps))
    y = pscale .* centered .* inv_std .+ pbias
    dy = zero(y)
    function affine_normalize_pullback!!(::NoRData)
        common = dy .* pscale .* inv_std
        dx .+= common
        dmean .+= NNlib._unbroadcast(-common, dmean)
        variance_cotangent = -P(0.5) .* common .* centered .* inv_std .^ 2
        dvariance .+= NNlib._unbroadcast(variance_cotangent, dvariance)
        dscale .+= NNlib._unbroadcast(dy .* centered .* inv_std, dscale)
        dbias .+= NNlib._unbroadcast(dy, dbias)
        return (
            NoRData(),
            NoRData(),
            NoRData(),
            NoRData(),
            NoRData(),
            NoRData(),
            sum(variance_cotangent),
        )
    end
    return CoDual(y, dy), affine_normalize_pullback!!
end

# Forward mode. The operations with reverse rules above are traced by forward mode. On the CPU
# that works, except for the convolutions: the GEMM under them is a `ccall` without a rule. On
# the GPU a traced kernel launch fails, or kills the process. Most of these operations
# are linear or bilinear, and the JVP is the primal applied to the tangents.

function _bilinear_jvp(f, kw, a, b, args...)
    pa, da = arrayify(a)
    pb, db = arrayify(b)
    dy = f(da, pb, args...; kw...)
    dy .+= f(pa, db, args...; kw...)
    return Dual(f(pa, pb, args...; kw...), dy)
end

for f in (:conv, :∇conv_data, :∇conv_filter)
    @eval begin
        @is_primitive MinimalCtx ForwardMode Tuple{
            typeof($f),SupportedArray{P,N},SupportedArray{P,M},ConvDims
        } where {P<:IEEEFloat,N,M}
        @is_primitive MinimalCtx ForwardMode Tuple{
            typeof(Core.kwcall),NamedTuple,typeof($f),SupportedArray{P,N},SupportedArray{P,M},ConvDims
        } where {P<:IEEEFloat,N,M}

        function frule!!(
            ::Dual{typeof($f)},
            a::Dual{<:SupportedArray{P,N}},
            b::Dual{<:SupportedArray{P,M}},
            cdims::Dual{<:ConvDims},
        ) where {P<:IEEEFloat,N,M}
            return _bilinear_jvp($f, (;), a, b, primal(cdims))
        end
        function frule!!(
            ::Dual{typeof(Core.kwcall)},
            kw::Dual{<:NamedTuple},
            ::Dual{typeof($f)},
            a::Dual{<:SupportedArray{P,N}},
            b::Dual{<:SupportedArray{P,M}},
            cdims::Dual{<:ConvDims},
        ) where {P<:IEEEFloat,N,M}
            return _bilinear_jvp($f, primal(kw), a, b, primal(cdims))
        end
    end
end

# Linear in the array, with the other arguments constant.
function _linear_jvp(f, kw, x, args...)
    px, dx = arrayify(x)
    return Dual(f(px, args...; kw...), f(dx, args...; kw...))
end

for (f, extra) in (
    (:meanpool, (:PoolDims,)),
    (:(NNlib.unfold), (:DenseConvDims,)),
    (:(NNlib.fold), (:(NTuple{<:Any,Int}), :DenseConvDims)),
)
    @eval begin
        @is_primitive MinimalCtx ForwardMode Tuple{
            typeof($f),GPUBackedArray{<:IEEEFloat,N},$(extra...)
        } where {N}
        @is_primitive MinimalCtx ForwardMode Tuple{
            typeof(Core.kwcall),NamedTuple,typeof($f),GPUBackedArray{<:IEEEFloat,N},$(extra...)
        } where {N}

        function frule!!(
            ::Dual{typeof($f)}, x::Dual{<:GPUBackedArray{<:IEEEFloat,N}}, args::Vararg{Dual}
        ) where {N}
            return _linear_jvp($f, (;), x, map(primal, args)...)
        end
        function frule!!(
            ::Dual{typeof(Core.kwcall)},
            kw::Dual{<:NamedTuple},
            ::Dual{typeof($f)},
            x::Dual{<:GPUBackedArray{<:IEEEFloat,N}},
            args::Vararg{Dual},
        ) where {N}
            return _linear_jvp($f, primal(kw), x, map(primal, args)...)
        end
    end
end

# cuDNN does not say which element a window takes. Unfolded, the windows are columns: the element
# is the largest of a column, and the tangent is read at the same position. The windows are
# padded with `-Inf`, including the part past the input that `ceil_mode` adds.
@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(maxpool),GPUBackedArray{<:IEEEFloat,N},PoolDims
} where {N}
@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(maxpool),GPUBackedArray{<:IEEEFloat,N},PoolDims
} where {N}

function _maxpool_jvp(kw, x, pdims)
    px, dx = arrayify(x)
    k, s, d = NNlib.kernel_size(pdims), NNlib.stride(pdims), NNlib.dilation(pdims)
    lo = NNlib.padding(pdims)[1:2:end]
    out, insize = NNlib.output_size(pdims), NNlib.input_size(pdims)
    hi = @. max((out - 1) * s + d * (k - 1) + 1 - insize - lo, 0)
    pad = (tuple.(lo, hi)..., (0, 0), (0, 0))
    channels_in_batch(a) = reshape(a, size(a)[1:(end - 2)]..., 1, :)
    windows = pad_constant(channels_in_batch(px), pad, -eltype(px)(Inf))
    cdims = DenseConvDims(
        pdims; I=size(windows)[1:(end - 2)], C_in=1, C_out=1, P=ntuple(_ -> 0, 2 * length(out)), G=1
    )
    selected = argmax(NNlib.unfold(windows, cdims); dims=2)
    dy = NNlib.gather(NNlib.unfold(pad_constant(channels_in_batch(dx), pad, 0), cdims), selected)
    return Dual(maxpool(px, pdims; kw...), reshape(dy, out..., size(px)[(end - 1):end]...))
end

function frule!!(
    ::Dual{typeof(maxpool)}, x::Dual{<:GPUBackedArray{<:IEEEFloat,N}}, pdims::Dual{<:PoolDims}
) where {N}
    return _maxpool_jvp((;), x, primal(pdims))
end
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    kw::Dual{<:NamedTuple},
    ::Dual{typeof(maxpool)},
    x::Dual{<:GPUBackedArray{<:IEEEFloat,N}},
    pdims::Dual{<:PoolDims},
) where {N}
    return _maxpool_jvp(primal(kw), x, primal(pdims))
end

# The softmax family. The Jacobian of `softmax` is symmetric, so its JVP is `∇softmax`.
function _softmax_jvp(::typeof(softmax), x, dims)
    px, dx = arrayify(x)
    y = softmax(px; dims)
    return Dual(y, NNlib.∇softmax(dx, y; dims))
end
function _softmax_jvp(::typeof(logsoftmax), x, dims)
    px, dx = arrayify(x)
    y = logsoftmax(px; dims)
    return Dual(y, dx .- sum(exp.(y) .* dx; dims))
end
function _softmax_jvp(::typeof(logsumexp), x, dims)
    px, dx = arrayify(x)
    y = logsumexp(px; dims)
    return Dual(y, sum(exp.(px .- y) .* dx; dims))
end

for (f, dims) in ((:softmax, 1), (:logsoftmax, 1), (:logsumexp, :(:)))
    @eval begin
        @is_primitive MinimalCtx ForwardMode Tuple{
            typeof($f),GPUBackedArray{T,N}
        } where {T<:IEEEFloat,N}
        @is_primitive MinimalCtx ForwardMode Tuple{
            typeof(Core.kwcall),NamedTuple,typeof($f),GPUBackedArray{T,N}
        } where {T<:IEEEFloat,N}

        function frule!!(
            ::Dual{typeof($f)}, x::Dual{<:GPUBackedArray{T,N}}
        ) where {T<:IEEEFloat,N}
            return _softmax_jvp($f, x, $dims)
        end
        function frule!!(
            ::Dual{typeof(Core.kwcall)},
            kw::Dual{<:NamedTuple{(:dims,)}},
            ::Dual{typeof($f)},
            x::Dual{<:GPUBackedArray{T,N}},
        ) where {T<:IEEEFloat,N}
            return _softmax_jvp($f, x, primal(kw).dims)
        end
    end
end

# The tangent of `pad_constant` is padded with the tangent of the constant, which is zero unless
# it is a float.
@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(pad_constant),GPUBackedArray{<:IEEEFloat,N},Any,Any
} where {N}
@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(pad_constant),GPUBackedArray{<:IEEEFloat,N},Any,Any
} where {N}

function _pad_constant_jvp(kw, x, pad, val)
    px, dx = arrayify(x)
    dval = tangent(val) isa Number ? tangent(val) : 0
    return Dual(
        pad_constant(px, primal(pad), primal(val); kw...),
        pad_constant(dx, primal(pad), dval; kw...),
    )
end

function frule!!(
    ::Dual{typeof(pad_constant)}, x::Dual{<:GPUBackedArray{<:IEEEFloat,N}}, pad::Dual, val::Dual
) where {N}
    return _pad_constant_jvp((;), x, pad, val)
end
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    kw::Dual{<:NamedTuple},
    ::Dual{typeof(pad_constant)},
    x::Dual{<:GPUBackedArray{<:IEEEFloat,N}},
    pad::Dual,
    val::Dual,
) where {N}
    return _pad_constant_jvp(primal(kw), x, pad, val)
end

# The dropout mask multiplies the input, so the JVP is the pullback of ChainRules' rule.
@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(dropout),AbstractRNG,GPUBackedArray{P,N},Real
} where {P<:IEEEFloat,N}
@is_primitive MinimalCtx ForwardMode Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(dropout),AbstractRNG,GPUBackedArray{P,N},Real
} where {P<:IEEEFloat,N}

function _dropout_jvp(kw, rng, x::Dual{<:GPUBackedArray{P}}, p) where {P}
    0 <= primal(p) <= 1 || throw(ArgumentError("dropout expects a probability 0 <= p <= 1"))
    px, dx = arrayify(x)
    NNlib._rng_compat_array(primal(rng), px)
    # As for the reverse rule, the input itself is the output at `p ≤ 0`.
    primal(p) <= 0 && dropout(primal(rng), px, primal(p); kw...) === px && return x
    y, pullback = Core.kwcall(kw, CRC.rrule, dropout, primal(rng), px, convert(P, primal(p)))
    return Dual(y, pullback(dx)[3])
end

function frule!!(
    ::Dual{typeof(dropout)},
    rng::Dual{<:AbstractRNG},
    x::Dual{<:GPUBackedArray{P,N}},
    p::Dual{<:Real},
) where {P<:IEEEFloat,N}
    return _dropout_jvp((;), rng, x, p)
end
function frule!!(
    ::Dual{typeof(Core.kwcall)},
    kw::Dual{<:NamedTuple},
    ::Dual{typeof(dropout)},
    rng::Dual{<:AbstractRNG},
    x::Dual{<:GPUBackedArray{P,N}},
    p::Dual{<:Real},
) where {P<:IEEEFloat,N}
    return _dropout_jvp(primal(kw), rng, x, p)
end

# `scatter` of `+`, `-` and `mean` is linear in the source. For `max` and `min` a destination takes
# the tangent of the source it holds, or the mean over several tied ones, as the reverse rule gives
# each of them its share of the cotangent, and `init` competes for the extremum as a source. Traced
# on the CPU, they pick one of the tied sources, so they have a rule there as well.
const LinearScatterOp = Union{typeof(+),typeof(-),typeof(mean)}
const ExtremumScatterOp = Union{typeof(max),typeof(min)}

for (Op, Arr) in ((:LinearScatterOp, :GPUBackedArray), (:ExtremumScatterOp, :SupportedArray))
    @eval begin
        @is_primitive MinimalCtx ForwardMode Tuple{
            typeof(NNlib.scatter),$Op,$Arr{P,N},SupportedArray{<:Union{Integer,Tuple},M}
        } where {P<:IEEEFloat,N,M}
        @is_primitive MinimalCtx ForwardMode Tuple{
            typeof(Core.kwcall),
            NamedTuple,
            typeof(NNlib.scatter),
            $Op,
            $Arr{P,N},
            SupportedArray{<:Union{Integer,Tuple},M},
        } where {P<:IEEEFloat,N,M}

        function frule!!(
            ::Dual{typeof(NNlib.scatter)},
            op::Dual{<:$Op},
            src::Dual{<:$Arr{P,N}},
            idx::Dual{<:SupportedArray{<:Union{Integer,Tuple},M}},
        ) where {P<:IEEEFloat,N,M}
            return _scatter_jvp(primal(op), (;), nothing, src, primal(idx))
        end
        function frule!!(
            ::Dual{typeof(Core.kwcall)},
            kw::Dual{<:NamedTuple},
            ::Dual{typeof(NNlib.scatter)},
            op::Dual{<:$Op},
            src::Dual{<:$Arr{P,N}},
            idx::Dual{<:SupportedArray{<:Union{Integer,Tuple},M}},
        ) where {P<:IEEEFloat,N,M}
            return _scatter_jvp(primal(op), primal(kw), tangent(kw), src, primal(idx))
        end
    end
end

function _scatter_jvp(op, kw, dkw, src, idx)
    psrc, dsrc = arrayify(src)
    y = NNlib.scatter(op, psrc, idx; kw...)
    # The tangent of the keywords is `NoTangent` unless `init` is a float.
    dinit = dkw isa NamedTuple ? dkw.init : nothing
    return Dual(y, _scatter_tangent(op, psrc, dsrc, idx, y, get(kw, :init, nothing), dinit))
end

function _scatter_tangent(op::LinearScatterOp, psrc, dsrc, idx, y, init, dinit)
    dy = NNlib.scatter(op, dsrc, idx; dstsize=size(y))
    dinit isa Number && (dy .+= dinit)
    return dy
end

function _scatter_tangent(::ExtremumScatterOp, psrc::AbstractArray{P}, dsrc, idx, y, init, dinit) where {P}
    tied = psrc .== NNlib.gather(y, idx)
    init_tie = isnothing(init) ? false : y .== convert(P, init)
    counts = NNlib.scatter!(+, fill!(similar(y, Int), 0), tied, idx)
    counts .+= init_tie
    dy = NNlib.scatter(+, ifelse.(tied, dsrc, zero(P)), idx; dstsize=size(y))
    dinit isa Number && (dy .+= init_tie .* convert(P, dinit))
    # A destination no index reaches has no tied source.
    return dy ./ max.(counts, 1)
end

# At `p ≤ 0` `dropout` returns its input itself, where ChainRules' rrule allocates and draws
# from `rng` regardless. The `===` is checked, not assumed, because if that fast path ever
# allocates, returning the input would alias where the primal does not.
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(dropout),AbstractRNG,SupportedArray{P,N},Real
} where {P<:IEEEFloat,N}
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(dropout),AbstractRNG,SupportedArray{P,N},Real
} where {P<:IEEEFloat,N}

# ChainRules called here rather than through `Mooncake.rrule_wrapper`, which adds the cotangent
# straight into the fdata: for the `Adjoint` and `Transpose` that `SupportedArray` admits, that
# fdata is the parent's and has its shape. Adding through `arrayify`'s wrapper relabels instead.
function Mooncake.rrule!!(
    f::CoDual{typeof(dropout)},
    rng::CoDual{<:AbstractRNG},
    x::CoDual{<:SupportedArray{P,N}},
    p::CoDual{<:Real},
) where {P<:IEEEFloat,N}
    0 <= primal(p) <= 1 || throw(ArgumentError("dropout expects a probability 0 <= p <= 1"))
    if primal(p) <= 0 && dropout(primal(rng), primal(x), primal(p)) === primal(x)
        return x, NoPullback(f, rng, x, p)
    end
    px, dx = arrayify(x)
    y, cr_pb = CRC.rrule(dropout, primal(rng), px, convert(P, primal(p)))
    res = zero_fcodual(y)
    dp = Mooncake.zero_rdata(primal(p))
    function dropout_pb!!(::NoRData)
        dx .+= cr_pb(tangent(res))[3]
        return NoRData(), NoRData(), NoRData(), dp
    end
    return res, dropout_pb!!
end

function Mooncake.rrule!!(
    kwcall::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    f::CoDual{typeof(dropout)},
    rng::CoDual{<:AbstractRNG},
    x::CoDual{<:SupportedArray{P,N}},
    p::CoDual{<:Real},
) where {P<:IEEEFloat,N}
    0 <= primal(p) <= 1 || throw(ArgumentError("dropout expects a probability 0 <= p <= 1"))
    pkw = primal(kw)
    if primal(p) <= 0 && dropout(primal(rng), primal(x), primal(p); pkw...) === primal(x)
        return x, NoPullback(kwcall, kw, f, rng, x, p)
    end
    px, dx = arrayify(x)
    y, cr_pb = Core.kwcall(pkw, CRC.rrule, dropout, primal(rng), px, convert(P, primal(p)))
    res = zero_fcodual(y)
    kw_rdata = Mooncake.zero_rdata(pkw)
    dp = Mooncake.zero_rdata(primal(p))
    function dropout_kw_pb!!(::NoRData)
        dx .+= cr_pb(tangent(res))[3]
        return NoRData(), kw_rdata, NoRData(), NoRData(), NoRData(), dp
    end
    return res, dropout_kw_pb!!
end

# logsoftmax rrules
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(logsoftmax),SupportedArray{T,N}
} where {T<:SoftmaxFloat,N}
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsoftmax),SupportedArray{T,N}
} where {T<:SoftmaxFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsoftmax)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:SoftmaxFloat,N}
    xp = primal(x)
    y = logsoftmax(xp)
    res = zero_fcodual(y)
    function logsoftmax_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        dx .+= NNlib.∇logsoftmax(dy, y; dims=1)
        return NoRData(), NoRData()
    end
    return res, logsoftmax_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(logsoftmax)},
    x::CoDual{<:SupportedArray{T,N}},
) where {T<:SoftmaxFloat,N}
    dims = primal(kw).dims
    xp = primal(x)
    y = logsoftmax(xp; dims)
    res = zero_fcodual(y)
    function logsoftmax_kw_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        dx .+= NNlib.∇logsoftmax(dy, y; dims)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, logsoftmax_kw_pb!!
end

# softmax rrules
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(softmax),SupportedArray{T,N}
} where {T<:SoftmaxFloat,N}
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(softmax),SupportedArray{T,N}
} where {T<:SoftmaxFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(softmax)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:SoftmaxFloat,N}
    xp = primal(x)
    y = softmax(xp)
    res = zero_fcodual(y)
    function softmax_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        dx .+= NNlib.∇softmax(dy, y; dims=1)
        return NoRData(), NoRData()
    end
    return res, softmax_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(softmax)},
    x::CoDual{<:SupportedArray{T,N}},
) where {T<:SoftmaxFloat,N}
    dims = primal(kw).dims
    xp = primal(x)
    y = softmax(xp; dims)
    res = zero_fcodual(y)
    function softmax_kw_pb!!(::NoRData)
        _, dx = arrayify(x)
        dy = tangent(res)
        dx .+= NNlib.∇softmax(dy, y; dims)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, softmax_kw_pb!!
end

# logsumexp rrules
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(logsumexp),SupportedArray{T,N}
} where {T<:SoftmaxFloat,N}
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsumexp),SupportedArray{T,N}
} where {T<:SoftmaxFloat,N}

# In the type NNlib's `logsumexp` computes and returns in (Float32 for half precision). The entries
# equal to the maximum count as one, which avoids `Inf - Inf` when it is infinite.
function _logsumexp_terms(x, dims)
    max_ = NNlib._acc.(_maximum(x, dims, typemin(eltype(x))))
    tmp = ifelse.(x .== max_, one(eltype(max_)), NNlib._exp.(x .- max_))
    s = sum(tmp; dims)
    return max_ .+ log.(s), tmp, s
end

function Mooncake.rrule!!(
    ::CoDual{typeof(logsumexp)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:SoftmaxFloat,N}
    y, tmp, s = _logsumexp_terms(primal(x), :)
    res = zero_fcodual(y)
    function logsumexp_pb!!(dy)
        _, dx = arrayify(x)
        dx .+= dy .* tmp ./ s
        return NoRData(), NoRData()
    end
    return res, logsumexp_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple{(:dims,)}},
    ::CoDual{typeof(logsumexp)},
    x::CoDual{<:SupportedArray{T,N}},
) where {T<:SoftmaxFloat,N}
    y, tmp, s = _logsumexp_terms(primal(x), primal(kw).dims)
    res = zero_fcodual(y)
    function logsumexp_kw_pb!!(::NoRData)
        _, dx = arrayify(x)
        dx .+= tangent(res) .* tmp ./ s
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, logsumexp_kw_pb!!
end

@from_rrule(
    MinimalCtx,
    Tuple{typeof(upsample_nearest),SupportedArray{<:IEEEFloat,N},NTuple{M,Int}} where {N,M},
)
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(NNlib.fold),SupportedArray{<:IEEEFloat,N},NTuple{M,Int},DenseConvDims
    } where {N,M},
)
@from_rrule(
    MinimalCtx,
    Tuple{typeof(NNlib.unfold),SupportedArray{<:IEEEFloat,N},DenseConvDims} where {N},
)
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(NNlib.scatter),
        Any,
        SupportedArray{P,N},
        SupportedArray{<:Union{Integer,Tuple},M},
    } where {P,N,M},
    true,
)

# ChainRules' `∇scatter_src` for `max`/`min` gives the full cotangent to every tied source,
# so the gradient sums to the tie multiplicity rather than to 1, not a subdifferential
# member. Dividing by the count picks the symmetric member; `init` competes for the same
# extremum, so a source tied with it gets `1/(m+1)`. Returns `init`'s share, or `nothing`
# when it has no rdata slot to receive one.
@inline function _scatter_extremum_grads!(
    dsrc, psrc::AbstractArray{P}, pidx, y, dy, init
) where {P}
    # `convert`, because NNlib rounds `init` into the destination before comparing.
    tied = psrc .== NNlib.gather(y, pidx)
    init_tie = isnothing(init) ? false : y .== convert(P, init)
    # `Int`, because a float count saturates once its spacing passes 1 — `Float16` at 2048,
    # `Float32` at 2^24 — inflating every share in a larger tie group.
    counts = NNlib.scatter!(+, fill!(similar(y, Int), 0), tied, pidx)
    isnothing(init) || (counts .+= init_tie)
    # `A` widens `Float16` for the division, whose operands are all O(1). No zero guard: `gather`
    # reads only destinations some index reaches, and each holds a tied source or `init` itself.
    A = P === Float16 ? Float32 : P
    total = A.(counts)
    dsrc .+= P.(tied .* A.(NNlib.gather(dy, pidx)) ./ NNlib.gather(total, pidx))
    # `oftype`, because `init` need not share `src`'s precision.
    (isnothing(init) || Mooncake.zero_rdata(init) isa Mooncake.NoRData) && return nothing
    return oftype(init, sum(A.(dy) .* init_tie ./ total))
end

function Mooncake.rrule!!(
    ::CoDual{typeof(NNlib.scatter)},
    op::CoDual{<:Union{typeof(max),typeof(min)}},
    src::CoDual{<:SupportedArray{P,N}},
    idx::CoDual{<:SupportedArray{<:Union{Integer,Tuple},M}},
) where {P<:IEEEFloat,N,M}
    psrc, dsrc = arrayify(src)
    pidx = primal(idx)
    res = zero_fcodual(NNlib.scatter(primal(op), psrc, pidx))
    function scatter_extremum_pb!!(::NoRData)
        _scatter_extremum_grads!(dsrc, psrc, pidx, primal(res), tangent(res), nothing)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, scatter_extremum_pb!!
end

function Mooncake.rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(NNlib.scatter)},
    op::CoDual{<:Union{typeof(max),typeof(min)}},
    src::CoDual{<:SupportedArray{P,N}},
    idx::CoDual{<:SupportedArray{<:Union{Integer,Tuple},M}},
) where {P<:IEEEFloat,N,M}
    psrc, dsrc = arrayify(src)
    pidx = primal(idx)
    pkw = primal(kw)
    res = zero_fcodual(NNlib.scatter(primal(op), psrc, pidx; pkw...))
    function scatter_extremum_kw_pb!!(::NoRData)
        # `haskey` on a `NamedTuple`'s type is settled at compile time, so each keyword set
        # gets its own specialisation and `kw_rdata` has one concrete type per call site.
        init = haskey(pkw, :init) ? pkw.init : nothing
        dinit = _scatter_extremum_grads!(dsrc, psrc, pidx, primal(res), tangent(res), init)
        # With `init` supplied the keyword `NamedTuple`'s rdata is not `NoRData`; without, it is.
        kw_rdata = if dinit === nothing
            Mooncake.zero_rdata(pkw)
        else
            merge(Mooncake.zero_rdata(pkw), (; init=dinit))
        end
        return NoRData(), kw_rdata, NoRData(), NoRData(), NoRData(), NoRData()
    end
    return res, scatter_extremum_kw_pb!!
end

# ChainRules defines an `rrule` only for the in-place `gather!`, not `gather`, so without
# this AD would trace `gather`'s scalar per-index loop (see
# https://github.com/chalk-lab/Mooncake.jl/issues/1234). The pullback is `∇gather_src`
# accumulating straight into `src`'s fdata, avoiding its intermediate allocation.
@is_primitive(
    MinimalCtx,
    ReverseMode,
    Tuple{
        typeof(NNlib.gather),SupportedArray{P,N},SupportedArray{<:Union{Integer,Tuple},M}
    } where {P<:IEEEFloat,N,M},
)
# Forward mode traces `gather` on the CPU. On the GPU a traced kernel launch does not survive
# the forward transform: the process dies with signal 4, no Julia exception to catch.
@is_primitive(
    MinimalCtx,
    ForwardMode,
    Tuple{
        typeof(NNlib.gather),GPUBackedArray{P,N},SupportedArray{<:Union{Integer,Tuple},M}
    } where {P<:IEEEFloat,N,M},
)
function Mooncake.frule!!(
    ::Dual{typeof(NNlib.gather)},
    src::Dual{<:GPUBackedArray{P,N}},
    idx::Dual{<:SupportedArray{<:Union{Integer,Tuple},M}},
) where {P<:IEEEFloat,N,M}
    psrc, dsrc = arrayify(src)
    pidx = primal(idx)
    return Dual(NNlib.gather(psrc, pidx), NNlib.gather(dsrc, pidx))
end

function Mooncake.rrule!!(
    ::CoDual{typeof(NNlib.gather)},
    src::CoDual{<:SupportedArray{P,N}},
    idx::CoDual{<:SupportedArray{<:Union{Integer,Tuple},M}},
) where {P<:IEEEFloat,N,M}
    pidx = primal(idx)
    res = zero_fcodual(NNlib.gather(primal(src), pidx))
    function gather_pb!!(::NoRData)
        _, dsrc = arrayify(src)
        if dsrc isa Union{Array,AbstractGPUArray}
            NNlib.scatter!(+, dsrc, tangent(res), pidx)
        else
            # `arrayify` keeps the wrapper and `scatter!` takes only a dense destination: a
            # wrapped one compiles to invalid IR on the GPU and finds no method on the CPU.
            # `similar` on an `Adjoint` re-wraps, so the buffer comes from the parent.
            buf = fill!(similar(parent(dsrc), size(dsrc)), 0)
            dsrc .+= NNlib.scatter!(+, buf, tangent(res), pidx)
        end
        return NoRData(), NoRData(), NoRData()
    end
    return res, gather_pb!!
end
# Views are taken by the rules in `NNlibMooncakeExt.jl`: `@from_rrule` cannot.
for conv in [:conv, :depthwiseconv]
    local ∇conv_data, ∇conv_filter = Symbol.(:∇, conv, [:_data, :_filter])

    @eval @from_rrule(
        MinimalCtx,
        Tuple{
            typeof($conv),SupportedArray{P,N},SupportedArray{P,M},ConvDims
        } where {P<:IEEEFloat,N,M},
        true,
    )
    @eval @from_rrule(
        MinimalCtx,
        Tuple{
            typeof($∇conv_data),SupportedArray{P,N},SupportedArray{P,M},ConvDims
        } where {P<:IEEEFloat,N,M},
        true,
    )
end
@from_rrule(
    MinimalCtx,
    Tuple{
        typeof(∇conv_filter),SupportedArray{P,N},SupportedArray{P,M},ConvDims
    } where {P<:IEEEFloat,N,M},
    true,
)
for pool in [:maxpool, :meanpool]
    @eval @from_rrule(
        MinimalCtx,
        Tuple{typeof($pool),SupportedArray{<:IEEEFloat,N},PoolDims} where {N},
        true,
    )
end
@from_rrule(
    MinimalCtx, Tuple{typeof(pad_constant),SupportedArray{P,N},Any,Any} where {P,N}, true,
)

# Direct rules for bias_act!(identity, x, b) on CPU and GPU arrays.
# bias_act! modifies x in-place (x .+= b), so we save x's primal before mutation,
# compute in-place, return x as output, and restore x's primal in the pullback.
@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(bias_act!),
        typeof(identity),
        SupportedArray{<:IEEEFloat,N} where {N},
        SupportedArray{<:IEEEFloat,M} where {M},
    },
)
function frule!!(
    ::Dual{typeof(bias_act!)},
    ::Dual{typeof(identity)},
    x::Dual{<:SupportedArray{<:IEEEFloat,N}},
    b::Dual{<:SupportedArray{<:IEEEFloat,M}},
) where {N,M}
    primal(x) .+= primal(b)
    tangent(x) .+= tangent(b)
    return x
end
function rrule!!(
    ::CoDual{typeof(bias_act!)},
    ::CoDual{typeof(identity)},
    x::CoDual{<:SupportedArray{P}},
    b::CoDual{<:SupportedArray{<:IEEEFloat}},
) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    pb, db = arrayify(b)
    px_copy = copy(px)
    px .+= pb
    # Dims over which b is broadcast (size 1 in b but potentially larger in x).
    broadcast_dims = Tuple(filter(d -> size(pb, d) == 1, 1:ndims(px)))
    function bias_act_id_pb!!(::NoRData)
        if isempty(broadcast_dims)
            db .+= dx
        else
            db .+= reshape(sum(dx; dims=broadcast_dims), size(pb))
        end
        copyto!(px, px_copy)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return x, bias_act_id_pb!!
end

# Handle Flux's common non-identity activations directly to avoid the generic NDual broadcast
# machinery. Like the identity rule above, these rules restore `x` during the pullback.
const GPUFastActivation = Union{typeof(tanh),typeof(tanh_fast)}
@is_primitive MinimalCtx Tuple{
    typeof(bias_act!),GPUFastActivation,AbstractGPUArray{P},AbstractGPUArray{P}
} where {P<:IEEEFloat}

@inline function _tanh_fast_derivative(x)
    u = exp(-2 * abs(x))
    return 4u / (one(x) + u)^2
end

@inline _bias_act_derivative(::GPUFastActivation, x) = _tanh_fast_derivative(x)

function frule!!(
    ::Dual{typeof(bias_act!)},
    σ::Dual{<:GPUFastActivation},
    x::Dual{<:AbstractGPUArray{P}},
    b::Dual{<:AbstractGPUArray{P}},
) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    pb, db = arrayify(b)
    pσ = primal(σ)
    dx .= _bias_act_derivative.(pσ, px .+ pb) .* (dx .+ db)
    bias_act!(pσ, px, pb)
    return x
end

function rrule!!(
    ::CoDual{typeof(bias_act!)},
    σ::CoDual{<:GPUFastActivation},
    x::CoDual{<:AbstractGPUArray{P}},
    b::CoDual{<:AbstractGPUArray{P}},
) where {P<:IEEEFloat}
    px, dx = arrayify(x)
    pb, db = arrayify(b)
    pσ = primal(σ)
    old_px = copy(px)
    bias_act!(pσ, px, pb)
    broadcast_dims = Tuple(filter(d -> size(pb, d) == 1, 1:ndims(px)))
    function gpu_bias_act_pb!!(::NoRData)
        dx .*= _bias_act_derivative.(pσ, old_px .+ pb)
        if isempty(broadcast_dims)
            db .+= dx
        else
            db .+= reshape(sum(dx; dims=broadcast_dims), size(pb))
        end
        copyto!(px, old_px)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return x, gpu_bias_act_pb!!
end

# σ is smooth, but tracing the primal is wrong at both ends. At zero it routes through
# `abs(x)`, so AD picks up `sign(0) == 0` and reports `0` for `1/4`. When saturated the
# textbook `σ(x) * (1 - σ(x))` collapses: floats near `1.0` are spaced `eps` apart, so once
# `σ(x)` rounds to `1.0` the factor `1 - σ(x)` is exactly `0` (Float64 `x ≳ 37`, Float32
# `≳ 17`, Float16 `≳ 8`) while the
# true value is still normal. `t / (1 + t)^2` with `t = exp(-abs(x))` holds it in an
# exponent instead, and `t ≤ 1` bounds the quotient. `exp` therefore runs twice, here and in
# the primal, rather than restating the primal's branch and `sigmoid_fast`'s clamps where
# they could drift from it — 2 ns of 5.5. Both rules differentiate the unclamped σ: past
# `x ≈ 36.8` the analytic value is all there is, and below `-80` `sigmoid_fast`'s clamp is
# documented as an accuracy compromise, not a different function.
for f in (:σ, :sigmoid_fast)
    @eval @is_primitive MinimalCtx Tuple{typeof(NNlib.$f),P} where {P<:IEEEFloat}
    @eval function Mooncake.frule!!(
        ::Dual{typeof(NNlib.$f)}, x::Dual{P}
    ) where {P<:IEEEFloat}
        t = exp(-abs(primal(x)))
        d = t / (one(P) + t)^2
        return Dual(NNlib.$f(primal(x)), tangent(x) * d)
    end
    @eval function Mooncake.rrule!!(
        ::CoDual{typeof(NNlib.$f)}, x::CoDual{P}
    ) where {P<:IEEEFloat}
        t = exp(-abs(primal(x)))
        d = t / (one(P) + t)^2
        sigmoid_pb!!(dΩ::P) = NoRData(), dΩ * d
        return zero_fcodual(NNlib.$f(primal(x))), sigmoid_pb!!
    end
    # GPU elementwise and reduction rules evaluate the whole fused broadcast on `NDual`s
    # inside one kernel, so they never reach the rules above and need the derivative too.
    @eval @inline function NNlib.$f(x::NDual{P,N}) where {P<:IEEEFloat,N}
        t = exp(-abs(x.value))
        d = t / (one(P) + t)^2
        return NDual{P,N}(NNlib.$f(x.value), ntuple(i -> x.partials[i] * d, Val(N)))
    end
end

# `tanh_fast` selects between branches with `ifelse`, which is a call, so the discarded
# branch is computed: past `|x| ≈ 355` the Float64 body's `(exp(2x) - 1) / (exp(2x) + 1)` is
# `Inf/Inf`, and a zero cotangent into that quotient's pullback forms `0 * Inf` and puts
# `NaN` on the argument. The Float32 body overflows its Remez rational the same way past
# `|x| ≈ 618587`, or `≈ 258.8` through `gelu_tanh`, well inside a Float32 pre-activation.
# `gelu`/`gelu_tanh` inherit the `NaN` from `|x| ≈ 21`. No `NDual` method — it is not an
# `IEEEFloat`, so the in-kernel path reaches `Base.tanh`.
#
# `4u / (1 + u)^2` with `u = exp(-2|x|)` for σ's reason: `1 - tanh(x)^2` collapses to `0`
# once `tanh(x)` rounds to `1.0` (Float64 `|x| ≳ 19.5`, Float32 `≳ 9`).
@is_primitive MinimalCtx Tuple{typeof(tanh_fast),P} where {P<:IEEEFloat}
function Mooncake.frule!!(::Dual{typeof(tanh_fast)}, x::Dual{P}) where {P<:IEEEFloat}
    d = _tanh_fast_derivative(primal(x))
    return Dual(tanh_fast(primal(x)), tangent(x) * d)
end
function Mooncake.rrule!!(::CoDual{typeof(tanh_fast)}, x::CoDual{P}) where {P<:IEEEFloat}
    d = _tanh_fast_derivative(primal(x))
    tanh_fast_pb!!(dΩ::P) = NoRData(), dΩ * d
    return zero_fcodual(tanh_fast(primal(x))), tanh_fast_pb!!
end
