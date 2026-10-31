# Ported from `ext/MooncakeNNlibExt.jl` of Mooncake.jl 0.5.63 (MIT, © Will Tebbutt and Hong Ge),
# keeping its formatting so that the two stay easy to compare while both exist.

using NNlib, Random, Mooncake
import NNlib.GPUArraysCore: AbstractGPUArray
using Base: IEEEFloat
using LinearAlgebra
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
    NoFData,
    NoRData,
    NoTangent,
    zero_fcodual,
    primal,
    tangent,
    arrayify,
    frule!!,
    Dual,
    NoPullback,
    ForwardMode,
    ReverseMode

@inline function _nf_logsumexp_accum(
    grad::NTuple{N,T}, w::T, partials::NTuple{N,T}
) where {N,T}
    return ntuple(k -> grad[k] + w * partials[k], Val(N))
end

@inline function _nf_logsumexp_scale(grad::NTuple{N,T}, inv_sw::T) where {N,T}
    return ntuple(k -> grad[k] * inv_sw, Val(N))
end

@inline function _nf_logsumexp_inf(x::AbstractVector{NDual{T,N}}, u::T) where {T,N}
    count_u = 0
    grad = ntuple(_ -> zero(T), Val(N))
    @inbounds for xi in x
        if xi.value == u
            count_u += 1
            grad = _nf_logsumexp_accum(grad, one(T), xi.partials)
        end
    end
    return NDual{T,N}(u, _nf_logsumexp_scale(grad, inv(T(count_u))))
end

function NNlib.logsumexp(x::AbstractVector{NDual{T,N}}) where {T<:IEEEFloat,N}
    isempty(x) && return NDual{T,N}(typemin(T))
    u = @inbounds x[begin].value
    @inbounds for i in (firstindex(x) + 1):lastindex(x)
        v = x[i].value
        v > u && (u = v)
    end
    isinf(u) && return _nf_logsumexp_inf(x, u)
    sum_w = zero(T)
    grad = ntuple(_ -> zero(T), Val(N))
    @inbounds for xi in x
        w = exp(xi.value - u)
        sum_w += w
        grad = _nf_logsumexp_accum(grad, w, xi.partials)
    end
    y_val = u + log(sum_w)
    return NDual{T,N}(y_val, _nf_logsumexp_scale(grad, inv(sum_w)))
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

# On Julia ≤ 1.11, `maximum(x::Adjoint/Transpose; dims, init)` routes through
# `LinearAlgebra.mapreducedim! → switch_dim12 → PermutedDimsArray`, leaving
# type parameters unresolved and causing JET type-stability failures.
# Collecting CPU-backed wrappers to a plain Array avoids that path.
@static if VERSION < v"1.12"
    function _maximum(
        x::Tx, dims, init
    ) where {T<:IEEEFloat,A<:Array{T},Tx<:Union{Adjoint{T,A},Transpose{T,A}}}
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
} where {T<:IEEEFloat,N}
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsoftmax),SupportedArray{T,N}
} where {T<:IEEEFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsoftmax)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:IEEEFloat,N}
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
) where {T<:IEEEFloat,N}
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
} where {T<:IEEEFloat,N}
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(softmax),SupportedArray{T,N}
} where {T<:IEEEFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(softmax)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:IEEEFloat,N}
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
) where {T<:IEEEFloat,N}
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
} where {T<:IEEEFloat,N}
@is_primitive MinimalCtx ReverseMode Tuple{
    typeof(Core.kwcall),NamedTuple,typeof(logsumexp),SupportedArray{T,N}
} where {T<:IEEEFloat,N}

function Mooncake.rrule!!(
    ::CoDual{typeof(logsumexp)}, x::CoDual{<:SupportedArray{T,N}}
) where {T<:IEEEFloat,N}
    xp = primal(x)
    max_ = maximum(xp; init=typemin(T))
    @fastmath tmp = exp.(xp .- max_)
    s = sum(tmp)
    @fastmath y = max_ + log(s)
    res = zero_fcodual(y)
    function logsumexp_pb!!(dy::T)
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
) where {T<:IEEEFloat,N}
    dims = primal(kw).dims
    xp = primal(x)
    max_ = _maximum(xp, dims, typemin(T))
    # avoids Inf instability when xp[i]==max_==Inf
    @fastmath tmp = ifelse.(xp .== max_, one(T), exp.(xp .- max_))
    s = sum(tmp; dims)
    @fastmath y = max_ .+ log.(s)
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
# A GPU kernel launch does not survive the forward transform: the process dies with signal 4, no
# Julia exception to catch. A forward primitive that raises keeps it an ordinary error.
@is_primitive(
    MinimalCtx,
    ForwardMode,
    Tuple{
        typeof(NNlib.gather),GPUBackedArray{P,N},SupportedArray{<:Union{Integer,Tuple},M}
    } where {P<:IEEEFloat,N,M},
)
function Mooncake.frule!!(
    ::Dual{typeof(NNlib.gather)},
    ::Dual{<:GPUBackedArray{P,N}},
    ::Dual{<:SupportedArray{<:Union{Integer,Tuple},M}},
) where {P<:IEEEFloat,N,M}
    throw(
        ArgumentError(
            "forward mode over `NNlib.gather` is not supported for GPU arrays: " *
            "differentiating the traced kernel launch crashes the process. Reverse mode " *
            "has a rule for this signature and does work; on the CPU, so does forward " *
            "mode.",
        ),
    )
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

# `bias_act!` overwrites the dense array `x` and returns it. The rules save the old `x` and have
# the pullback restore it, as Mooncake's reverse pass has to. The element types are bounded in
# `BiasActFloat` alone. The bias is `false` for a layer without one.
const BiasActFloat = IEEEFloat
const BiasActArray{P} = Union{Array{P},AbstractGPUArray{P}}
const BiasActBias{P} = Union{BiasActArray{P},Bool}

# `fast_act` turns `tanh` and `sigmoid` (`σ`) into the fast variants, which have rules of their own.
const BiasActivation = Union{typeof(identity),typeof(tanh),Activation}

@is_primitive MinimalCtx Tuple{
    typeof(bias_act!),BiasActivation,BiasActArray{P},BiasActBias{P}
} where {P<:BiasActFloat}

# The derivative at the pre-activation `z`, `Ω` being the output. These are NNlib's, except for
# `tanh` and `σ` where `Ω` collapses to `±1` and `Ω * (1 - Ω)` to `0` long before the derivative
# does (see the rules for `σ` and `tanh_fast` below).
@inline _bias_act_derivative(Ω, σ, z) = NNlib.only_derivative(Ω, σ, z)
@inline _bias_act_derivative(_, ::typeof(identity), z) = one(z)
@inline _bias_act_derivative(_, ::Union{typeof(tanh),typeof(tanh_fast)}, z) =
    _tanh_fast_derivative(z)
@inline _bias_act_derivative(_, ::Union{typeof(NNlib.σ),typeof(sigmoid_fast)}, z) =
    _sigmoid_derivative(z)

# Whether the derivative is a function of the output alone, as that of `relu` is.
_derivative_of_output(σ::F, ::Type{P}) where {F,P} =
    isconcretetype(Core.Compiler.return_type(_bias_act_derivative, Tuple{P,F,NNlib.NotaNumber}))

# `dx .*= σ'.(x .+ b)`, with the output `Ω` in `px` and the input in `x`. A derivative that
# needs `x` is formed first, apart from `dx`: with `dx` read inside the broadcast, that of
# `swish` and the like is not compiled to SIMD code, and neither is it in `_fast_broadcast!`
# with more than one array besides `dx`.
_scale_by_derivative!(dx, ::typeof(identity), Ω, x, b) = dx
function _scale_by_derivative!(dx, σ, Ω, x, b)
    if _derivative_of_output(σ, eltype(dx))
        return NNlib._fast_broadcast!(
            (d, Ω) -> d * _bias_act_derivative(Ω, σ, NNlib.NotaNumber()), dx, Ω
        )
    end
    d = similar(dx)  # a `BitArray` otherwise, for a `Bool` derivative
    d .= _bias_act_derivative.(Ω, σ, x .+ b)
    return NNlib._fast_broadcast!(*, dx, d)
end

# `db .+= sum(dx)` over the dimensions along which `b` is broadcast.
_add_bias_gradient!(::NoFData, dx) = nothing
function _add_bias_gradient!(db, dx)
    if size(db) == size(dx)
        db .+= dx
    else
        N = ndims(dx)
        # Summing over `N + 1` is NNlib's trick to keep the dimensions type-stable.
        dims = ntuple(d -> size(db, d) == 1 ? d : N + 1, N)
        db .+= reshape(sum(dx; dims), size(db))
    end
    return nothing
end

_add_tangent(dx, db) = dx .+ db
_add_tangent(dx, ::NoTangent) = dx

function frule!!(
    ::Dual{typeof(bias_act!)},
    σ::Dual{<:BiasActivation},
    x::Dual{<:BiasActArray{P}},
    b::Dual{<:BiasActBias{P}},
) where {P<:BiasActFloat}
    px, dx = primal(x), tangent(x)
    pb, db = primal(b), tangent(b)
    pσ = primal(σ)
    z = px .+ pb
    bias_act!(pσ, px, pb)
    dx .= _add_tangent(dx, db) .* _bias_act_derivative.(px, pσ, z)
    return x
end

function rrule!!(
    ::CoDual{typeof(bias_act!)},
    σ::CoDual{<:BiasActivation},
    x::CoDual{<:BiasActArray{P}},
    b::CoDual{<:BiasActBias{P}},
) where {P<:BiasActFloat}
    px, dx = primal(x), tangent(x)
    pb, db = primal(b), tangent(b)
    pσ = primal(σ)
    old_px = copy(px)
    bias_act!(pσ, px, pb)
    function bias_act_pb!!(::NoRData)
        _scale_by_derivative!(dx, pσ, px, old_px, pb)
        _add_bias_gradient!(db, dx)
        copyto!(px, old_px)
        return NoRData(), NoRData(), NoRData(), NoRData()
    end
    return x, bias_act_pb!!
end

# σ is smooth, but tracing the primal is wrong at both ends. At zero it routes through
# `abs(x)`, so AD picks up `sign(0) == 0` and reports `0` for `1/4`. When saturated the
# textbook `σ(x) * (1 - σ(x))` collapses: floats near `1.0` are spaced `eps` apart, so once
# `σ(x)` rounds to `1.0` the factor `1 - σ(x)` is exactly `0` (Float64 `x ≳ 37`, Float32
# `≳ 17`, Float16 `≳ 8`) while the
# true value is still normal. `t / (1 + t)^2` with `t = exp(-abs(x))` holds it in an
# exponent instead, and `t ≤ 1` bounds the quotient. `_exp` therefore runs twice, here and in
# the primal, rather than restating the primal's branch and `sigmoid_fast`'s clamps where
# they could drift from it — 2 ns of 5.5. Both rules differentiate the unclamped σ: past
# `x ≈ 36.8` the analytic value is all there is, and below `-80` `sigmoid_fast`'s clamp is
# documented as an accuracy compromise, not a different function.
@inline function _sigmoid_derivative(x)
    t = NNlib._exp(-abs(x))
    return t / (one(t) + t)^2
end

for f in (:σ, :sigmoid_fast)
    @eval @is_primitive MinimalCtx Tuple{typeof(NNlib.$f),P} where {P<:IEEEFloat}
    @eval function Mooncake.frule!!(
        ::Dual{typeof(NNlib.$f)}, x::Dual{P}
    ) where {P<:IEEEFloat}
        return Dual(NNlib.$f(primal(x)), tangent(x) * _sigmoid_derivative(primal(x)))
    end
    @eval function Mooncake.rrule!!(
        ::CoDual{typeof(NNlib.$f)}, x::CoDual{P}
    ) where {P<:IEEEFloat}
        d = _sigmoid_derivative(primal(x))
        sigmoid_pb!!(dΩ::P) = NoRData(), dΩ * d
        return zero_fcodual(NNlib.$f(primal(x))), sigmoid_pb!!
    end
    # GPU elementwise and reduction rules evaluate the whole fused broadcast on `NDual`s
    # inside one kernel, so they never reach the rules above and need the derivative too.
    @eval @inline function NNlib.$f(x::NDual{P,N}) where {P<:IEEEFloat,N}
        d = _sigmoid_derivative(x.value)
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
@inline function _tanh_fast_derivative(x)
    u = NNlib._exp(-2 * abs(x))
    return 4u / (one(x) + u)^2
end

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
