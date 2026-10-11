using ChainRulesCore: ChainRulesCore, NoTangent, unthunk
using NNlib: _check_affine, _unbroadcast
import NNlib: instancenorm, groupnorm, layernorm

# cuDNN-accelerated instancenorm / groupnorm / layernorm.
#
# cuDNN has no dedicated instance/group/layer-norm kernels, but each is a batch
# normalization over a re-grouped view of `x`: reshape so the axes normalised over
# become the spatial+batch reduction of a `(S, 1, C′, 1)` tensor, and cuDNN's tuned
# `batchnorm` fast path does the standardisation. The affine transform is then applied
# generically (differentiably) since its parameters follow the *original* channel
# layout, which cuDNN's per-channel view no longer matches (except for batchnorm).
#
# Reusing the existing cuDNN `batchnorm` forward/backward means the pullbacks need no
# new cuDNN plumbing — the backward simply feeds the (affine-adjusted) cotangent back
# through the cuDNN `batchnorm` `rrule`.
#
# Only the pure-normalisation cases route to cuDNN. These fall through to the generic
# differentiable implementation in NNlib core:
# - instancenorm with running statistics (`running_mean`/`running_var` given),
# - half-precision (`Float16`/`BFloat16`) and other non-`_NormFloat` eltypes, and
# - layernorm over non-leading `dims`.

# Eltypes for which cuDNN batchnorm's parameter type equals the value type, so the
# generic affine needs no precision juggling. Half precision uses the generic path.
const _NormFloat = Union{Float32,Float64}

# Standardise `x4` (an `(S, 1, C′, 1)` reshape) to zero mean / unit variance over its
# spatial+batch axes via the cuDNN `batchnorm` fast path (affine off). There are never
# running statistics here, so normalisation always uses the batch statistics
# (`training=true`). Returns `(x̂4, dx4)` where `dx4(dŷ4)` is the input gradient through
# the cuDNN `batchnorm` backward — reusing the existing cuDNN `rrule`, no new plumbing.
function _bn_standardize(x4::DenseCuArray; eps)
    x̂4, bn_back = ChainRulesCore.rrule(batchnorm, nothing, nothing, x4, nothing, nothing, 0.1f0;
                                       eps, training=true, track_stats=false)
    dx4(dŷ4) = bn_back(dŷ4)[4]  # rrule tangents: (self, dg, db, dx, drm, drv, dmom)
    return x̂4, dx4
end

# Per-channel (N-1) affine shape and the reduction dims for the per-channel parameter
# gradients (every dimension except the channel).
_ch_shape(x::AbstractArray{<:Any,N}) where {N} = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
_ch_reddims(::AbstractArray{<:Any,N}) where {N} = ntuple(i -> i < N-1 ? i : i+1, N-1)

# --- instancenorm ------------------------------------------------------------
# Reduce over the spatial dims per (channel, sample): fold (channel, batch) into the
# cuDNN channel dimension with batch 1, so cuDNN reduces over the spatial axes only.
_in_reshape(x::AbstractArray{T,N}) where {T,N} =
    reshape(x, prod(ntuple(i -> size(x, i), N-2)), 1, size(x, N-1) * size(x, N), 1)

function instancenorm(g, b, x::DenseCuArray{T,N}, running_mean::Nothing=nothing, running_var::Nothing=nothing,
                      momentum=0.1f0; eps=1f-5, training::Bool=true, track_stats::Bool=false) where {T<:_NormFloat,N}
    N > 2 || throw(ArgumentError("instancenorm expects an array with at least 3 dimensions, got $N"))
    _check_affine(g, b)
    x̂ = reshape(batchnorm(nothing, nothing, _in_reshape(x), nothing, nothing, 0.1f0;
                          eps, training=true, track_stats=false), size(x))
    g === nothing && return x̂
    cs = _ch_shape(x)
    return reshape(g, cs) .* x̂ .+ reshape(b, cs)
end

function ChainRulesCore.rrule(::typeof(instancenorm), g, b, x::DenseCuArray{T,N},
                              running_mean::Nothing=nothing, running_var::Nothing=nothing, momentum=0.1f0;
                              eps=1f-5, training::Bool=true, track_stats::Bool=false) where {T<:_NormFloat,N}
    N > 2 || throw(ArgumentError("instancenorm expects an array with at least 3 dimensions, got $N"))
    _check_affine(g, b)
    x̂4, dx4 = _bn_standardize(_in_reshape(x); eps)
    x̂ = reshape(x̂4, size(x))
    cs, rd = _ch_shape(x), _ch_reddims(x)
    y = g === nothing ? x̂ : reshape(g, cs) .* x̂ .+ reshape(b, cs)
    function instancenorm_pullback(Δraw)
        Δ = unthunk(Δraw)
        if g === nothing
            dg = db = NoTangent(); dx̂ = Δ
        else
            dg = reshape(sum(Δ .* x̂; dims=rd), size(g)); db = reshape(sum(Δ; dims=rd), size(b))
            dx̂ = reshape(g, cs) .* Δ
        end
        dx = reshape(dx4(reshape(dx̂, size(x̂4))), size(x))
        (NoTangent(), dg, db, dx, NoTangent(), NoTangent(), NoTangent())
    end
    return y, instancenorm_pullback
end

# --- groupnorm ---------------------------------------------------------------
# Reduce over the spatial dims and the (C÷G) channels within each group, per (group,
# sample): fold (spatial, C÷G) into the cuDNN spatial dimension and (group, batch) into
# the cuDNN channel dimension with batch 1.
function _gn_reshape(x::AbstractArray{T,N}, G) where {T,N}
    C = size(x, N-1)
    S = prod(ntuple(i -> size(x, i), N-2)) * (C ÷ G)
    return reshape(x, S, 1, G * size(x, N), 1)
end

function groupnorm(g, b, x::DenseCuArray{T,N}, G::Integer; eps=1f-5) where {T<:_NormFloat,N}
    N > 2 || throw(ArgumentError("groupnorm expects an array with at least 3 dimensions, got $N"))
    C = size(x, N-1)
    C % G == 0 || throw(ArgumentError("the number of groups G=$G must divide the number of channels C=$C"))
    _check_affine(g, b)
    x̂ = reshape(batchnorm(nothing, nothing, _gn_reshape(x, G), nothing, nothing, 0.1f0;
                          eps, training=true, track_stats=false), size(x))
    g === nothing && return x̂
    cs = _ch_shape(x)
    return reshape(g, cs) .* x̂ .+ reshape(b, cs)
end

function ChainRulesCore.rrule(::typeof(groupnorm), g, b, x::DenseCuArray{T,N}, G::Integer;
                              eps=1f-5) where {T<:_NormFloat,N}
    N > 2 || throw(ArgumentError("groupnorm expects an array with at least 3 dimensions, got $N"))
    C = size(x, N-1)
    C % G == 0 || throw(ArgumentError("the number of groups G=$G must divide the number of channels C=$C"))
    _check_affine(g, b)
    x̂4, dx4 = _bn_standardize(_gn_reshape(x, G); eps)
    x̂ = reshape(x̂4, size(x))
    cs, rd = _ch_shape(x), _ch_reddims(x)
    y = g === nothing ? x̂ : reshape(g, cs) .* x̂ .+ reshape(b, cs)
    function groupnorm_pullback(Δraw)
        Δ = unthunk(Δraw)
        if g === nothing
            dg = db = NoTangent(); dx̂ = Δ
        else
            dg = reshape(sum(Δ .* x̂; dims=rd), size(g)); db = reshape(sum(Δ; dims=rd), size(b))
            dx̂ = reshape(g, cs) .* Δ
        end
        dx = reshape(dx4(reshape(dx̂, size(x̂4))), size(x))
        (NoTangent(), dg, db, dx, NoTangent())
    end
    return y, groupnorm_pullback
end

# --- layernorm ---------------------------------------------------------------
# Reduce over the leading `dims`: fold them into the cuDNN spatial dimension and the
# remaining dims into the cuDNN channel dimension with batch 1. Only leading
# `dims == 1:k` reshape contiguously; other `dims` fall back to the generic path.
_leading_dims(dims) = (d = sort!(collect(dims isa Integer ? (dims,) : dims)); d == collect(1:length(d)))

function _ln_reshape(x::AbstractArray, k)
    S = prod(ntuple(i -> size(x, i), k))
    return reshape(x, S, 1, length(x) ÷ S, 1)
end

function layernorm(g, b, x::DenseCuArray{T}; dims=1, eps=1f-5) where {T<:_NormFloat}
    _leading_dims(dims) || return invoke(layernorm, Tuple{Any,Any,AbstractArray}, g, b, x; dims, eps)
    _check_affine(g, b)
    k = dims isa Integer ? 1 : length(dims)
    x̂ = reshape(batchnorm(nothing, nothing, _ln_reshape(x, k), nothing, nothing, 0.1f0;
                          eps, training=true, track_stats=false), size(x))
    g === nothing && return x̂
    return g .* x̂ .+ b
end

function ChainRulesCore.rrule(::typeof(layernorm), g, b, x::DenseCuArray{T}; dims=1, eps=1f-5) where {T<:_NormFloat}
    _leading_dims(dims) ||
        return invoke(ChainRulesCore.rrule, Tuple{typeof(layernorm),Any,Any,AbstractArray}, layernorm, g, b, x; dims, eps)
    _check_affine(g, b)
    k = dims isa Integer ? 1 : length(dims)
    x̂4, dx4 = _bn_standardize(_ln_reshape(x, k); eps)
    x̂ = reshape(x̂4, size(x))
    y = g === nothing ? x̂ : g .* x̂ .+ b
    function layernorm_pullback(Δraw)
        Δ = unthunk(Δraw)
        if g === nothing
            dg = db = NoTangent(); dx̂ = Δ
        else
            dg = _unbroadcast(Δ .* x̂, g); db = _unbroadcast(Δ, b); dx̂ = g .* Δ
        end
        dx = reshape(dx4(reshape(dx̂, size(x̂4))), size(x))
        (NoTangent(), dg, db, dx)
    end
    return y, layernorm_pullback
end
