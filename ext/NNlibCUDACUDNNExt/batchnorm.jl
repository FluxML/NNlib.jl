using cuDNN: batchnorm_gradient!, batchnorm_inference!, batchnorm_training!,
             graph_unsupported
using ChainRulesCore: ChainRulesCore, NoTangent, unthunk
using NNlib: within_gradient
import NNlib: batchnorm, ∇batchnorm

# Batch statistics saved by the forward pass (`mean` and `ivar = 1/√(var + eps)`) for
# the backward pass to consume, as in cuDNN's own interface.
mutable struct BNCache
  mean
  ivar
end

BNCache() = BNCache(nothing, nothing)

# cuDNN batchnorm handles 4D and 5D tensors; 2D and 3D inputs are reshaped to 4D.
const BNArray{T} = Union{DenseCuArray{T,4},DenseCuArray{T,5}}
const BNReshapedArray{T} = Union{DenseCuArray{T,2},DenseCuArray{T,3}}

@inline batchnorm_param_size(x::AbstractArray{<:Any,N}) where {N} =
  ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)

# Every dimension except the channel one.
@inline batchnorm_reduce_dims(::AbstractArray{<:Any,N}) where {N} =
  (ntuple(identity, N-2)..., N)

# cuDNN requires the batchnorm affine and statistics tensors (scale `g`, bias `b`,
# running mean/var, saved mean/ivar) to be Float32 when the feature maps `x` are
# half precision (Float16/BFloat16), and to share the feature-map type otherwise.
# We accept only the array-type combinations that satisfy this contract rather than
# silently converting, so a type mismatch surfaces as a clear error here and no
# hidden copies are made. `P == bnparam(T)` is the required parameter eltype for
# feature-map eltype `T`.
bnparam(::Type{Float16})  = Float32
bnparam(::Type{BFloat16}) = Float32
bnparam(::Type{T}) where {T<:CUDNNFloat} = T

@inline function _check_bn_param_types(::Type{T}, ::Type{P}, running_mean, running_var) where {T,P}
  P === bnparam(T) || throw(ArgumentError(
    "cuDNN batchnorm on $T feature maps requires $(bnparam(T)) scale/bias tensors, got $P. " *
    "cuDNN needs Float32 affine parameters for half-precision (Float16/BFloat16) data."))
  for s in (running_mean, running_var)
    s === nothing || eltype(s) === P || throw(ArgumentError(
      "cuDNN batchnorm on $T feature maps requires $(bnparam(T)) running statistics, got $(eltype(s))."))
  end
  return nothing
end

# The running statistics normalise the input only in inference mode, and are updated
# only in training mode. Untracked statistics are neither read nor updated.
function batchnorm_running_stats(running_mean, running_var, track_stats)
  (running_mean === nothing) == (running_var === nothing) ||
    throw(ArgumentError("both or neither of running_mean and running_var must be nothing"))
  return track_stats ? (running_mean, running_var) : (nothing, nothing)
end

# Batch statistics in the form cuDNN's gradient expects, for when the forward pass did
# not save them. Half-precision data accumulates in the parameter type `P`.
function batchnorm_statistics(x::DenseCuArray{T}, ::Type{P}, eps) where {T,P}
  dims = batchnorm_reduce_dims(x)
  xP = P === T ? x : P.(x)
  μ = mean(xP; dims)
  σ² = var(xP; mean=μ, dims, corrected=false)
  return μ, @. inv(sqrt(σ² + P(eps)))
end

# Restricted to cuDNN eltypes so that non-CUDNNFloat arrays (notably `ForwardDiff.Dual`
# under forward-over-reverse) fall through to the generic differentiable `batchnorm`
# rather than erroring in `bnparam`/cuDNN. See FluxML/Flux.jl#2154.
function batchnorm(g::Nothing, b::Nothing, x::DenseCuArray{<:CUDNNFloat},
                   running_mean, running_var, momentum; kws...)
  affine_sz = batchnorm_param_size(x)
  P = bnparam(eltype(x))
  g = fill!(similar(x, P, affine_sz), 1)
  b = fill!(similar(x, P, affine_sz), 0)
  return batchnorm(g, b, x, running_mean, running_var, momentum; kws...)
end

# Reshape a 2D (C, N) tensor into 4D (1, 1, C, N).
function batchnorm(g::DenseCuArray{P}, b::DenseCuArray{P}, x::DenseCuArray{T,2},
                   running_mean, running_var, momentum; kws...) where {T<:CUDNNFloat, P}
  _check_bn_param_types(T, P, running_mean, running_var)
  x = reshape(x, 1, 1, size(x, 1), size(x, 2))
  y = batchnorm(g, b, x, running_mean, running_var, momentum; kws...)
  return dropdims(y, dims = (1, 2))
end

# Likewise, reshape a 3D (W, C, N) tensor into 4D (1, W, C, N) so the feature
# (channel) dimension stays second-to-last, as cuDNN batchnorm requires.
function batchnorm(g::DenseCuArray{P}, b::DenseCuArray{P}, x::DenseCuArray{T,3},
                   running_mean, running_var, momentum; kws...) where {T<:CUDNNFloat, P}
  _check_bn_param_types(T, P, running_mean, running_var)
  x = reshape(x, 1, size(x, 1), size(x, 2), size(x, 3))
  y = batchnorm(g, b, x, running_mean, running_var, momentum; kws...)
  return dropdims(y, dims = 1)
end

function batchnorm(g::DenseCuArray{P}, b::DenseCuArray{P}, x::BNArray{T},
                   running_mean, running_var, momentum; kws...) where {T<:CUDNNFloat, P}
  _check_bn_param_types(T, P, running_mean, running_var)
  cudnnBNForward!(similar(x), g, b, x, running_mean, running_var, momentum; kws...)
end

# `alpha` and `beta` are forwarded to cuDNN, which only supports the defaults.
function cudnnBNForward!(y::DenseCuArray{T}, g::DenseCuArray{P}, b::DenseCuArray{P}, x::DenseCuArray{T},
                        running_mean, running_var, momentum;
                        cache = nothing,
                        alpha = 1, beta = 0,
                        eps = T(1e-5),
                        training = true,
                        affine = true,
                        track_stats = true) where {T<:CUDNNFloat, P}
  running_mean, running_var = batchnorm_running_stats(running_mean, running_var, track_stats)
  try
    if !training && running_mean !== nothing
      batchnorm_inference!(y, x, g, b, running_mean, running_var; epsilon=eps, alpha, beta)
    else
      # Without running statistics to normalise with, inference uses the batch
      # statistics too, as the generic `batchnorm` does.
      μ, ivar = batchnorm_training!(y, x, g, b; running_mean, running_var, momentum,
                                    epsilon=eps, alpha, beta)
      if cache !== nothing
        cache.mean = μ
        cache.ivar = ivar
      end
    end
  catch e
    # cuDNN has no engine for this configuration: use the generic implementation.
    graph_unsupported(e) || rethrow()
    copyto!(y, _batchnorm_generic(g, b, x, running_mean, running_var, momentum;
                                  eps, training, track_stats))
  end
  return y
end

function ∇batchnorm(g::Nothing, b::Nothing, x::DenseCuArray, dy::DenseCuArray,
                    running_mean, running_var, momentum; kws...)
  # Under nested AD go straight to the differentiable generic VJP, which handles the
  # affine-free case natively — bypassing both the non-differentiable cuDNN backward
  # and the `fill!(similar(...))` allocation the outer AD can't trace (Flux.jl#2154).
  if within_gradient(x) || within_gradient(dy)
    return _∇batchnorm_generic(nothing, nothing, x, dy, running_mean, running_var; kws...)
  end
  affine_sz = batchnorm_param_size(x)
  P = bnparam(eltype(x))
  g = fill!(similar(x, P, affine_sz), 1)
  b = fill!(similar(x, P, affine_sz), 0)
  return ∇batchnorm(g, b, x, dy, running_mean, running_var, momentum; kws...)
end

function ∇batchnorm(g::DenseCuArray{P}, b::DenseCuArray{P}, x::DenseCuArray{T, 2}, dy::DenseCuArray{T, 2},
            running_mean, running_var, momentum;
            kws...) where {T<:CUDNNFloat, P}
  _check_bn_param_types(T, P, running_mean, running_var)
  dg, db, dx = ∇batchnorm(g, b, reshape(x, 1, 1, size(x, 1), size(x, 2)), reshape(dy, 1, 1, size(dy, 1),
                          size(dy, 2)), running_mean, running_var, momentum; kws...)
  (dg, db, dropdims(dx, dims = (1, 2)))
end

function ∇batchnorm(g::DenseCuArray{P}, b::DenseCuArray{P}, x::DenseCuArray{T, 3}, dy::DenseCuArray{T, 3},
            running_mean, running_var, momentum;
            kws...) where {T<:CUDNNFloat, P}
  _check_bn_param_types(T, P, running_mean, running_var)
  dg, db, dx = ∇batchnorm(g, b, reshape(x, 1, size(x, 1), size(x, 2), size(x, 3)),
                          reshape(dy, 1, size(dy, 1), size(dy, 2), size(dy, 3)),
                          running_mean, running_var, momentum; kws...)
  (dg, db, dropdims(dx, dims = 1))
end

function ∇batchnorm(g::DenseCuArray{P}, b::DenseCuArray{P}, x::BNArray{T}, dy::BNArray{T},
                    running_mean, running_var, momentum;
                    affine=true, kws...) where {T<:CUDNNFloat, P}
  _check_bn_param_types(T, P, running_mean, running_var)
  # The cuDNN backward is a non-differentiable foreigncall, so when this VJP is itself
  # being differentiated (second-order / nested AD, e.g. a Hessian-vector product)
  # reverse-mode can't trace it. Route through the generic broadcast VJP in NNlib core,
  # which is built from `mean`/`var`/broadcast and is itself differentiable.
  # Mirrors the `within_gradient` fast/slow split in `∇softmax`. See FluxML/Flux.jl#2154.
  # (Forward-over-reverse already avoids cuDNN: `ForwardDiff.Dual` eltypes miss this
  # `T<:CUDNNFloat` method and dispatch straight to the generic `∇batchnorm`.)
  if within_gradient(x) || within_gradient(dy)
    dg, db, dx = _∇batchnorm_generic(g, b, x, dy, running_mean, running_var; kws...)
    return affine ? (dg, db, dx) : (nothing, nothing, dx)
  end
  dg = similar(g)
  db = similar(b)
  dx = similar(x)
  cudnnBNBackward!(dg, g, db, dx, x, dy, running_mean, running_var, momentum; kws...)
  if affine
    (dg, db, dx)
  else
    # cuDNN always calculates dg and db, therefore we just have to drop them
    (nothing, nothing, dx)
  end
end

# Generic batchnorm (the `_norm_layer` engine in NNlib core), for the configurations
# cuDNN has no engine for. The cuDNN `batchnorm` methods above shadow the generic
# `batchnorm` for `CuArray`s, so call the engine directly.
function _batchnorm_generic(g, b, x, running_mean, running_var, momentum;
                            eps, training, track_stats)
  # the engine updates the running statistics as vectors
  running_mean === nothing || (running_mean = vec(running_mean); running_var = vec(running_var))
  return NNlib._norm_layer(g, b, x, running_mean, running_var, batchnorm_reduce_dims(x);
                           training, momentum, eps, track_stats)
end

# Generic, differentiable batchnorm VJP (the `_∇norm_channel` engine in NNlib core).
# The cuDNN `∇batchnorm` methods above shadow the generic `∇batchnorm` for `CuArray`s,
# so call the engine directly to reach the differentiable path for nested AD.
function _∇batchnorm_generic(g, b, x, dy, running_mean, running_var;
                             eps=1f-5, training::Bool=true, kws...)
  return NNlib._∇norm_channel(g, b, x, dy, running_mean, running_var,
                              batchnorm_reduce_dims(x); eps, training)
end

# `alpha`, `beta`, `dalpha` and `dbeta` are forwarded to cuDNN, which only supports the
# defaults.
function cudnnBNBackward!(dg::DenseCuArray{P}, g::DenseCuArray{P}, db::DenseCuArray{P},
                          dx::DenseCuArray{T}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                          running_mean, running_var,
                          momentum; cache = nothing, eps = T(1e-5),
                          alpha = 1, beta = 0,
                          dalpha = 1, dbeta = 0, training = true,
                          track_stats = true) where {T<:CUDNNFloat, P}
  running_mean, running_var = batchnorm_running_stats(running_mean, running_var, track_stats)

  # cuDNN only implements the *training*-mode gradient, which differentiates through the
  # per-batch mean/variance. In inference mode with tracked statistics, the forward pass
  # instead normalises by the *fixed* running mean/variance, so its gradient is a plain
  # per-channel affine rescaling. Calling the cuDNN backward here would yield silently
  # wrong `dx`/`dg` (FluxML/Flux.jl#2179), so compute the inference gradient directly.
  if !training && running_mean !== nothing
    alpha == 1 && beta == 0 && dalpha == 1 && dbeta == 0 ||
      throw(ArgumentError("the batchnorm gradient requires alpha=1, beta=0, dalpha=1, and dbeta=0"))
    ws = batchnorm_param_size(x)                      # (1,…,1,C,1): channel dim is N-1
    rstd = reshape(inv.(sqrt.(running_var .+ P(eps))), ws)
    dx .= dy .* reshape(g, ws) .* rstd
    reddims = batchnorm_reduce_dims(x)
    x̂ = (x .- reshape(running_mean, ws)) .* rstd
    # `dg`/`db` may arrive as length-C vectors (affine params) or as `ws`-shaped
    # arrays (the affine=false fill path), so reshape the reduction to their layout.
    dg .= reshape(sum(dy .* x̂; dims = reddims), size(dg))
    db .= reshape(sum(dy; dims = reddims), size(db))
    return dg, db, dx
  end

  # cuDNN's gradient consumes the batch statistics saved by the forward pass.
  μ, ivar = cache === nothing ? (nothing, nothing) : (cache.mean, cache.ivar)
  if μ === nothing || ivar === nothing
    μ, ivar = batchnorm_statistics(x, P, eps)
  end
  try
    batchnorm_gradient!(dx, dg, db, dy, x, g, μ, ivar; epsilon=eps, alpha, beta, dalpha, dbeta)
  catch e
    # cuDNN has no engine for this configuration: use the generic implementation.
    graph_unsupported(e) || rethrow()
    # (only the shape of the bias matters, which `db` shares)
    gdg, gdb, gdx = _∇batchnorm_generic(g, db, x, dy, nothing, nothing; eps, training=true)
    copyto!(dg, gdg)
    copyto!(db, gdb)
    copyto!(dx, gdx)
  end
  return dg, db, dx
end

# GPU differentiation of the cuDNN fast path. The forward pass saves the batch
# statistics for the pullback, which cuDNN's gradient would otherwise have recomputed.
function ChainRulesCore.rrule(::typeof(batchnorm),
                              g::Union{Nothing,DenseCuArray}, b::Union{Nothing,DenseCuArray},
                              x::Union{BNArray{T},BNReshapedArray{T}},
                              running_mean, running_var, momentum;
                              cache = nothing, kw...) where {T<:CUDNNFloat}
  cache = something(cache, BNCache())
  y = batchnorm(g, b, x, running_mean, running_var, momentum; cache, kw...)
  function batchnorm_pullback(Δ)
    grad = ∇batchnorm(g, b, x, unthunk(Δ), running_mean, running_var, momentum; cache, kw...)
    (NoTangent(), grad..., NoTangent(), NoTangent(), NoTangent())
  end
  y, batchnorm_pullback
end
