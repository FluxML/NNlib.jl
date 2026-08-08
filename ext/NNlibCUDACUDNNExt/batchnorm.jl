using cuDNN: CUDNN_BN_MIN_EPSILON, cudnnBatchNormalizationBackward,
             cudnnBatchNormalizationForwardInference, CUDNN_BATCHNORM_SPATIAL,
             cudnnBatchNormalizationForwardTraining
using ChainRulesCore: ChainRulesCore, NoTangent, unthunk
import NNlib: batchnorm, ∇batchnorm

# TODO: replace with new cudnn normalization interface
# https://github.com/JuliaGPU/CUDA.jl/blob/master/lib/cudnn/normalization.jl

mutable struct BNCache
  mean
  ivar
end

BNCache() = BNCache(nothing, nothing)

@inline _wsize(x::AbstractArray{<:Any,N}) where N = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)

# cuDNN requires the batchnorm affine and statistics tensors (scale `g`, bias `b`,
# running mean/var, saved mean/ivar) to be Float32 when the feature maps `x` are
# half precision (Float16/BFloat16), and to share the feature-map type otherwise.
# We accept only the array-type combinations that satisfy this contract rather than
# silently converting, so a type mismatch surfaces as a clear error here instead of
# an opaque CUDNN_STATUS_BAD_PARAM. `P == bnparam(T)` is the required parameter
# eltype for feature-map eltype `T`.
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

function batchnorm(g::Nothing, b::Nothing, x::DenseCuArray,
                   running_mean, running_var, momentum; kws...)
  affine_sz = _wsize(x)
  P = bnparam(eltype(x))
  g = fill!(similar(x, P, affine_sz), 1)
  b = fill!(similar(x, P, affine_sz), 0)
  return batchnorm(g, b, x, running_mean, running_var, momentum; kws...)
end

# NOTE: CuDNN supports only 4D and 5D Tensors for BatchNorm Operations
# so reshape a 2D Tensor into 4D
function batchnorm(g::DenseCuArray{P}, b::DenseCuArray{P}, x::DenseCuArray{T,2},
                   running_mean, running_var, momentum; kws...) where {T<:CUDNNFloat, P}
  _check_bn_param_types(T, P, running_mean, running_var)
  x = reshape(x, 1, 1, size(x, 1), size(x, 2))
  y = batchnorm(g, b, x, running_mean, running_var, momentum; kws...)
  return dropdims(y, dims = (1, 2))
end

# Likewise, reshape a 3D (W, C, N) Tensor into 4D (1, W, C, N) so the feature
# (channel) dimension stays second-to-last, as cuDNN batchnorm requires.
function batchnorm(g::DenseCuArray{P}, b::DenseCuArray{P}, x::DenseCuArray{T,3},
                   running_mean, running_var, momentum; kws...) where {T<:CUDNNFloat, P}
  _check_bn_param_types(T, P, running_mean, running_var)
  x = reshape(x, 1, size(x, 1), size(x, 2), size(x, 3))
  y = batchnorm(g, b, x, running_mean, running_var, momentum; kws...)
  return dropdims(y, dims = 1)
end

function batchnorm(g::DenseCuArray{P}, b::DenseCuArray{P}, x::Union{DenseCuArray{T,4},DenseCuArray{T,5}},
                   running_mean, running_var, momentum; kws...) where {T<:CUDNNFloat, P}
  _check_bn_param_types(T, P, running_mean, running_var)
  cudnnBNForward!(similar(x), g, b, x, running_mean, running_var, momentum; kws...)
end

function cudnnBNForward!(y::DenseCuArray{T}, g::DenseCuArray{P}, b::DenseCuArray{P}, x::DenseCuArray{T},
                        running_mean, running_var, momentum;
                        cache = nothing,
                        alpha = T(1), beta = T(0),
                        eps = T(1e-5),
                        training = true,
                        affine = true,
                        track_stats = true) where {T<:CUDNNFloat, P}
  dims = _wsize(x)
  if eps < CUDNN_BN_MIN_EPSILON
    @warn "eps $eps is too small for CuDNN, setting to CUDNN_BN_MIN_EPSILON=$CUDNN_BN_MIN_EPSILON"
    eps = CUDNN_BN_MIN_EPSILON
  end

  if running_mean === nothing || running_var === nothing
    running_mean !== running_var && throw(ArgumentError("both or neither of running_mean and running_var must be nothing"))
    if track_stats || !training
      running_mean = fill!(similar(x, P, dims), 0)
      running_var = fill!(similar(x, P, dims), 1)
    end
  end

  xd = cudnnTensorDescriptor(x)
  yd = cudnnTensorDescriptor(y)
  gd = cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, cudnnDataType(P), Cint(length(dims)), dim4(dims,Val(CUDNN_TENSOR_NCHW)))

  if training
    if !track_stats
      running_mean = CU_NULL
      running_var = CU_NULL
    end

    if cache !== nothing
      mean = fill!(similar(x, P, dims), 0)
      ivar = fill!(similar(x, P, dims), 1)
    else
      mean = CU_NULL
      ivar = CU_NULL
    end

    cudnnBatchNormalizationForwardTraining(handle(), CUDNN_BATCHNORM_SPATIAL, scalingParameter(T, alpha), scalingParameter(T, beta), xd, x, yd, y, gd, g, b, momentum, running_mean, running_var, eps, mean, ivar)

    if cache !== nothing
      cache.mean = mean
      cache.ivar = ivar
    end
  else
    if track_stats
      cudnnBatchNormalizationForwardInference(handle(), CUDNN_BATCHNORM_SPATIAL, scalingParameter(T, alpha), scalingParameter(T, beta), xd, x, yd, y, gd, g, b, running_mean, running_var, eps)
    else
      # cudnnBatchNormalizationForwardInference does not accept CV_NULL for running_mean
      # and running_var. We could calculate mean and var of `x` here, but instead use
      # cudnnBatchNormalizationFowardTraining. cudnnBatchNormalizationForwardTraining does
      # accept CV_NULL and will calculate mean and var itself.
      cudnnBatchNormalizationForwardTraining(handle(), CUDNN_BATCHNORM_SPATIAL, scalingParameter(T, alpha), scalingParameter(T, beta), xd, x, yd, y, gd, g, b, momentum, CU_NULL, CU_NULL, eps, CU_NULL, CU_NULL)
    end
  end
  return y
end

function ∇batchnorm(g::Nothing, b::Nothing, x::DenseCuArray, dy::DenseCuArray,
                    running_mean, running_var, momentum; kws...)
  affine_sz = _wsize(x)
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


function ∇batchnorm(g::DenseCuArray{P}, b::DenseCuArray{P}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                    running_mean, running_var, momentum;
                    affine=true, kws...) where {T<:CUDNNFloat, P}
  _check_bn_param_types(T, P, running_mean, running_var)
  dg = similar(g)
  db = similar(b)
  dx = similar(x)
  cudnnBNBackward!(dg, g, db, dx, x, dy, running_mean, running_var, T(momentum); kws...)
  if affine
    (dg, db, dx)
  else
    # cuDNN always calculates dg and db, therefore we just have to drop them
    (nothing, nothing, dx)
  end
end

function cudnnBNBackward!(dg::DenseCuArray{P}, g::DenseCuArray{P}, db::DenseCuArray{P},
                          dx::DenseCuArray{T}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                          running_mean, running_var,
                          momentum; cache = nothing, eps = T(1e-5),
                          alpha = T(1), beta = T(0),
                          dalpha = T(1), dbeta = T(0), training = true,
                          track_stats = true) where {T<:CUDNNFloat, P}
  if eps < CUDNN_BN_MIN_EPSILON
    @warn "eps $eps is too small for CuDNN, setting to CUDNN_BN_MIN_EPSILON=$CUDNN_BN_MIN_EPSILON"
    eps = CUDNN_BN_MIN_EPSILON
  end

  # `cudnnBatchNormalizationBackward` only implements the *training*-mode gradient: it
  # differentiates through the per-batch mean/variance and ignores the running statistics.
  # In inference mode with tracked statistics, the forward pass instead normalises by the
  # *fixed* running mean/variance (`cudnnBatchNormalizationForwardInference`), so its
  # gradient is a plain per-channel affine rescaling. Calling the cuDNN backward here yields
  # silently wrong `dx`/`dg` (FluxML/Flux.jl#2179), so compute the inference gradient
  # directly. This assumes the default scaling parameters (alpha=1, beta=0, dalpha=1,
  # dbeta=0), which is the only combination Flux/NNlib exercise.
  if !training && track_stats && running_mean isa DenseCuArray && running_var isa DenseCuArray
    N = ndims(x)
    ws = _wsize(x)                                    # (1,…,1,C,1): channel dim is N-1
    rstd = reshape(inv.(sqrt.(running_var .+ P(eps))), ws)
    dx .= dy .* reshape(g, ws) .* rstd
    reddims = ntuple(i -> i < N-1 ? i : i+1, N-1)     # all dims except the channel dim
    x̂ = (x .- reshape(running_mean, ws)) .* rstd
    # `dg`/`db` may arrive as length-C vectors (affine params) or as `_wsize`-shaped
    # arrays (the affine=false fill path), so reshape the reduction to their layout.
    dg .= reshape(sum(dy .* x̂; dims = reddims), size(dg))
    db .= reshape(sum(dy; dims = reddims), size(db))
    return
  end

  if !track_stats
    running_mean = CU_NULL
    running_var = CU_NULL
  end

  xd = cudnnTensorDescriptor(x)
  dyd = cudnnTensorDescriptor(dy)
  dxd = cudnnTensorDescriptor(dx)
  gd = cudnnTensorDescriptor(CUDNN_TENSOR_NCHW, cudnnDataType(P), Cint(length(_wsize(x))), dim4(_wsize(x),Val(CUDNN_TENSOR_NCHW)))
  if cache !== nothing
    @debug "fetching mean and ivar from the cache"
    mean, ivar = cache.mean, cache.ivar
  else
    mean, ivar = CU_NULL, CU_NULL
  end

  cudnnBatchNormalizationBackward(handle(), CUDNN_BATCHNORM_SPATIAL,
        scalingParameter(T, alpha), scalingParameter(T, beta), scalingParameter(T, dalpha), scalingParameter(T, dbeta),
        xd, x, dyd, dy, dxd, dx, gd, g, dg, db, eps, mean, ivar)
end

# GPU differentiation of the cuDNN fast path. The generic `batchnorm` in NNlib core
# has no `rrule`, so on the CPU (and other array types) it is differentiated through
# the standard AD path instead.
function ChainRulesCore.rrule(::typeof(batchnorm), g, b, x::DenseCuArray,
                              running_mean, running_var, momentum; kw...)
  y = batchnorm(g, b, x, running_mean, running_var, momentum; kw...)
  function batchnorm_pullback(Δ)
    grad = ∇batchnorm(g, b, x, unthunk(Δ), running_mean, running_var, momentum; kw...)
    (NoTangent(), grad..., NoTangent(), NoTangent(), NoTangent())
  end
  y, batchnorm_pullback
end
