using cuDNN: CUDNN_BN_MIN_EPSILON, batchnorm_gradient!, batchnorm_inference!,
             batchnorm_training!
import NNlib: batchnorm, ∇batchnorm

mutable struct BNCache
  mean
  ivar
end

BNCache() = BNCache(nothing, nothing)

@inline _wsize(x::AbstractArray{<:Any,N}) where N = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)

function batchnorm(g::Nothing, b::Nothing, x::DenseCuArray,
                   running_mean, running_var, momentum; kws...)
  affine_sz = _wsize(x)
  g = fill!(similar(x, affine_sz), 1)
  b = fill!(similar(x, affine_sz), 0)
  return batchnorm(g, b, x, running_mean, running_var, momentum; kws...)
end

# NOTE: CuDNN supports only 4D and 5D Tensors for BatchNorm Operations
# so reshape a 2D Tensor into 4D
function batchnorm(g::DenseCuArray{T}, b::DenseCuArray{T}, x::DenseCuArray{T,2},
                   running_mean, running_var, momentum; kws...) where T<:CUDNNFloat
  x = reshape(x, 1, 1, size(x, 1), size(x, 2))
  y = batchnorm(g, b, x, running_mean, running_var, momentum; kws...)
  return dropdims(y, dims = (1, 2))
end

function batchnorm(g::DenseCuArray{T}, b::DenseCuArray{T}, x::Union{DenseCuArray{T,4},DenseCuArray{T,5}},
                   running_mean, running_var, momentum; kws...) where T<:CUDNNFloat
  cudnnBNForward!(similar(x), g, b, x, running_mean, running_var, momentum; kws...)
end

function cudnnBNForward!(y::DenseCuArray{T}, g::DenseCuArray{T}, b::DenseCuArray{T}, x::DenseCuArray{T},
                        running_mean, running_var, momentum;
                        cache = nothing,
                        alpha = T(1), beta = T(0),
                        eps = T(1e-5),
                        training = true,
                        affine = true,
                        track_stats = true) where T<:CUDNNFloat
  dims = _wsize(x)
  if eps < CUDNN_BN_MIN_EPSILON
    @warn "eps $eps is too small for CuDNN, setting to CUDNN_BN_MIN_EPSILON=$CUDNN_BN_MIN_EPSILON"
    eps = CUDNN_BN_MIN_EPSILON
  end

  if running_mean === nothing || running_var === nothing
    running_mean !== running_var && throw(ArgumentError("both or neither of running_mean and running_var must be nothing"))
    if track_stats || !training
      running_mean = fill!(similar(x, dims), 0)
      running_var = fill!(similar(x, dims), 1)
    end
  end

  if training
    rm = track_stats ? running_mean : nothing
    rv = track_stats ? running_var : nothing
    mean, ivar = batchnorm_training!(y, x, g, b; running_mean=rm, running_var=rv,
                                     momentum, epsilon=eps, alpha, beta)
    if cache !== nothing
      cache.mean = mean
      cache.ivar = ivar
    end
  else
    if track_stats
      batchnorm_inference!(y, x, g, b, running_mean, running_var; epsilon=eps,
                           alpha, beta)
    else
      batchnorm_training!(y, x, g, b; momentum, epsilon=eps, alpha, beta)
    end
  end
  return y
end

function ∇batchnorm(g::Nothing, b::Nothing, x::DenseCuArray, dy::DenseCuArray,
                    running_mean, running_var, momentum; kws...)
  affine_sz = _wsize(x)
  g = fill!(similar(x, affine_sz), 1)
  b = fill!(similar(x, affine_sz), 0)
  return ∇batchnorm(g, b, x, dy, running_mean, running_var, momentum; kws...)
end

function ∇batchnorm(g::DenseCuArray{T}, b::DenseCuArray{T}, x::DenseCuArray{T, 2}, dy::DenseCuArray{T, 2},
            running_mean, running_var, momentum;
            kws...) where T<:CUDNNFloat
  dg, db, dx = ∇batchnorm(g, b, reshape(x, 1, 1, size(x, 1), size(x, 2)), reshape(dy, 1, 1, size(dy, 1),
                          size(dy, 2)), running_mean, running_var, momentum; kws...)
  (dg, db, dropdims(dx, dims = (1, 2)))
end


function ∇batchnorm(g::DenseCuArray{T}, b::DenseCuArray{T}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                    running_mean, running_var, momentum;
                    affine=true, kws...) where T<:CUDNNFloat
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

function cudnnBNBackward!(dg::DenseCuArray{T}, g::DenseCuArray{T}, db::DenseCuArray{T},
                          dx::DenseCuArray{T}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                          running_mean, running_var,
                          momentum; cache = nothing, eps = T(1e-5),
                          alpha = T(1), beta = T(0),
                          dalpha = T(1), dbeta = T(0), training = true,
                          track_stats = true) where T<:CUDNNFloat
  if cache !== nothing
    @debug "fetching mean and ivar from the cache"
    mean, ivar = cache.mean, cache.ivar
  else
    tmp = similar(x)
    mean, ivar = batchnorm_training!(tmp, x, g, db; epsilon=eps)
  end

  if eps < CUDNN_BN_MIN_EPSILON
    @warn "eps $eps is too small for CuDNN, setting to CUDNN_BN_MIN_EPSILON=$CUDNN_BN_MIN_EPSILON"
    eps = CUDNN_BN_MIN_EPSILON
  end

  batchnorm_gradient!(dx, dg, db, dy, x, g, mean, ivar; epsilon=eps, alpha, beta,
                      dalpha, dbeta)
end
