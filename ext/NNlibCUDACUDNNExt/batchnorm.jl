using cuDNN: CUDNN_BN_MIN_EPSILON, batchnorm_gradient!, batchnorm_inference!,
             batchnorm_training!, graph_unsupported
import NNlib: batchnorm, ∇batchnorm

mutable struct BNCache
  mean
  ivar
end

BNCache() = BNCache(nothing, nothing)

batchnorm_stat_type(::Type{Float64}) = Float64
batchnorm_stat_type(::Type) = Float32

@inline batchnorm_param_size(x::AbstractArray{<:Any,N}) where {N} =
  ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)

batchnorm_reduce_dims(x) = Tuple([1:ndims(x)-2; ndims(x)])

function batchnorm_param_array(name, a::DenseCuArray, S, dims)
  length(a) == prod(dims) ||
    throw(DimensionMismatch("$name must have $(prod(dims)) elements"))
  a = eltype(a) == S ? a : S.(a)
  return reshape(a, dims)
end

batchnorm_param_array(name, ::Nothing, S, dims) = nothing

function batchnorm_param_array(name, a, S, dims)
  throw(ArgumentError("$name must be a DenseCuArray or nothing"))
end

function batchnorm_statistics(x::DenseCuArray, S, eps)
  dims = batchnorm_reduce_dims(x)
  n = prod(size(x, d) for d in dims)
  xS = eltype(x) == S ? x : S.(x)
  mean = sum(xS; dims) ./ S(n)
  centered = xS .- mean
  variance = sum(abs2, centered; dims) ./ S(n)
  invvar = @. inv(sqrt(variance + S(eps)))
  return xS, mean, variance, invvar, n
end

function batchnorm_generic_forward!(y, g, b, x, running_mean, running_var, momentum;
                                    cache, alpha, beta, eps, training, track_stats)
  S = eltype(g)
  if training || !track_stats
    xS, mean, variance, invvar, n = batchnorm_statistics(x, S, eps)
    if training && track_stats
      correction = n > 1 ? S(n / (n - 1)) : one(S)
      @. running_mean = (1 - momentum) * running_mean + momentum * mean
      @. running_var = (1 - momentum) * running_var + momentum * correction * variance
    end
  else
    xS = eltype(x) == S ? x : S.(x)
    mean, variance = running_mean, running_var
    invvar = @. inv(sqrt(variance + S(eps)))
  end
  if beta == 0
    @. y = alpha * (g * (xS - mean) * invvar + b)
  else
    @. y = alpha * (g * (xS - mean) * invvar + b) + beta * y
  end
  if cache !== nothing
    cache.mean = mean
    cache.ivar = invvar
  end
  return y
end

function batchnorm_generic_backward!(dg, db, dx, x, dy, g, mean, invvar;
                                     alpha, beta, dalpha, dbeta)
  S = eltype(g)
  dims = batchnorm_reduce_dims(x)
  n = prod(size(x, d) for d in dims)
  xS = eltype(x) == S ? x : S.(x)
  dyS = eltype(dy) == S ? dy : S.(dy)
  xhat = @. (xS - mean) * invvar
  new_db = sum(dyS; dims)
  new_dg = sum(dyS .* xhat; dims)
  new_dx = @. g * invvar / n * (n * dyS - new_db - xhat * new_dg)
  beta == 0 ? (dx .= alpha .* new_dx) : (dx .= alpha .* new_dx .+ beta .* dx)
  dbeta == 0 ? (dg .= dalpha .* new_dg) : (dg .= dalpha .* new_dg .+ dbeta .* dg)
  dbeta == 0 ? (db .= dalpha .* new_db) : (db .= dalpha .* new_db .+ dbeta .* db)
  return dg, db, dx
end

function batchnorm(g::Nothing, b::Nothing, x::DenseCuArray,
                   running_mean, running_var, momentum; kws...)
  S = batchnorm_stat_type(eltype(x))
  affine_sz = batchnorm_param_size(x)
  g = fill!(similar(x, S, affine_sz), 1)
  b = fill!(similar(x, S, affine_sz), 0)
  return batchnorm(g, b, x, running_mean, running_var, momentum; kws...)
end

# cuDNN batchnorm accepts rank 4 or 5, so promote matrices to rank 4.
function batchnorm(g::DenseCuArray, b::DenseCuArray, x::DenseCuArray{T,2},
                   running_mean, running_var, momentum; kws...) where {T<:CUDNNFloat}
  x = reshape(x, 1, 1, size(x, 1), size(x, 2))
  y = batchnorm(g, b, x, running_mean, running_var, momentum; kws...)
  return dropdims(y, dims = (1, 2))
end

function batchnorm(g::DenseCuArray, b::DenseCuArray,
                   x::Union{DenseCuArray{T,4},DenseCuArray{T,5}},
                   running_mean, running_var, momentum; kws...) where {T<:CUDNNFloat}
  cudnnBNForward!(similar(x), g, b, x, running_mean, running_var, momentum; kws...)
end

function cudnnBNForward!(y::DenseCuArray{T}, g::DenseCuArray, b::DenseCuArray,
                        x::DenseCuArray{T},
                        running_mean, running_var, momentum;
                        cache = nothing,
                        alpha = T(1), beta = T(0),
                        eps = T(1e-5),
                        training = true,
                        affine = true,
                        track_stats = true) where T<:CUDNNFloat
  dims = batchnorm_param_size(x)
  S = batchnorm_stat_type(T)
  gS = batchnorm_param_array("scale", g, S, dims)
  bS = batchnorm_param_array("bias", b, S, dims)
  if eps < CUDNN_BN_MIN_EPSILON
    @warn "eps $eps is too small for CuDNN, setting to CUDNN_BN_MIN_EPSILON=$CUDNN_BN_MIN_EPSILON"
    eps = CUDNN_BN_MIN_EPSILON
  end

  if running_mean === nothing || running_var === nothing
    running_mean !== running_var && throw(ArgumentError("both or neither of running_mean and running_var must be nothing"))
    if track_stats || !training
      running_mean = fill!(similar(x, S, dims), 0)
      running_var = fill!(similar(x, S, dims), 1)
    end
  end

  original_mean, original_var = running_mean, running_var
  running_mean = batchnorm_param_array("running_mean", running_mean, S, dims)
  running_var = batchnorm_param_array("running_var", running_var, S, dims)

  use_graph = alpha == 1 && beta == 0
  if use_graph
    try
      if training
        rm = track_stats ? running_mean : nothing
        rv = track_stats ? running_var : nothing
        mean, ivar = batchnorm_training!(y, x, gS, bS; running_mean=rm, running_var=rv,
                                         momentum, epsilon=eps)
        if cache !== nothing
          cache.mean = mean
          cache.ivar = ivar
        end
      elseif track_stats
        batchnorm_inference!(y, x, gS, bS, running_mean, running_var; epsilon=eps)
      else
        batchnorm_training!(y, x, gS, bS; momentum, epsilon=eps)
      end
    catch e
      graph_unsupported(e) || rethrow()
      batchnorm_generic_forward!(y, gS, bS, x, running_mean, running_var, momentum;
                                 cache, alpha, beta, eps, training, track_stats)
    end
  else
    batchnorm_generic_forward!(y, gS, bS, x, running_mean, running_var, momentum;
                               cache, alpha, beta, eps, training, track_stats)
  end
  if training && track_stats
    running_mean === original_mean ||
      (original_mean .= reshape(running_mean, size(original_mean)))
    running_var === original_var ||
      (original_var .= reshape(running_var, size(original_var)))
  end
  return y
end

function ∇batchnorm(g::Nothing, b::Nothing, x::DenseCuArray, dy::DenseCuArray,
                    running_mean, running_var, momentum; kws...)
  S = batchnorm_stat_type(eltype(x))
  affine_sz = batchnorm_param_size(x)
  g = fill!(similar(x, S, affine_sz), 1)
  b = fill!(similar(x, S, affine_sz), 0)
  return ∇batchnorm(g, b, x, dy, running_mean, running_var, momentum; kws...)
end

function ∇batchnorm(g::DenseCuArray, b::DenseCuArray, x::DenseCuArray{T,2},
                    dy::DenseCuArray{T,2},
            running_mean, running_var, momentum;
            kws...) where {T<:CUDNNFloat}
  dg, db, dx = ∇batchnorm(g, b, reshape(x, 1, 1, size(x, 1), size(x, 2)), reshape(dy, 1, 1, size(dy, 1),
                          size(dy, 2)), running_mean, running_var, momentum; kws...)
  (dg, db, dropdims(dx, dims = (1, 2)))
end


function ∇batchnorm(g::DenseCuArray, b::DenseCuArray, x::DenseCuArray{T},
                    dy::DenseCuArray{T},
                    running_mean, running_var, momentum;
                    affine=true, kws...) where {T<:CUDNNFloat}
  dg = similar(g)
  db = similar(b)
  dx = similar(x)
  cudnnBNBackward!(dg, g, db, dx, x, dy, running_mean, running_var, T(momentum); kws...)
  if affine
    (dg, db, dx)
  else
    (nothing, nothing, dx)
  end
end

function cudnnBNBackward!(dg::DenseCuArray, g::DenseCuArray, db::DenseCuArray,
                          dx::DenseCuArray{T}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                          running_mean, running_var,
                          momentum; cache = nothing, eps = T(1e-5),
                          alpha = T(1), beta = T(0),
                          dalpha = T(1), dbeta = T(0), training = true,
                          track_stats = true) where {T<:CUDNNFloat}
  if eps < CUDNN_BN_MIN_EPSILON
    @warn "eps $eps is too small for CuDNN, setting to CUDNN_BN_MIN_EPSILON=$CUDNN_BN_MIN_EPSILON"
    eps = CUDNN_BN_MIN_EPSILON
  end
  dims = batchnorm_param_size(x)
  S = batchnorm_stat_type(T)
  gS = batchnorm_param_array("scale", g, S, dims)
  dg_buffer = eltype(dg) == S ? dg : similar(dg, S)
  db_buffer = eltype(db) == S ? db : similar(db, S)
  dgS = batchnorm_param_array("dscale", dg_buffer, S, dims)
  dbS = batchnorm_param_array("dbias", db_buffer, S, dims)
  dbeta == 0 || begin
    dg_buffer === dg || (dgS .= reshape(dg, dims))
    db_buffer === db || (dbS .= reshape(db, dims))
  end
  if cache !== nothing
    mean, ivar = cache.mean, cache.ivar
  else
    _, mean, _, ivar, _ = batchnorm_statistics(x, S, eps)
  end

  use_graph = alpha == 1 && beta == 0 && dalpha == 1 && dbeta == 0
  if use_graph
    try
      batchnorm_gradient!(dx, dgS, dbS, dy, x, gS, mean, ivar; epsilon=eps)
    catch e
      graph_unsupported(e) || rethrow()
      batchnorm_generic_backward!(dgS, dbS, dx, x, dy, gS, mean, ivar;
                                  alpha, beta, dalpha, dbeta)
    end
  else
    batchnorm_generic_backward!(dgS, dbS, dx, x, dy, gS, mean, ivar;
                                alpha, beta, dalpha, dbeta)
  end
  dg_buffer === dg || (dg .= reshape(dgS, size(dg)))
  db_buffer === db || (db .= reshape(dbS, size(db)))
  return dg, db, dx
end
