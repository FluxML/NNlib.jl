module NNlibReactantExt

using NNlib: NNlib
using Reactant: Reactant, AnyTracedRArray, TracedRNumber
using Statistics: mean, var

# Reactant traces an array of `T` as an array of `TracedRNumber{T}`. The normalization
# functions treat half precision as they do for `T` itself: they require Float32
# parameters, accumulate in Float32 and cast the result back to the input eltype.
# Otherwise Float32 parameters would promote the result to Float32.
NNlib._stats_type(::Type{TracedRNumber{T}}) where {T} = TracedRNumber{NNlib._stats_type(T)}

# Enzyme differentiates `var(x; mean=μ)` through `μ` as well as `x`. That term is zero, as `μ`
# is the mean of `x`, but it costs a full-size multiply-and-reduce in the backward pass of
# every normalization. The variance about a detached copy `μ₀` of the mean, corrected by the
# shift `(μ - μ₀)²`, has the same value and derivatives without it, as in LuxLib.
function NNlib._mean_var(x::AnyTracedRArray, dims)
    μ = mean(x; dims)
    μ₀ = Reactant.Ops.ignore_derivatives(Reactant.materialize_traced_array(μ))
    return μ, var(x; mean=μ₀, dims, corrected=false) .- abs2.(μ .- μ₀)
end

# With the statistics of the batch, `batchnorm` lowers to `stablehlo.batch_norm_training`, from
# which XLA computes the mean and the mean square in one pass over `x`, and the gradient in one
# more. Normalizing with the running statistics, or with parameters of another eltype, takes
# the generic code.
function NNlib.batchnorm(g, b, x::AnyTracedRArray{<:Any,N}, running_mean, running_var, momentum;
                         eps=1f-5, training::Bool=true, track_stats::Bool=true, kws...) where {N}
    Tx = eltype(x)
    Tc = NNlib._stats_type(Tx)
    T = Reactant.unwrapped_eltype(Tc)
    if !(N > 1 && (training || running_mean === nothing) && _bn_param(g, T) && _bn_param(b, T))
        return invoke(NNlib.batchnorm, Tuple{Any, Any, AbstractArray, Any, Any, Any},
                      g, b, x, running_mean, running_var, momentum; eps, training, track_stats)
    end
    NNlib._check_affine(g, b)
    NNlib._check_stats(running_mean, running_var)
    NNlib._check_norm_types(Tx, g, b, running_mean, running_var)
    xc = Reactant.materialize_traced_array(Tc === Tx ? x : Tc.(x))
    y, μ, σ² = Reactant.Ops.batch_norm_training(xc, _bn_vec(g), _bn_vec(b);
                                                epsilon=T(eps), feature_index=N - 1)
    if track_stats && running_mean !== nothing
        NNlib._update_running_stats!(running_mean, running_var, Reactant.Ops.ignore_derivatives(μ),
                                     Reactant.Ops.ignore_derivatives(σ²), momentum,
                                     (ntuple(identity, N - 2)..., N), size(x))
    end
    return Tc === Tx ? y : Tx.(y)
end

_bn_param(::Nothing, T) = true
_bn_param(p, T) = p isa AnyTracedRArray && Reactant.unwrapped_eltype(p) === T

_bn_vec(::Nothing) = nothing
_bn_vec(p) = Reactant.materialize_traced_array(vec(p))

end
