# TODO: add CPU implementation
function batchnorm end

function ∇batchnorm end


function ChainRulesCore.rrule(::typeof(batchnorm), g, b, x, running_mean, running_var, momentum; kw...)
  y = batchnorm(g, b, x, running_mean, running_var, momentum; kw...) 
  function batchnorm_pullback(Δ)
    grad = ∇batchnorm(g, b, x, unthunk(Δ), running_mean, running_var, momentum; kw...)
    (NoTangent(), grad..., NoTangent(), NoTangent(), NoTangent())
  end
  y, batchnorm_pullback
end

"""
    norm_stats(x, dims)

Calculates sample mean and (uncorrected) variance of `x` along `dims`.

  - `dims=(1,...,N-2,N)` for batchnorm
  - `dims=(1,...,N-2)` for instancenorm and groupnorm
  - `dims=(1,...,S)` where S < N for layernorm

This is more efficient than calling `mean(x; dims)` and `var(x; dims)` separately,
because it can share some computation across both.
Implementors may want to overload this function to use custom kernels and more.
"""
function norm_stats(x, dims)
    μ = mean(x; dims)
    σ² = var(x; dims, mean = μ, corrected = false)
    return μ, σ²
end

function rrule(::typeof(norm_stats), x, dims)
    μ, mean_pullback = rrule(mean, x; dims)
    σ², var_pullback = rrule(var, x; dims, mean = μ, corrected = false)
    function norm_stats_pullback(dargs)
        dμ, dσ² = unthunk(dargs)
        dx = ChainRulesCore.add!!(var_pullback(dμ)[2], mean_pullback(dσ²)[2])
        return (NoTangent(), dx, NoTangent())
    end
    return (μ, σ²), norm_stats_pullback
end

_maybe_reshape(::Nothing, _) = nothing
_maybe_reshape(x, dims) = reshape(x, dims)
_apply_scale_bias(x, ::Nothing, ::Nothing) = x
_apply_scale_bias(x, scale, bias) = x .* scale .+ bias

"""
    norm_helper(x, μ, σ², scale::Union{AbstractArray, Nothing},
                bias::Union{AbstractArray, Nothing}, ϵ::Real, affine_size = size(μ))

Shared code path for all built-in norm functions.

`μ` and `σ²` should be calculated on the fly using [`norm_stats`](@ref),
or extracted from an existing collection such as [`RunningStats`](@ref).
`bias` and `scale` are consistent with cuDNN and Flux.Scale.
We opt for `scale` over `weight` to avoid confusion with dense layers.
If the size of the statistics and affine parameters differ,
use `affine_size` to add padding dimensions as required to match the input.
"""
function norm_helper(x, μ, σ², scale::Union{AbstractArray, Nothing},
                     bias::Union{AbstractArray, Nothing}, ϵ::Real, affine_size = size(μ))
    @ignore_derivatives if isnothing(scale) != isnothing(bias)
        error("both scale and bias must be provided or left as nothing")
    end
    scale′, bias′ = _maybe_reshape(scale, affine_size), _maybe_reshape(bias, affine_size)
    denom = inv.(sqrt.(σ² .+ ϵ))
    return _apply_scale_bias((x .- μ) .* denom, scale′, bias′)
end

"""
    RunningStats(mean, variance, momentum)

Contains running mean and variance estimates for stateful norm functions.
`momentum` controls the strength of the moving average update.

Parameters should be mutable and will be updated in-place.

See also [`update_running_stats!`](@ref).
"""
struct RunningStats{M <: AbstractArray, V <: AbstractArray, MT <: Real}
    mean::M
    variance::V
    momentum::MT
end

# Conditionally pulls running stats or calculates them on the fly.
# Part of the reason this is a dedicated function is to have a more type stable pullback.
function maybe_norm_stats(stats::Union{RunningStats, Nothing}, x, dims,
                          use_running_stats::Bool)
    if stats !== nothing && use_running_stats
        # Maintains consistency with mean/var
        sz = Base.setindex(Base.reduced_indices(x, dims) |> Base.to_shape, :, ndims(x) - 1)
        return reshape(stats.mean, sz), reshape(stats.variance, sz)
    end
    # No running stats exist or are disabled in inference mode
    return norm_stats(x, dims)
end

# Kludge so we can close over a Union inner pullback type
struct MaybeNormStatsPullback{B, P <: ProjectTo{AbstractArray}}
    back::B
    projector::P
end
function (pb::MaybeNormStatsPullback)(dargs)
    _, dx = unthunk(pb.back(dargs))
    return (NoTangent(), NoTangent(), pb.projector(dx), NoTangent(), NoTangent())
end
function rrule(::typeof(maybe_norm_stats), stats::Union{RunningStats, Nothing}, x, dims,
               use_running_stats::Bool)
    project = ProjectTo(x)
    noop_back(_) = (NoTangent(), NoTangent())
    if stats === nothing || !use_running_stats
        (μ, σ²), back = rrule(norm_stats, x, dims)
    else
        # The default is to track, so this only happens when a layer is frozen
        sz = Base.setindex(Base.reduced_indices(x, dims) |> Base.to_shape, :, ndims(x) - 1)
        μ, σ², back = reshape(stats.mean, sz), reshape(stats.variance, sz), noop_back
    end
    back_type = Union{typeof(noop_back), _rrule_pullback_rt(norm_stats, x, dims)}
    return (μ, σ²), MaybeNormStatsPullback{back_type, typeof(project)}(back, project)
end

"""
    update_running_stats!(stats::RunningStats, x::AbstractArray{<:Any, N}, μ, σ²,
                          reduce_dims) where {N}

Performs a moving average update for layers with tracked statistics.
`μ` and `σ²` are the sample mean and variance, most likely from [`norm_stats`](@ref).
`reduce_dims` should also match the `dims` argument of [`norm_stats`](@ref).

See also [`RunningStats`](@ref).
"""
function update_running_stats!(stats::RunningStats, x, μ, σ², reduce_dims::Dims)
    V = eltype(σ²)
    momentum = stats.momentum
    res_mtm = one(V) - momentum
    m = prod(size(x, i) for i in reduce_dims; init = 1)
    correction = m / (m - one(V))

    running_mean, running_var = stats.mean, stats.variance
    stats.mean .= res_mtm .* running_mean .+ momentum .* vec(μ)
    stats.variance .= res_mtm .* running_var .+ momentum .* correction .* vec(σ²)
    return
end

# Convenience functions
# We follow roughly the same arg order as torch.nn.functional.*_norm:
# input, unique args for this particular norm type, bias + scale, eps; kwargs...

"""
    layernorm(x::AbstractArray{<:Any,N}, ::Val{S}, scale = nothing, bias = nothing,
              ϵ=ofeltype(x, 1e-5)) where {N, S}

Functional [Layer Normalization](https://arxiv.org/abs/1607.06450) operation.

Normalizes `x` along the first `S` dimensions.

For an additional learned affine transform, provide a `S`-dimensional `scale` and `bias`.

See also [`batchnorm`](@ref), [`instancenorm`](@ref), and [`groupnorm`](@ref).

# Examples

```jldoctest
julia> using Statistics

julia> xs = rand(3, 3, 3, 2);  # a batch of 2 images, each having 3 channels

julia> y = NNlib.layernorm(xs, Val(3));

julia> isapprox(std(y; dims = 1:3), ones(1, 1, 1, 2); atol = 0.1) &&
           std(y; dims = 1:3) != std(xs; dims = 1:3)
true
```
"""
function layernorm(x::AbstractArray{<:Any, N}, ::Val{S}, scale = nothing, bias = nothing,
                   ϵ = ofeltype(x, 1e-5)) where {N, S}
    @ignore_derivatives if S > N
        throw(DimensionMismatch("got $S reduction dims for $N-dimensional array"))
    end
    μ, σ² = norm_stats(x, ntuple(identity, S))
    return norm_helper(x, μ, σ², scale, bias, ϵ, size(x)[1:S]::Dims{S})
end

"""
    batchnorm(x::AbstractArray{<:Any, N},
              running_stats::Union{RunningStats, Nothing} = nothing,
              scale::Union{AbstractVector, Nothing} = nothing,
              bias::Union{AbstractVector, Nothing} = nothing, ϵ = ofeltype(x, 1e-5);
              training::Bool = within_grad()) where {N}

Functional [Batch Normalization](https://arxiv.org/abs/1502.03167) operation.

Normalizes `x` along each ``D_1×...×D_{N-2}×1×D_N`` input slice,
where `N-1` is the "channel" (or "feature", for 2D inputs) dimension.

Provide a [`RunningStats`](@ref) to fix a estimated mean and variance.
`batchnorm` will renormalize the input using these statistics during inference,
and update them using batch-level statistics when training.
To override this behaviour, manually set a value for `training`.

If specified, `scale` and `bias` will be applied as an additional learned affine transform.

See also [`layernorm`](@ref), [`instancenorm`](@ref), and [`groupnorm`](@ref).
"""
function batchnorm(x::AbstractArray{<:Any, N},
                   running_stats::Union{RunningStats, Nothing} = nothing,
                   scale::Union{AbstractVector, Nothing} = nothing,
                   bias::Union{AbstractVector, Nothing} = nothing, ϵ = ofeltype(x, 1e-5);
                   training::Bool = within_grad()) where {N}
    reduce_dims = ((1:(N - 2))..., N)
    μ, σ² = maybe_norm_stats(running_stats, x, reduce_dims, !training)
    # Because μ and σ² could be updated in-place, we compute the output first
    y = norm_helper(x, μ, σ², scale, bias, ϵ)
    @ignore_derivatives if running_stats !== nothing && training
        update_running_stats!(running_stats, x, μ, σ², reduce_dims)
    end
    return y
end

"""
    instancenorm(x::AbstractArray{<:Any, N},
                 running_stats::Union{RunningStats, Nothing} = nothing,
                 scale::Union{AbstractVector, Nothing} = nothing,
                 bias::Union{AbstractVector, Nothing} = nothing, ϵ = ofeltype(x, 1e-5);
                 training::Bool = within_grad()) where {N}

Functional [Instance Normalization](https://arxiv.org/abs/1607.08022) operation.

Normalizes `x` along each ``D_1×...×D_{N-2}×1×1`` input slice,

Provide a [`RunningStats`](@ref) to fix a estimated mean and variance.
`instancenorm` will renormalize the input using these statistics during inference,
and update them using batch-level statistics when training.
To override this behaviour, manually set a value for `training`.

If specified, `scale` and `bias` will be applied as an additional learned affine transform.

See also [`layernorm`](@ref), [`batchnorm`](@ref), and [`groupnorm`](@ref).
"""
function instancenorm(x::AbstractArray{<:Any, N},
                      running_stats::Union{RunningStats, Nothing} = nothing,
                      scale::Union{AbstractVector, Nothing} = nothing,
                      bias::Union{AbstractVector, Nothing} = nothing, ϵ = ofeltype(x, 1e-5);
                      training::Bool = within_grad()) where {N}
    affine_size = (ntuple(_ -> 1, N - 2)..., size(x, N - 1), :)
    reduce_dims = ((1:(N - 2))...,)
    μ, σ² = maybe_norm_stats(running_stats, x, reduce_dims, !training)
    # Because μ and σ² could be updated in-place, we compute the output first
    y = norm_helper(x, μ, σ², scale, bias, ϵ, affine_size)
    ChainRulesCore.@ignore_derivatives if running_stats !== nothing && training
        μ′, σ²′ = mean(μ; dims = N), mean(σ²; dims = N) # Need to sum (C, N) -> (C,)
        update_running_stats!(running_stats, x, μ′, σ²′, reduce_dims)
    end
    return y
end

"""
    groupnorm(x::AbstractArray{<:Any, N}, groups::Integer,
              scale::Union{AbstractVector, Nothing} = nothing,
              bias::Union{AbstractVector, Nothing} = nothing,
              ϵ = ofeltype(x, 1e-5)) where {N}

Functional [Group Normalization](https://arxiv.org/abs/1803.08494) operation.

Normalizes `x` along the first `N - 2` (spatial) dimensions,
where `N-1` is the "channel" (or "feature", for 2D inputs) dimension,
and the channel dimension is divided into `groups` groups along which statistics are computed.
The number of channels must be an integer multiple of the number of groups.

If specified, `scale` and `bias` will be applied as an additional learned affine transform.

See also [`layernorm`](@ref), [`batchnorm`](@ref), and [`instancenorm`](@ref).

# Examples

```jldoctest
julia> using Statistics

julia> xs = rand(3, 3, 4, 2);  # a batch of 2 images, each having 4 channels

julia> y = NNlib.groupnorm(xs, 4);

julia> isapprox(std(y[:, :, 1:2, 1]), 1; atol = 0.1) &&
           std(xs[:, :, 1:2, 1]) != std(y[:, :, 1:2, 1])
true

julia> isapprox(std(y[:, :, 3:4, 2]), 1; atol = 0.1) &&
           std(xs[:, :, 3:4, 2]) != std(y[:, :, 3:4, 2])
true
```
"""
function groupnorm(x::AbstractArray{<:Any, N}, groups::Integer,
                   scale::Union{AbstractVector, Nothing} = nothing,
                   bias::Union{AbstractVector, Nothing} = nothing,
                   ϵ = ofeltype(x, 1e-5)) where {N}
    sz = size(x)
    channels = @ignore_derivatives begin
        ch = sz[max(1, N - 1)]
        newch, remainder = divrem(ch, groups)
        remainder == 0 ? newch :
        throw(ArgumentError("channels $ch should be multiple of groups $groups"))
    end
    affine_size = (ntuple(_ -> 1, N - 2)..., channels, groups, :)
    grouped_size = (sz[1:(N - 2)]..., channels, groups, :)
    x′ = reshape(x, grouped_size)
    μ, σ² = norm_stats(x′, ((1:(N - 2))...,))
    return reshape(norm_helper(x′, μ, σ², scale, bias, ϵ, affine_size), sz)
end
