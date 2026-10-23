# Functional normalization operators.
#
# These are device-agnostic, differentiable (through the generic AD path — no
# custom rrules are needed since they are built from `mean`/`var`/broadcast) and
# mirror the maths of the corresponding Flux layers. Statistics-tracking layers
# (`Flux.BatchNorm`, `Flux.InstanceNorm`, ...) can delegate their forward pass to
# `batchnorm`/`instancenorm`/`groupnorm`/`layernorm` here.
#
# All four share the same argument layout as the cuDNN `batchnorm` fast path: the
# scale `g` and bias `b` (either may be `nothing` for no affine transform) come
# first, then the feature maps `x`. For `WHCN`-style data the `N-1`th dimension is
# the channel dimension. On the GPU, `batchnorm` on 3D/4D/5D `CuArray`s is handled
# by the cuDNN methods in `NNlibCUDACUDNNExt`. Otherwise, all of them run the fused
# kernels of `normalization_fused.jl` on dense CPU and GPU arrays (`batchnorm` only with
# the statistics of the batch). Every other case uses the generic code below.
#
# `rmsnorm` has a scale but no bias, so its layout is `(g, x)`.

# `eps` converted to the (real) float eltype of `x`, avoiding Float64 promotion of
# Float16/Float32 data.
_epsof(x::AbstractArray, eps) = convert(float(real(eltype(x))), eps)

# Half-precision (Float16/BFloat16) feature maps are normalised in Float32: the
# reductions and running-statistic updates accumulate in `_stats_type`, and the
# result is cast back to the input eltype. This mirrors the cuDNN contract, which
# requires Float32 affine parameters and running statistics for half-precision data.
_stats_type(::Type{T}) where {T} = T
_stats_type(::Type{Float16}) = Float32
_stats_type(::Type{BFloat16}) = Float32

# Enforce the Float32-parameter contract for half-precision inputs so a mismatch
# surfaces as a clear error rather than a silent precision loss.
function _check_norm_types(::Type{T}, g, b, running_mean, running_var) where {T}
    _stats_type(T) === T && return nothing  # full-precision: parameters may promote freely
    for (name, a) in (("scale g", g), ("bias b", b),
                      ("running_mean", running_mean), ("running_var", running_var))
        a === nothing && continue
        eltype(a) === Float32 || throw(ArgumentError(
            "$T normalization requires Float32 $name, got $(eltype(a)); " *
            "half-precision statistics are numerically unstable."))
    end
    return nothing
end

# The scale `g` and bias `b` must be given together or not at all (matching the
# cuDNN `batchnorm` methods, which only accept both arrays or both `nothing`).
_check_affine(g, b) = (g === nothing) == (b === nothing) ? nothing : throw(ArgumentError(
    "both or neither of the scale `g` and bias `b` must be `nothing`"))
_check_stats(rm, rv) = (rm === nothing) == (rv === nothing) ? nothing : throw(ArgumentError(
    "both or neither of running_mean and running_var must be `nothing`"))

# Strip `ForwardDiff.Dual`s when writing back into running statistics (see #2122 in
# Flux). Extended in `NNlibForwardDiffExt`; identity everywhere else.
_value(x) = x

"""
    normalise(x; dims=ndims(x), eps=1f-5)

Normalise `x` to zero mean and unit standard deviation across the dimension(s)
given by `dims`. Per default, `dims` is the last dimension. `eps` is a small term
added to the variance for numerical stability.

It is [`layernorm`](@ref) without scale and bias. For half-precision (`Float16`/`BFloat16`)
inputs the statistics are computed in `Float32`.

# Examples
```jldoctest
julia> using Statistics

julia> x = [90, 100, 110, 130, 70];

julia> y = NNlib.normalise(x);

julia> isapprox(std(y; corrected=false), 1, atol=1e-5)
true
```
"""
normalise(x::AbstractArray; dims=ndims(x), eps=1f-5) = layernorm(nothing, nothing, x; dims, eps)

# Core affine-normalize transform: `y = g * (x - μ)/√(σ² + ϵ) + b`.
# `μ`, `σ²`, `g`, `b` are already reshaped to broadcast against `x`; `g`/`b` may be
# `nothing` (no affine transform).
function _affine_normalize(x, μ, σ², g, b, ϵ)
    denom = sqrt.(σ² .+ ϵ)
    if g === nothing && b === nothing
        return (x .- μ) ./ denom
    else
        g = g === nothing ? one(eltype(x)) : g
        b = b === nothing ? zero(eltype(x)) : b
        return @. g / denom * (x - μ) + b
    end
end

# In-place moving-average update of the running statistics, mirroring
# `Flux._track_stats!`. `reduce_dims` are the dimensions the batch statistics were
# computed over; the batch dimension `N` is averaged out when it is not among them
# (InstanceNorm). Non-differentiable: never traced by reverse-mode AD (Mooncake does not
# read ChainRules declarations, so `NNlibMooncakeExt` declares it separately).
function _update_running_stats!(running_mean, running_var, μ, σ², momentum, reduce_dims, sz)
    V = eltype(running_var)
    mtm = V(momentum)
    res_mtm = one(V) - mtm
    N = length(sz)
    m = prod(i -> sz[i], reduce_dims)
    μnew = vec(N in reduce_dims ? μ : mean(μ; dims=N))
    σ²new = vec(N in reduce_dims ? σ² : mean(σ²; dims=N))
    running_mean .= res_mtm .* running_mean .+ mtm .* _value.(μnew)
    running_var  .= res_mtm .* running_var  .+ mtm .* (m / (m - one(V))) .* _value.(σ²new)
    return nothing
end

ChainRulesCore.@non_differentiable _update_running_stats!(::Any...)

# Shared engine for `batchnorm` and `instancenorm`: per-channel statistics/affine
# over `reduce_dims`, with optional running-statistics tracking. The channel
# dimension is `N-1`.
function _norm_layer(g, b, x::AbstractArray{T,N}, running_mean, running_var,
                     reduce_dims; training, momentum, eps, track_stats) where {T,N}
    _check_affine(g, b)
    _check_stats(running_mean, running_var)
    _check_norm_types(T, g, b, running_mean, running_var)
    Tc = _stats_type(T)
    xc = Tc === T ? x : Tc.(x)  # half-precision: accumulate statistics in Float32
    ϵ = _epsof(xc, eps)
    affine_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
    if !training && running_mean !== nothing
        μ = reshape(running_mean, affine_shape)
        σ² = reshape(running_var, affine_shape)
    else
        μ = mean(xc; dims=reduce_dims)
        σ² = var(xc; mean=μ, dims=reduce_dims, corrected=false)
        if track_stats && running_mean !== nothing
            _update_running_stats!(running_mean, running_var, μ, σ², momentum, reduce_dims, size(x))
        end
    end
    g = g === nothing ? nothing : reshape(g, affine_shape)
    b = b === nothing ? nothing : reshape(b, affine_shape)
    y = _affine_normalize(xc, μ, σ², g, b, ϵ)
    return Tc === T ? y : T.(y)
end

"""
    batchnorm(g, b, x, running_mean=nothing, running_var=nothing, momentum=0.1f0;
              eps=1f-5, training=true, track_stats=true)

Functional [batch normalization](https://arxiv.org/abs/1502.03167). `g` and `b`
are the per-channel scale and bias (either may be `nothing` for no affine
transform); `x` are the feature maps. For an input with `N` dimensions the `N-1`th
is the channel dimension (the usual convention for `WHCN` images); statistics are
computed over every `D_1×…×D_{N-2}×1×D_N` slice, so per channel.

If `running_mean`/`running_var` are supplied:
- with `training=true` (default) they are updated **in place** (when
  `track_stats=true`) with an exponential moving average controlled by `momentum`,
  while statistics of the current batch are used to normalise;
- with `training=false` they are used to normalise (inference / test mode).

`eps` is added to the variance for numerical stability. On the GPU, 3D/4D/5D
`CuArray`s are dispatched to the cuDNN implementation.

For half-precision (`Float16`/`BFloat16`) feature maps the statistics are computed
in `Float32` and the result is cast back; `g`, `b`, `running_mean` and
`running_var` must then be `Float32`.

See also [`instancenorm`](@ref), [`groupnorm`](@ref), [`layernorm`](@ref).
"""
function batchnorm(g, b, x::AbstractArray{T,N},
                   running_mean=nothing, running_var=nothing, momentum=0.1f0;
                   eps=1f-5, training::Bool=true, track_stats::Bool=true, kws...) where {T,N}
    # `kws` drops the keywords of the cuDNN methods (`cache`, `alpha`, `beta`), which
    # callers such as Flux ≤ 0.16.10 pass for any `CuArray`.
    N > 1 || throw(ArgumentError("batchnorm expects an array with at least 2 dimensions, got $N"))
    use_fused_batchnorm(g, b, x, running_mean, training) &&
        return fused_batchnorm(g, b, x, running_mean, running_var, momentum, eps, track_stats)[1]
    reduce_dims = (ntuple(identity, N-2)..., N)
    return _norm_layer(g, b, x, running_mean, running_var, reduce_dims;
                       training, momentum, eps, track_stats)
end

# `batchnorm` with the statistics of `x` runs the fused kernels.
use_fused_batchnorm(g, b, x, running_mean, training) =
    (training || running_mean === nothing) && !within_gradient(x) && fused_batchnorm_applies(x, g, b)

# `batchnorm` with the fused kernels, and its statistics for the gradient.
function fused_batchnorm(g, b, x::AbstractArray{T,N}, running_mean, running_var,
                         momentum, eps, track_stats) where {T,N}
    _check_affine(g, b)
    _check_stats(running_mean, running_var)
    _check_norm_types(T, g, b, running_mean, running_var)
    μ, rσ, coef = fused_batchnorm_stats(g, b, x, eps)
    if track_stats && running_mean !== nothing
        _update_running_stats_fused!(running_mean, running_var, μ, rσ, momentum, eps,
                                     (ntuple(identity, N-2)..., N), size(x))
    end
    return batchnorm_map!(batchnorm_y, similar(x), coef, x), μ, rσ
end

# The running statistics updated from the mean and reciprocal deviation of the kernels, with
# an entry per channel, or per channel and sample if the batch is not among `reduce_dims`.
function _update_running_stats_fused!(running_mean, running_var, μ, rσ, momentum, eps, reduce_dims, sz)
    N = length(sz)
    shape = ntuple(i -> i == N-1 || (i == N && N ∉ reduce_dims) ? sz[i] : 1, N)
    σ² = inv.(rσ) .^ 2 .- _epsof(rσ, eps)
    _update_running_stats!(running_mean, running_var, reshape(μ, shape), reshape(σ², shape),
                           momentum, reduce_dims, sz)
end

"""
    instancenorm(g, b, x, running_mean=nothing, running_var=nothing, momentum=0.1f0;
                 eps=1f-5, training=true, track_stats=false)

Functional [instance normalization](https://arxiv.org/abs/1607.08022). Arguments
match [`batchnorm`](@ref), but for an input with `N > 2` dimensions statistics are
computed over every `D_1×…×D_{N-2}×1×1` slice, so per channel **and** per sample in
the batch. When tracked, the running statistics (length `size(x, N-1)`) accumulate
the per-channel average across the batch.

Normalising a dense CPU or GPU array with its own statistics, with `g` and `b` of length
`size(x, N-1)`, runs fused kernels; other cases use the generic code.

See also [`batchnorm`](@ref), [`groupnorm`](@ref), [`layernorm`](@ref).
"""
function instancenorm(g, b, x::AbstractArray{T,N},
                      running_mean=nothing, running_var=nothing, momentum=0.1f0;
                      eps=1f-5, training::Bool=true, track_stats::Bool=false) where {T,N}
    N > 2 || throw(ArgumentError("instancenorm expects an array with at least 3 dimensions, got $N"))
    use_fused_instancenorm(g, b, x, running_mean, training) &&
        return fused_instancenorm(g, b, x, running_mean, running_var, momentum, eps, track_stats)[1]
    reduce_dims = ntuple(identity, N-2)
    return _norm_layer(g, b, x, running_mean, running_var, reduce_dims;
                       training, momentum, eps, track_stats)
end

# `instancenorm` with the statistics of `x` is `groupnorm` with a group per channel.
use_fused_instancenorm(g, b, x, running_mean, training) = ndims(x) > 2 &&
    (training || running_mean === nothing) && use_fused_norm(x, NormGroups(size(x, ndims(x) - 1)), g, b)

# `instancenorm` with the fused kernels, and its statistics for the gradient.
function fused_instancenorm(g, b, x::AbstractArray{T,N}, running_mean, running_var,
                            momentum, eps, track_stats) where {T,N}
    _check_affine(g, b)
    _check_stats(running_mean, running_var)
    _check_norm_types(T, g, b, running_mean, running_var)
    y, μ, rσ = fused_norm(g, b, x, NormGroups(size(x, N-1)), eps, true)
    if track_stats && running_mean !== nothing
        _update_running_stats_fused!(running_mean, running_var, μ, rσ, momentum, eps,
                                     ntuple(identity, N-2), size(x))
    end
    return y, μ, rσ
end

"""
    groupnorm(g, b, x, G::Integer; eps=1f-5)

Functional [group normalization](https://arxiv.org/abs/1803.08494). `g` and `b`
are the per-channel scale and bias (either may be `nothing`); `x` are the feature
maps. For an input with `N > 2` dimensions the `N-1`th is the channel dimension;
its `C = size(x, N-1)` channels are split into `G` groups (`G` must divide `C`) and
statistics are computed over each group together with the spatial dimensions, per
sample. `eps` is added to the variance.

Normalising a dense CPU or GPU array, with `g` and `b` of length `C`, runs fused kernels;
other cases use the generic code.

See also [`batchnorm`](@ref), [`instancenorm`](@ref), [`layernorm`](@ref).
"""
function groupnorm(g, b, x::AbstractArray{T,N}, G::Integer; eps=1f-5) where {T,N}
    _check_groupnorm(g, b, x, G)
    use_fused_norm(x, NormGroups(G), g, b) && return fused_norm(g, b, x, NormGroups(G), eps, true)[1]
    C = size(x, N-1)
    Tc = _stats_type(T)
    sz = size(x)
    # Split the channel dimension into (C÷G) × G, giving an (N+1)-dim array.
    x2 = reshape(Tc === T ? x : Tc.(x), sz[1:N-2]..., C ÷ G, G, sz[N])
    reduce_dims = ntuple(identity, N-1)  # spatial dims + the intra-group channel dim
    μ = mean(x2; dims=reduce_dims)
    σ² = var(x2; mean=μ, dims=reduce_dims, corrected=false)
    ϵ = _epsof(x2, eps)
    affine_shape = (ntuple(_ -> 1, N-2)..., C ÷ G, G, 1)
    g2 = g === nothing ? nothing : reshape(g, affine_shape)
    b2 = b === nothing ? nothing : reshape(b, affine_shape)
    y = _affine_normalize(x2, μ, σ², g2, b2, ϵ)
    y = reshape(y, sz)
    return Tc === T ? y : T.(y)
end

function _check_groupnorm(g, b, x::AbstractArray{T,N}, G) where {T,N}
    N > 2 || throw(ArgumentError("groupnorm expects an array with at least 3 dimensions, got $N"))
    C = size(x, N-1)
    C % G == 0 || throw(ArgumentError("the number of groups G=$G must divide the number of channels C=$C"))
    _check_affine(g, b)
    _check_norm_types(T, g, b, nothing, nothing)
end

"""
    layernorm(g, b, x; dims=1, eps=1f-5)

Functional [layer normalization](https://arxiv.org/abs/1607.06450). `g` and `b`
are the scale and bias (either may be `nothing`); `x` is normalised over the
dimensions `dims` (the leading dimension by default).

`g` and `b`, when given, must broadcast against the normalised region, e.g. have
size `size(x)[dims]` with singleton trailing dimensions.

Normalising a dense CPU or GPU array over leading `dims` (`1:k`), with `g` and `b` of
the shape of those dimensions, runs fused kernels; other cases use the generic code.

See also [`normalise`](@ref), [`batchnorm`](@ref), [`instancenorm`](@ref),
[`groupnorm`](@ref).
"""
function layernorm(g, b, x::AbstractArray{T}; dims=1, eps=1f-5) where {T}
    _check_affine(g, b)
    _check_norm_types(T, g, b, nothing, nothing)
    use_fused_norm(x, dims, g, b) && return fused_norm(g, b, x, dims, eps, true)[1]
    Tc = _stats_type(T)
    xc = Tc === T ? x : Tc.(x)
    μ = mean(xc; dims)
    σ² = var(xc; dims, mean=μ, corrected=false)
    y = _affine_normalize(xc, μ, σ², g, b, _epsof(xc, eps))
    return Tc === T ? y : T.(y)
end

"""
    rmsnorm(g, x; dims=1, eps=1f-5)

Functional [root mean square normalization](https://arxiv.org/abs/1910.07467).
`x` is divided by its root mean square over the dimensions `dims` (the leading
dimension by default) and multiplied by the scale `g` (`nothing` for no scaling):

    y = g .* x ./ sqrt.(mean(abs2, x; dims) .+ eps)

Unlike [`layernorm`](@ref), the mean is not subtracted and there is no bias.

`g`, when given, must broadcast against the normalised region, e.g. have size
`size(x)[dims]` with singleton trailing dimensions. `eps` is added to the mean
square for numerical stability.

For half-precision (`Float16`/`BFloat16`) inputs the mean square is computed in
`Float32` and the result is cast back; `g` must then be `Float32`.

See also [`layernorm`](@ref), [`normalise`](@ref).

# Examples
```jldoctest
julia> x = Float32[3 0; 4 2];

julia> NNlib.rmsnorm(nothing, x) ≈ x ./ sqrt.(Float32[12.5 2])
true

julia> NNlib.rmsnorm(Float32[2, 1], x) ≈ Float32[2, 1] .* NNlib.rmsnorm(nothing, x)
true
```
"""
function rmsnorm(g, x::AbstractArray{T}; dims=1, eps=1f-5) where {T}
    _check_norm_types(T, g, nothing, nothing, nothing)
    use_fused_norm(x, dims, g) && return fused_norm(g, nothing, x, dims, eps, false)[1]
    Tc = _stats_type(T)
    xc = Tc === T ? x : Tc.(x)
    rms = sqrt.(mean(abs2, xc; dims) .+ _epsof(xc, eps))
    y = g === nothing ? xc ./ rms : @. g / rms * xc
    return Tc === T ? y : T.(y)
end

# --- Gradient operators (VJPs) -----------------------------------------------
# Each `∇op(g, b, x, dy, ...)` returns `(dg, db, dx)`, the vector-Jacobian product
# of `op` at `x` contracted with `dy` (`dg`/`db` are `nothing` when the matching
# parameter is). They are the reverse rules used by the `rrule`s below, and are
# also useful directly. Crucially they are built from `mean`/`var`/broadcast, so
# they are themselves differentiable — differentiating a gradient gives correct
# second-order derivatives (e.g. Hessian-vector products).

# Reduce a broadcast result `Δ` down to the shape of parameter `p` (summing the
# dimensions `p` was broadcast over), for parameter gradients.
function _unbroadcast(Δ::AbstractArray, p::AbstractArray)
    size(Δ) == size(p) && return Δ
    rdims = ntuple(d -> (d > ndims(p) || size(p, d) == 1) ? d : 0, ndims(Δ))
    return reshape(sum(Δ; dims=filter(!=(0), rdims)), size(p))
end

# Shared VJP for the channel-wise operators (batchnorm/instancenorm). `g`/`b` are
# per-channel vectors, statistics run over `reduce_dims`, and parameter gradients
# reduce over every dimension except the channel `N-1`.
function _∇norm_channel(g, b, x::AbstractArray{T,N}, dy, running_mean, running_var,
                        reduce_dims; eps, training) where {T,N}
    _check_affine(g, b)
    Tc = _stats_type(T)                        # half precision: work in Float32
    xc = Tc === T ? x : Tc.(x)
    dyc = Tc === T ? dy : Tc.(dy)
    ϵ = _epsof(xc, eps)
    affine_shape = ntuple(i -> i == N-1 ? size(x, N-1) : 1, N)
    stat_from_x = training || running_mean === nothing
    if stat_from_x
        μ = mean(xc; dims=reduce_dims)
        σ² = var(xc; mean=μ, dims=reduce_dims, corrected=false)
    else
        μ = reshape(running_mean, affine_shape)
        σ² = reshape(running_var, affine_shape)
    end
    σ = sqrt.(σ² .+ ϵ)
    x̂ = (xc .- μ) ./ σ
    # Reduce over `reduce_dims` once. The per-channel scale `g` is constant over
    # `reduce_dims`, so the input gradient factors as
    # `dx = g/σ ⋅ (dy - mean_R(dy) - x̂ ⋅ mean_R(dy⋅x̂))`, letting a single `dy⋅x̂`
    # product serve both `dg` and the `x̂`-correction term (rather than materialising
    # `dy⋅x̂`, `dy⋅g` and `dy⋅g⋅x̂` separately). The parameter gradients further reduce
    # the batch dim `N` for InstanceNorm (`N ∉ reduce_dims`); for BatchNorm it is
    # already among `reduce_dims`, so `sum_dy`/`sum_p` are the parameter gradients.
    sum_dy = sum(dyc;      dims=reduce_dims)
    sum_p  = sum(dyc .* x̂; dims=reduce_dims)
    foldbatch(s) = N in reduce_dims ? s : sum(s; dims=N)
    dg = g === nothing ? nothing : reshape(foldbatch(sum_p),  size(g))
    db = b === nothing ? nothing : reshape(foldbatch(sum_dy), size(b))
    if stat_from_x
        m = prod(i -> size(x, i), reduce_dims)
        s_dy = sum_dy ./ m
        s_p  = sum_p  ./ m
        if g === nothing
            dx = @. (dyc - s_dy - x̂ * s_p) / σ
        else
            gc = reshape(g, affine_shape)
            dx = @. gc * (dyc - s_dy - x̂ * s_p) / σ
        end
    else
        dx = g === nothing ? dyc ./ σ : reshape(g, affine_shape) .* dyc ./ σ
    end
    return dg, db, Tc === T ? dx : T.(dx)
end

"""
    ∇batchnorm(g, b, x, dy, running_mean=nothing, running_var=nothing, momentum=0.1f0;
               eps=1f-5, training=true)

Gradient of [`batchnorm`](@ref): given the upstream gradient `dy`, return
`(dg, db, dx)`. On the GPU, 3D/4D/5D `CuArray`s are dispatched to the cuDNN
implementation in `NNlibCUDACUDNNExt`; this generic method is the fallback used
everywhere else and is itself differentiable (second-order).
"""
function ∇batchnorm(g, b, x::AbstractArray{T,N}, dy,
                    running_mean=nothing, running_var=nothing, momentum=0.1f0;
                    eps=1f-5, training::Bool=true, kws...) where {T,N}
    _check_affine(g, b)
    if use_fused_batchnorm(g, b, x, running_mean, training)
        μ, rσ, _ = fused_batchnorm_stats(nothing, nothing, x, eps)
        return ∇fused_batchnorm(g, b, x, dy, μ, rσ)
    end
    reduce_dims = (ntuple(identity, N-2)..., N)
    return _∇norm_channel(g, b, x, dy, running_mean, running_var, reduce_dims; eps, training)
end

"""
    ∇instancenorm(g, b, x, dy, running_mean=nothing, running_var=nothing, momentum=0.1f0;
                  eps=1f-5, training=true)

Gradient of [`instancenorm`](@ref), returning `(dg, db, dx)`.
"""
function ∇instancenorm(g, b, x::AbstractArray{T,N}, dy,
                       running_mean=nothing, running_var=nothing, momentum=0.1f0;
                       eps=1f-5, training::Bool=true, kws...) where {T,N}
    _check_affine(g, b)
    if use_fused_instancenorm(g, b, x, running_mean, training)
        groups = NormGroups(size(x, N-1))
        _, μ, rσ = fused_norm(g, b, x, groups, eps, true; output=false)
        return ∇fused_norm(g, b, x, dy, μ, rσ, groups)
    end
    reduce_dims = ntuple(identity, N-2)
    return _∇norm_channel(g, b, x, dy, running_mean, running_var, reduce_dims; eps, training)
end

"""
    ∇groupnorm(g, b, x, dy, G::Integer; eps=1f-5)

Gradient of [`groupnorm`](@ref), returning `(dg, db, dx)`.
"""
function ∇groupnorm(g, b, x::AbstractArray{T,N}, dy, G::Integer; eps=1f-5) where {T,N}
    _check_affine(g, b)
    if use_fused_norm(x, NormGroups(G), g, b)
        _, μ, rσ = fused_norm(g, b, x, NormGroups(G), eps, true; output=false)
        return ∇fused_norm(g, b, x, dy, μ, rσ, NormGroups(G))
    end
    Tc = _stats_type(T)
    C = size(x, N-1); sz = size(x)
    x2 = reshape(Tc === T ? x : Tc.(x), sz[1:N-2]..., C ÷ G, G, sz[N])
    dy2 = reshape(Tc === T ? dy : Tc.(dy), size(x2))
    ϵ = _epsof(x2, eps)
    M = N + 1
    reduce_dims = ntuple(identity, M-2)             # spatial + intra-group channel
    param_dims = (ntuple(identity, N-2)..., M)      # every dimension but (C÷G, G)
    affine_shape = (ntuple(_ -> 1, N-2)..., C ÷ G, G, 1)
    μ = mean(x2; dims=reduce_dims)
    σ² = var(x2; mean=μ, dims=reduce_dims, corrected=false)
    σ = sqrt.(σ² .+ ϵ)
    x̂ = (x2 .- μ) ./ σ
    dg = g === nothing ? nothing : reshape(sum(dy2 .* x̂; dims=param_dims), size(g))
    db = b === nothing ? nothing : reshape(sum(dy2; dims=param_dims), size(b))
    dx̂ = g === nothing ? dy2 : dy2 .* reshape(g, affine_shape)
    dx2 = (dx̂ .- mean(dx̂; dims=reduce_dims) .- x̂ .* mean(dx̂ .* x̂; dims=reduce_dims)) ./ σ
    dx = reshape(dx2, sz)
    return dg, db, Tc === T ? dx : T.(dx)
end

"""
    ∇layernorm(g, b, x, dy; dims=1, eps=1f-5)

Gradient of [`layernorm`](@ref), returning `(dg, db, dx)`.
"""
function ∇layernorm(g, b, x::AbstractArray{T,N}, dy; dims=1, eps=1f-5) where {T,N}
    _check_affine(g, b)
    if use_fused_norm(x, dims, g, b)
        _, μ, rσ = fused_norm(g, b, x, dims, eps, true; output=false)
        return ∇fused_norm(g, b, x, dy, μ, rσ, dims)
    end
    Tc = _stats_type(T)
    xc = Tc === T ? x : Tc.(x)
    dyc = Tc === T ? dy : Tc.(dy)
    ϵ = _epsof(xc, eps)
    μ = mean(xc; dims)
    σ² = var(xc; mean=μ, dims, corrected=false)
    σ = sqrt.(σ² .+ ϵ)
    x̂ = (xc .- μ) ./ σ
    dg = g === nothing ? nothing : _unbroadcast(dyc .* x̂, g)
    db = b === nothing ? nothing : _unbroadcast(dyc, b)
    dx̂ = g === nothing ? dyc : dyc .* g
    dx = (dx̂ .- mean(dx̂; dims) .- x̂ .* mean(dx̂ .* x̂; dims)) ./ σ
    return dg, db, Tc === T ? dx : T.(dx)
end

"""
    ∇rmsnorm(g, x, dy; dims=1, eps=1f-5)

Gradient of [`rmsnorm`](@ref), returning `(dg, dx)`.
"""
function ∇rmsnorm(g, x::AbstractArray{T}, dy; dims=1, eps=1f-5) where {T}
    if use_fused_norm(x, dims, g)
        _, _, rσ = fused_norm(g, nothing, x, dims, eps, false; output=false)
        dg, _, dx = ∇fused_norm(g, nothing, x, dy, nothing, rσ, dims)
        return dg, dx
    end
    Tc = _stats_type(T)
    xc = Tc === T ? x : Tc.(x)
    dyc = Tc === T ? dy : Tc.(dy)
    rms = sqrt.(mean(abs2, xc; dims) .+ _epsof(xc, eps))
    x̂ = xc ./ rms
    dg = g === nothing ? nothing : _unbroadcast(dyc .* x̂, g)
    dx̂ = g === nothing ? dyc : dyc .* g
    dx = (dx̂ .- x̂ .* mean(dx̂ .* x̂; dims)) ./ rms
    return dg, Tc === T ? dx : T.(dx)
end

# --- rrules ------------------------------------------------------------------
# Route reverse-mode AD through the explicit gradient operators above. Because
# those are differentiable, the pullbacks are too, so nested AD yields correct
# second-order derivatives. On the GPU the cuDNN `batchnorm` `rrule` (typed on
# `DenseCuArray`, in `NNlibCUDACUDNNExt`) is more specific and wins for 3D/4D/5D
# `CuArray`s. The fused kernels keep the statistics of the forward pass for the
# pullback.

_tangent(::Nothing) = NoTangent()
_tangent(x) = x

function ChainRulesCore.rrule(::typeof(batchnorm), g, b, x::AbstractArray,
                              running_mean=nothing, running_var=nothing, momentum=0.1f0;
                              eps=1f-5, training::Bool=true, track_stats::Bool=true, kws...)
    if use_fused_batchnorm(g, b, x, running_mean, training)
        y, μ, rσ = fused_batchnorm(g, b, x, running_mean, running_var, momentum, eps, track_stats)
        function batchnorm_fused_pullback(Δ)
            Δ = unthunk(Δ)
            Δ isa AbstractZero && return (NoTangent(), Δ, Δ, Δ, NoTangent(), NoTangent(), NoTangent())
            dg, db, dx = ∇fused_batchnorm(g, b, x, Δ, μ, rσ)
            (NoTangent(), _tangent(dg), _tangent(db), dx, NoTangent(), NoTangent(), NoTangent())
        end
        return y, batchnorm_fused_pullback
    end
    y = batchnorm(g, b, x, running_mean, running_var, momentum; eps, training, track_stats)
    function batchnorm_pullback(Δ)
        dg, db, dx = ∇batchnorm(g, b, x, unthunk(Δ), running_mean, running_var, momentum; eps, training)
        (NoTangent(), _tangent(dg), _tangent(db), dx, NoTangent(), NoTangent(), NoTangent())
    end
    return y, batchnorm_pullback
end

function ChainRulesCore.rrule(::typeof(instancenorm), g, b, x::AbstractArray,
                              running_mean=nothing, running_var=nothing, momentum=0.1f0;
                              eps=1f-5, training::Bool=true, track_stats::Bool=false)
    if use_fused_instancenorm(g, b, x, running_mean, training)
        y, μ, rσ = fused_instancenorm(g, b, x, running_mean, running_var, momentum, eps, track_stats)
        function instancenorm_fused_pullback(Δ)
            Δ = unthunk(Δ)
            Δ isa AbstractZero && return (NoTangent(), Δ, Δ, Δ, NoTangent(), NoTangent(), NoTangent())
            dg, db, dx = ∇fused_norm(g, b, x, Δ, μ, rσ, NormGroups(size(x, ndims(x) - 1)))
            (NoTangent(), _tangent(dg), _tangent(db), dx, NoTangent(), NoTangent(), NoTangent())
        end
        return y, instancenorm_fused_pullback
    end
    y = instancenorm(g, b, x, running_mean, running_var, momentum; eps, training, track_stats)
    function instancenorm_pullback(Δ)
        dg, db, dx = ∇instancenorm(g, b, x, unthunk(Δ), running_mean, running_var, momentum; eps, training)
        (NoTangent(), _tangent(dg), _tangent(db), dx, NoTangent(), NoTangent(), NoTangent())
    end
    return y, instancenorm_pullback
end

function ChainRulesCore.rrule(::typeof(groupnorm), g, b, x::AbstractArray, G::Integer; eps=1f-5)
    _check_groupnorm(g, b, x, G)
    if use_fused_norm(x, NormGroups(G), g, b)
        y, μ, rσ = fused_norm(g, b, x, NormGroups(G), eps, true)
        function groupnorm_fused_pullback(Δ)
            Δ = unthunk(Δ)
            Δ isa AbstractZero && return (NoTangent(), Δ, Δ, Δ, NoTangent())
            dg, db, dx = ∇fused_norm(g, b, x, Δ, μ, rσ, NormGroups(G))
            (NoTangent(), _tangent(dg), _tangent(db), dx, NoTangent())
        end
        return y, groupnorm_fused_pullback
    end
    y = groupnorm(g, b, x, G; eps)
    function groupnorm_pullback(Δ)
        dg, db, dx = ∇groupnorm(g, b, x, unthunk(Δ), G; eps)
        (NoTangent(), _tangent(dg), _tangent(db), dx, NoTangent())
    end
    return y, groupnorm_pullback
end

function ChainRulesCore.rrule(::typeof(layernorm), g, b, x::AbstractArray; dims=1, eps=1f-5)
    _check_affine(g, b)
    _check_norm_types(eltype(x), g, b, nothing, nothing)
    if use_fused_norm(x, dims, g, b)
        y, μ, rσ = fused_norm(g, b, x, dims, eps, true)
        function layernorm_fused_pullback(Δ)
            Δ = unthunk(Δ)
            Δ isa AbstractZero && return (NoTangent(), Δ, Δ, Δ)
            dg, db, dx = ∇fused_norm(g, b, x, Δ, μ, rσ, dims)
            (NoTangent(), _tangent(dg), _tangent(db), dx)
        end
        return y, layernorm_fused_pullback
    end
    y = layernorm(g, b, x; dims, eps)
    function layernorm_pullback(Δ)
        dg, db, dx = ∇layernorm(g, b, x, unthunk(Δ); dims, eps)
        (NoTangent(), _tangent(dg), _tangent(db), dx)
    end
    return y, layernorm_pullback
end

function ChainRulesCore.rrule(::typeof(rmsnorm), g, x::AbstractArray; dims=1, eps=1f-5)
    _check_norm_types(eltype(x), g, nothing, nothing, nothing)
    if use_fused_norm(x, dims, g)
        y, _, rσ = fused_norm(g, nothing, x, dims, eps, false)
        function rmsnorm_fused_pullback(Δ)
            Δ = unthunk(Δ)
            Δ isa AbstractZero && return (NoTangent(), Δ, Δ)
            dg, _, dx = ∇fused_norm(g, nothing, x, Δ, nothing, rσ, dims)
            (NoTangent(), _tangent(dg), dx)
        end
        return y, rmsnorm_fused_pullback
    end
    y = rmsnorm(g, x; dims, eps)
    function rmsnorm_pullback(Δ)
        dg, dx = ∇rmsnorm(g, x, unthunk(Δ); dims, eps)
        (NoTangent(), _tangent(dg), dx)
    end
    return y, rmsnorm_pullback
end
