# Fused layer and RMS normalization.
#
# Normalizing over the leading `k` dimensions of `x` normalizes the columns of `x` seen as
# an `M × N` matrix. A kernel computes the statistics of a column and normalizes it,
# another computes the input gradient of a column, a third the gradients of the scale and
# bias, so that each reads `x` from memory once. The kernels run on the CPU and the GPUs
# alike, with different layouts: on a GPU the `S` lanes of a workgroup go down a column
# together and combine their sums in local memory; on the CPU a workgroup is a single
# work-item, which normalizes whole columns, and whose loops vectorize down them.

const FusedNormFloat = Union{Float16,BFloat16,Float32,Float64}

# Whether the kernels normalize `x` over `dims` with the scale and bias `ps`. Under
# automatic differentiation the generic code runs instead, as its rules nest.
use_fused_norm(x, dims, ps...) = !within_gradient(x) && fused_norm_applies(x, dims, ps...)

function fused_norm_applies(x::Union{Array{T},AbstractGPUArray{T}}, dims, ps...) where {T<:FusedNormFloat}
    k = leading_ndims(x, dims)
    (k == 0 || isempty(x)) && return false
    return all(p -> p === nothing || fused_norm_param(p, x, k), ps)
end
fused_norm_applies(x, dims, ps...) = false

@non_differentiable fused_norm_applies(::Any...)

# `k` if `dims` are the dimensions `1:k`, otherwise 0.
function leading_ndims(x, dims)
    dims isa Colon && return ndims(x)
    k = length(dims)
    return all(d -> d in dims, 1:k) ? k : 0
end

# A scale or bias of the shape of the normalized dimensions, on the device of `x`, with the
# eltype of its statistics.
fused_norm_param(p::AbstractArray, x, k) =
    same_device(p, x) && eltype(p) === _stats_type(eltype(x)) &&
    length(p) == prod(d -> size(x, d), 1:k) && all(d -> size(p, d) == size(x, d), 1:min(k, ndims(p)))
fused_norm_param(p, x, k) = false

same_device(p::Array, x::Array) = true
same_device(p::AbstractGPUArray, x::AbstractGPUArray) = get_backend(p) == get_backend(x)
same_device(p, x) = false

norm_columns(x, dims) = reshape(x, prod(d -> size(x, d), 1:leading_ndims(x, dims)), :)

maybe_vec(::Nothing) = nothing
maybe_vec(p) = vec(p)

@inline norm_scale(v, ::Nothing, i) = v
@inline norm_scale(v, g, i) = v * @inbounds(g[i])
@inline norm_shift(v, ::Nothing, i) = v
@inline norm_shift(v, b, i) = v + @inbounds(b[i])
@inline norm_load(::Nothing, j, ::Type{A}) where {A} = zero(A)
@inline norm_load(a, j, ::Type{A}) where {A} = @inbounds a[j]
@inline norm_store!(::Nothing, I, v) = nothing
@inline norm_store!(a, I, v) = (@inbounds a[I] = v; nothing)
@inline add_pairs(a, b) = (a[1] + b[1], a[2] + b[2])

# `v` reduced with `op` over the `S` lanes of a workgroup (a power of 2), on every lane,
# through the local memory `shared`. (`@groupreduce` needs KernelAbstractions 0.10.)
@inline function group_reduce(op, shared, v, t, ::Val{S}) where {S}
    @inbounds shared[t] = v
    @synchronize
    s = S ÷ 2
    while s > 0
        if t <= s
            @inbounds shared[t] = op(shared[t], shared[t + s])
        end
        @synchronize
        s ÷= 2
    end
    r = @inbounds shared[1]
    @synchronize
    return r
end

# The `S` lanes of workgroup `j` normalize column `j`, then every `jstride`-th one after it:
# `y = g * (x - μ) * rσ + b`, with `μ = 0` and no `b` without centering (`rmsnorm`).
# `y === nothing` computes the statistics only.
@kernel inbounds=true unsafe_indices=true function fused_norm_kernel!(
        y, μ, rσ, @Const(x), g, b, eps, jstride, ::Val{S}, ::Val{center}) where {S,center}
    t = @index(Local, Linear)
    j = @index(Group, Linear)
    M, N = size(x)
    A = eltype(rσ)
    shared = @localmem eltype(rσ) (S,)
    while j <= N
        m = zero(A)
        if center
            s = zero(A)
            @simd for i in t:S:M
                s += A(x[i, j])
            end
            if S > 1
                s = group_reduce(+, shared, s, t, Val(S))
            end
            m = s / M
        end
        v = zero(A)
        @simd for i in t:S:M
            v += abs2(A(x[i, j]) - m)
        end
        if S > 1
            v = group_reduce(+, shared, v, t, Val(S))
        end
        r = inv(sqrt(v / M + eps))
        if y !== nothing
            @simd for i in t:S:M
                y[i, j] = norm_shift(norm_scale(r * (A(x[i, j]) - m), g, i), b, i)
            end
        end
        if t == 1
            norm_store!(μ, j, m)
            rσ[j] = r
        end
        j += jstride
    end
end

# With `x̂ = (x - μ) * rσ` and `dx̂ = g * dy`: `dx = rσ * (dx̂ - mean(dx̂) - x̂ * mean(dx̂ * x̂))`,
# without the mean of `dx̂` if not centering.
@kernel inbounds=true unsafe_indices=true function fused_norm_dx_kernel!(
        dx, @Const(dy), @Const(x), g, μ, @Const(rσ), jstride, ::Val{S}) where {S}
    t = @index(Local, Linear)
    j = @index(Group, Linear)
    M, N = size(x)
    A = eltype(rσ)
    shared = @localmem NTuple{2,eltype(rσ)} (S,)
    while j <= N
        m = norm_load(μ, j, A)
        r = rσ[j]
        a = zero(A)
        c = zero(A)
        @simd for i in t:S:M
            dyg = norm_scale(A(dy[i, j]), g, i)
            a += dyg
            c += dyg * (A(x[i, j]) - m)
        end
        if S > 1
            a, c = group_reduce(add_pairs, shared, (a, c), t, Val(S))
        end
        m1 = μ === nothing ? zero(A) : a / M
        m2 = c * r / M
        @simd for i in t:S:M
            dx[i, j] = r * (norm_scale(A(dy[i, j]), g, i) - m1 - (A(x[i, j]) - m) * r * m2)
        end
        j += jstride
    end
end

@inline function norm_params_tile(sg::NTuple{R,A}, sb::NTuple{R,A}, dy, x, i0, j, m, r, ::Val{S}) where {R,A,S}
    d = ntuple(q -> A(@inbounds dy[i0 + (q - 1) * S, j]), Val(R))
    h = ntuple(q -> (A(@inbounds x[i0 + (q - 1) * S, j]) - m) * r, Val(R))
    return ntuple(q -> sg[q] + d[q] * h[q], Val(R)), ntuple(q -> sb[q] + d[q], Val(R))
end

# The sums of `dy * x̂` and `dy` over the `L` columns `(k - 1) * L + 1 : k * L`, into
# `pg[:, k]` and `pb[:, k]`, a workgroup for every block of `MR` rows of them. Its `S` lanes
# take the rows in tiles of `R` rows, `S` apart: on the CPU, `R` consecutive rows, which
# vectorize.
@kernel inbounds=true unsafe_indices=true function fused_norm_params_kernel!(
        pg, pb, @Const(dy), @Const(x), μ, @Const(rσ), L, MR, ::Val{S}, ::Val{R}) where {S,R}
    t = @index(Local, Linear)
    w = @index(Group, Linear) - 1
    M, N = size(x)
    A = eltype(rσ)
    nblocks = cld(M, MR)
    k = w ÷ nblocks + 1
    js = ((k - 1) * L + 1):min(k * L, N)
    iend = min((w % nblocks + 1) * MR, M)
    i0 = (w % nblocks) * MR + t
    while i0 + (R - 1) * S <= iend
        sg = ntuple(_ -> zero(A), Val(R))
        sb = sg
        for j in js
            sg, sb = norm_params_tile(sg, sb, dy, x, i0, j, norm_load(μ, j, A), rσ[j], Val(S))
        end
        for q in 1:R
            pg[i0 + (q - 1) * S, k] = sg[q]
            norm_store!(pb, CartesianIndex(i0 + (q - 1) * S, k), sb[q])
        end
        i0 += R * S
    end
    for i in i0:S:iend
        sg = zero(A)
        sb = zero(A)
        for j in js
            d = A(dy[i, j])
            sb += d
            sg += d * (A(x[i, j]) - norm_load(μ, j, A)) * rσ[j]
        end
        pg[i, k] = sg
        norm_store!(pb, CartesianIndex(i, k), sb)
    end
end

# Lanes per column, and the number of workgroups: on the CPU a few per thread, each taking
# every `G`-th column; on a GPU one per column, with about eight elements per lane.
function fused_norm_layout(backend, M, N)
    backend isa CPU && return 1, (should_use_spawn() ? min(N, 8 * Threads.nthreads(:default)) : 1)
    return clamp(prevpow(2, max(M ÷ 8, 1)), 32, 1024), N
end

# Lanes per workgroup, rows per tile, and the columns and rows of a workgroup of the
# parameter gradients. On the CPU a few workgroups per thread, with at most `N ÷ 8` blocks of
# columns, so that the partial sums stay small, and blocks of rows for the rest; on a GPU a
# row per lane of 64 columns.
function fused_norm_params_layout(backend, M, N)
    if backend isa CPU
        n = should_use_spawn() ? 4 * Threads.nthreads(:default) : 1
        K = clamp(N ÷ 8, 1, n)
        return 1, 16, cld(N, K), 16 * cld(M, 16 * cld(n, K))
    end
    S = clamp(prevpow(2, M), 32, 256)
    return S, 1, 64, S
end

# The normalization of `x` over its leading dimensions `dims`, centered (`layernorm`) or
# not (`rmsnorm`), with the mean (if centering) and the reciprocal deviation of every
# column. `output=false` computes the statistics only.
function fused_norm(g, b, x::AbstractArray{T}, dims, eps, center::Bool; output::Bool=true) where {T}
    X = norm_columns(x, dims)
    M, N = size(X)
    A = _stats_type(T)
    backend = get_backend(X)
    S, G = fused_norm_layout(backend, M, N)
    y = output ? similar(X) : nothing
    μ = center ? similar(X, A, N) : nothing
    rσ = similar(X, A, N)
    fused_norm_kernel!(backend, S)(y, μ, rσ, X, maybe_vec(g), maybe_vec(b), A(eps), G, Val(S), Val(center);
                                   ndrange=S * G)
    return output ? reshape(y, size(x)) : nothing, μ, rσ
end

# `(dg, db, dx)` from the statistics of `fused_norm`.
function ∇fused_norm(g, b, x::AbstractArray, dy, μ, rσ, dims)
    X = norm_columns(x, dims)
    dY = reshape(same_device(dy, x) ? dy : copyto!(similar(x, eltype(dy)), dy), size(X))
    M, N = size(X)
    backend = get_backend(X)
    S, G = fused_norm_layout(backend, M, N)
    dx = similar(X)
    fused_norm_dx_kernel!(backend, S)(dx, dY, X, maybe_vec(g), μ, rσ, G, Val(S); ndrange=S * G)
    g === nothing && return nothing, nothing, reshape(dx, size(x))
    S, R, L, MR = fused_norm_params_layout(backend, M, N)
    K = cld(N, L)
    pg = similar(X, eltype(rσ), M, K)
    pb = b === nothing ? nothing : similar(X, eltype(rσ), M, K)
    fused_norm_params_kernel!(backend, S)(pg, pb, dY, X, μ, rσ, L, MR, Val(S), Val(R);
                                          ndrange=S * cld(M, MR) * K)
    dg = reshape(sum(pg; dims=2), size(g))
    db = b === nothing ? nothing : reshape(sum(pb; dims=2), size(b))
    return dg, db, reshape(dx, size(x))
end
