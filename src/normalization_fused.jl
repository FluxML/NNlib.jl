# Fused layer, RMS, group, instance and batch normalization.
#
# Normalizing over the leading `k` dimensions of `x` normalizes the columns of `x` seen as
# an `M × N` matrix, and so does normalizing groups of channels, which are contiguous in
# memory; the scale and bias have an entry per row in the first case, per block of rows (a
# channel) in the second. A kernel computes the statistics of a column and normalizes it,
# another computes the input gradient of a column (and the gradients of a scale and bias
# per channel), a third the gradients of a scale and bias per row, so that each reads `x`
# from memory once. The kernels run on the CPU and the GPUs alike, with different layouts:
# on a GPU the `S` lanes of a workgroup go down a column together and combine their sums in
# local memory; on the CPU a workgroup is a single work-item, which normalizes whole
# columns, and whose loops vectorize down them.

const FusedNormFloat = Union{Float16,BFloat16,Float32,Float64}

# The channels of `x` normalized in `G` groups, in place of the `dims` of `layernorm`.
struct NormGroups
    G::Int
end

# A column of `C` blocks of `L` rows, a block per channel of a group, the columns taking the
# `G` groups in turn.
struct ChannelBlocks
    L::Int
    C::Int
    G::Int
end

# Whether the kernels normalize `x` over `dims` with the scale and bias `ps`. Under
# automatic differentiation the generic code runs instead, as its rules nest.
use_fused_norm(x, dims, ps...) = !within_gradient(x) && fused_norm_applies(x, dims, ps...)

function fused_norm_applies(x::Union{Array{T},AbstractGPUArray{T}}, dims, ps...) where {T<:FusedNormFloat}
    k = leading_ndims(x, dims)
    (k == 0 || isempty(x)) && return false
    return all(p -> p === nothing || fused_norm_param(p, x, k), ps)
end
function fused_norm_applies(x::Union{Array{T},AbstractGPUArray{T}}, ::NormGroups, ps...) where {T<:FusedNormFloat}
    (ndims(x) < 3 || isempty(x)) && return false
    return all(p -> p === nothing || fused_norm_param(p, x), ps)
end
fused_norm_applies(x, dims, ps...) = false

@non_differentiable fused_norm_applies(::Any...)

# `k` if `dims` are the dimensions `1:k`, otherwise 0.
function leading_ndims(x, dims)
    dims isa Colon && return ndims(x)
    k = length(dims)
    return all(d -> d in dims, 1:k) ? k : 0
end

# A scale or bias of the shape of the normalized dimensions, or with an entry per channel,
# on the device of `x`, with the eltype of its statistics.
fused_norm_param(p::AbstractArray, x, k) =
    fused_norm_eltype(p, x) && length(p) == prod(d -> size(x, d), 1:k) &&
    all(d -> size(p, d) == size(x, d), 1:min(k, ndims(p)))
fused_norm_param(p::AbstractArray, x) = fused_norm_eltype(p, x) && length(p) == size(x, ndims(x) - 1)
fused_norm_param(p, x, k...) = false

fused_norm_eltype(p, x) = same_device(p, x) && eltype(p) === _stats_type(eltype(x))

same_device(p::Array, x::Array) = true
same_device(p::AbstractGPUArray, x::AbstractGPUArray) = get_backend(p) == get_backend(x)
same_device(p, x) = false

# `x` as the columns that are normalized, and the blocks of rows that take an entry of the
# scale and bias, `nothing` for an entry per row.
norm_columns(x, dims) = reshape(x, prod(d -> size(x, d), 1:leading_ndims(x, dims)), :), nothing
function norm_columns(x, ng::NormGroups)
    L = prod(d -> size(x, d), 1:ndims(x)-2)
    C = size(x, ndims(x) - 1) ÷ ng.G
    return reshape(x, L * C, :), ChannelBlocks(L, C, ng.G)
end

# The rows per block and the blocks per column.
norm_blocks(::Nothing, M) = M, 1
norm_blocks(cb::ChannelBlocks, M) = cb.L, cb.C

# The entries of the scale and bias before those of column `j`, and the scale or bias over its
# block `c`: all of it if it has an entry per row.
@inline block_offset(::Nothing, j) = 0
@inline block_offset(cb::ChannelBlocks, j) = ((j - 1) % cb.G) * cb.C
@inline block_param(p, ::Nothing, c, o) = p
@inline block_param(::Nothing, ::ChannelBlocks, c, o) = nothing
@inline block_param(p, ::ChannelBlocks, c, o) = @inbounds p[o + c]

# A scale per row multiplies the terms of the sums over a block, a scale per block the sums.
@inline term_scale(g::AbstractArray) = g
@inline term_scale(g) = nothing
@inline sum_scale(g::Number) = g
@inline sum_scale(g) = nothing

maybe_vec(::Nothing) = nothing
maybe_vec(p) = vec(p)

@inline norm_scale(v, ::Nothing, i) = v
@inline norm_scale(v, g::Number, i) = v * g
@inline norm_scale(v, g, i) = v * @inbounds(g[i])
@inline norm_shift(v, ::Nothing, i) = v
@inline norm_shift(v, b::Number, i) = v + b
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
        y, μ, rσ, @Const(x), g, b, blocks, eps, jstride, ::Val{S}, ::Val{center}) where {S,center}
    t = @index(Local, Linear)
    j = @index(Group, Linear)
    M, N = size(x)
    L, nb = norm_blocks(blocks, M)
    A = eltype(rσ)
    shared = @localmem eltype(rσ) (S,)
    while j <= N
        po = block_offset(blocks, j)
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
            for c in 1:nb
                gc = block_param(g, blocks, c, po)
                bc = block_param(b, blocks, c, po)
                o = (c - 1) * L
                @simd for i in t:S:L
                    q = o + i
                    y[q, j] = norm_shift(norm_scale(r * (A(x[q, j]) - m), gc, q), bc, q)
                end
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
# without the mean of `dx̂` if not centering. With a scale and bias per block, the sums of
# `dy * x̂` and `dy` over block `k` go to `pg[k, j]` and `pb[k, j]`.
@kernel inbounds=true unsafe_indices=true function fused_norm_dx_kernel!(
        dx, pg, pb, @Const(dy), @Const(x), g, μ, @Const(rσ), blocks, jstride, ::Val{S}) where {S}
    t = @index(Local, Linear)
    j = @index(Group, Linear)
    M, N = size(x)
    L, nb = norm_blocks(blocks, M)
    A = eltype(rσ)
    partials = pg !== nothing || pb !== nothing
    shared = @localmem NTuple{2,eltype(rσ)} (S,)
    while j <= N
        po = block_offset(blocks, j)
        m = norm_load(μ, j, A)
        r = rσ[j]
        a = zero(A)
        c = zero(A)
        for k in 1:nb
            gk = block_param(g, blocks, k, po)
            o = (k - 1) * L
            sd = zero(A)
            sx = zero(A)
            @simd for i in t:S:L
                q = o + i
                d = norm_scale(A(dy[q, j]), term_scale(gk), q)
                sd += d
                sx += d * (A(x[q, j]) - m)
            end
            if partials
                if S > 1
                    sd, sx = group_reduce(add_pairs, shared, (sd, sx), t, Val(S))
                end
                if t == 1
                    norm_store!(pg, CartesianIndex(k, j), sx * r)
                    norm_store!(pb, CartesianIndex(k, j), sd)
                end
            end
            a += norm_scale(sd, sum_scale(gk), k)
            c += norm_scale(sx, sum_scale(gk), k)
        end
        if !partials && S > 1
            a, c = group_reduce(add_pairs, shared, (a, c), t, Val(S))
        end
        m1 = μ === nothing ? zero(A) : a / M
        m2 = c * r / M
        for k in 1:nb
            gk = block_param(g, blocks, k, po)
            o = (k - 1) * L
            @simd for i in t:S:L
                q = o + i
                dx[q, j] = r * (norm_scale(A(dy[q, j]), gk, q) - m1 - (A(x[q, j]) - m) * r * m2)
            end
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

# The normalization of `x` over its leading dimensions `dims`, or over groups of channels
# (`dims::NormGroups`), centered (`layernorm`) or not (`rmsnorm`), with the mean (if
# centering) and the reciprocal deviation of every column. `output=Val(false)` computes the
# statistics only.
function fused_norm(g, b, x::AbstractArray{T}, dims, eps, center::Bool; output::Val=Val(true)) where {T}
    X, blocks = norm_columns(x, dims)
    M, N = size(X)
    A = _stats_type(T)
    backend = get_backend(X)
    S, G = fused_norm_layout(backend, M, N)
    y = output === Val(true) ? similar(X) : nothing
    μ = center ? similar(X, A, N) : nothing
    rσ = similar(X, A, N)
    fused_norm_kernel!(backend, S)(y, μ, rσ, X, maybe_vec(g), maybe_vec(b), blocks, A(eps), G,
                                   Val(S), Val(center); ndrange=S * G)
    return y === nothing ? nothing : reshape(y, size(x)), μ, rσ
end

# The sums over every block of every column, for the gradient of a scale or bias per channel.
block_partials(p, X, blocks, A) =
    p === nothing || blocks === nothing ? nothing : similar(X, A, blocks.C, size(X, 2))

# The gradient of a scale or bias per channel, from its partial sums over every sample.
channel_grad(::Nothing, p) = nothing
channel_grad(s, p) = reshape(sum(reshape(s, length(p), :); dims=2), size(p))

# `(dg, db, dx)` from the statistics of `fused_norm`.
function ∇fused_norm(g, b, x::AbstractArray, dy, μ, rσ, dims)
    X, blocks = norm_columns(x, dims)
    dY = reshape(same_device(dy, x) ? dy : copyto!(similar(x, eltype(dy)), dy), size(X))
    M, N = size(X)
    backend = get_backend(X)
    S, G = fused_norm_layout(backend, M, N)
    dx = similar(X)
    pg = block_partials(g, X, blocks, eltype(rσ))
    pb = block_partials(b, X, blocks, eltype(rσ))
    fused_norm_dx_kernel!(backend, S)(dx, pg, pb, dY, X, maybe_vec(g), μ, rσ, blocks, G, Val(S);
                                      ndrange=S * G)
    blocks === nothing || return channel_grad(pg, g), channel_grad(pb, b), reshape(dx, size(x))
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

# --- Batch normalization -------------------------------------------------------------
# Seen as a matrix of `Sp·C` rows (`Sp` the spatial size, `C` channels) and a column per
# sample, batch normalization reduces every block of `Sp` rows (a channel) over all columns.
# A kernel sums over the columns of every row, as the one of the parameter gradients above;
# another adds up the rows of every channel into its statistics (or the gradients of its
# scale and bias) and the coefficients of the output (or of the input gradient), which a
# third kernel, or a broadcast on the CPU, computes elementwise. The sums are shifted by an
# element of each channel (by its mean in the gradient), for accuracy.

function fused_batchnorm_applies(x::Union{Array{T},AbstractGPUArray{T}}, ps...) where {T<:FusedNormFloat}
    (ndims(x) < 2 || isempty(x)) && return false
    return all(p -> p === nothing || fused_norm_param(p, x), ps)
end
fused_batchnorm_applies(x, ps...) = false

@non_differentiable fused_batchnorm_applies(::Any...)

# `x` as the matrix above, and the rows per channel.
batchnorm_rows(x) = reshape(x, :, size(x, ndims(x))), prod(d -> size(x, d), 1:ndims(x)-2; init=1)

# The shift of row `i`: the first element of its channel, or the channel's mean `μ`.
@inline batchnorm_shift(x, ::Nothing, i, Sp, ::Type{A}) where {A} = A(@inbounds x[i - (i - 1) % Sp, 1])
@inline batchnorm_shift(x, μ, i, Sp, ::Type{A}) where {A} = @inbounds μ[(i - 1) ÷ Sp + 1]
@inline function batchnorm_shifts(x, μ, i0, Sp, ::Type{A}, ::Val{S}, ::Val{R}) where {A,S,R}
    if (i0 - 1) ÷ Sp == (i0 + (R - 1) * S - 1) ÷ Sp
        m = batchnorm_shift(x, μ, i0, Sp, A)
        return ntuple(_ -> m, Val(R))
    end
    return ntuple(q -> batchnorm_shift(x, μ, i0 + (q - 1) * S, Sp, A), Val(R))
end

# `d = x - m` and `d²` added to the sums of `R` rows `S` apart, or `dy` and `dy * d`.
@inline function batchnorm_tile(s1::NTuple{R,A}, s2::NTuple{R,A}, ::Nothing, x, m, i0, j, ::Val{S}) where {R,A,S}
    d = ntuple(q -> A(@inbounds x[i0 + (q - 1) * S, j]) - m[q], Val(R))
    return ntuple(q -> s1[q] + d[q], Val(R)), ntuple(q -> s2[q] + d[q] * d[q], Val(R))
end
@inline function batchnorm_tile(s1::NTuple{R,A}, s2::NTuple{R,A}, dy, x, m, i0, j, ::Val{S}) where {R,A,S}
    e = ntuple(q -> A(@inbounds dy[i0 + (q - 1) * S, j]), Val(R))
    d = ntuple(q -> A(@inbounds x[i0 + (q - 1) * S, j]) - m[q], Val(R))
    return ntuple(q -> s1[q] + e[q], Val(R)), ntuple(q -> s2[q] + e[q] * d[q], Val(R))
end

# The sums over the columns `(k - 1) * L + 1 : k * L` of every row, into `p1[:, k]` and
# `p2[:, k]`, with the layout of `fused_norm_params_kernel!`.
@kernel inbounds=true unsafe_indices=true function batchnorm_sums_kernel!(
        p1, p2, @Const(dy), @Const(x), μ, Sp, L, MR, ::Val{S}, ::Val{R}) where {S,R}
    t = @index(Local, Linear)
    w = @index(Group, Linear) - 1
    M, N = size(x)
    A = eltype(p1)
    nblocks = cld(M, MR)
    k = w ÷ nblocks + 1
    js = ((k - 1) * L + 1):min(k * L, N)
    iend = min((w % nblocks + 1) * MR, M)
    i0 = (w % nblocks) * MR + t
    while i0 + (R - 1) * S <= iend
        m = batchnorm_shifts(x, μ, i0, Sp, A, Val(S), Val(R))
        s1 = ntuple(_ -> zero(A), Val(R))
        s2 = s1
        for j in js
            s1, s2 = batchnorm_tile(s1, s2, dy, x, m, i0, j, Val(S))
        end
        for q in 1:R
            p1[i0 + (q - 1) * S, k] = s1[q]
            p2[i0 + (q - 1) * S, k] = s2[q]
        end
        i0 += R * S
    end
    for i in i0:S:iend
        m = (batchnorm_shift(x, μ, i, Sp, A),)
        s1 = (zero(A),)
        s2 = s1
        for j in js
            s1, s2 = batchnorm_tile(s1, s2, dy, x, m, i, j, Val(S))
        end
        p1[i, k] = s1[1]
        p2[i, k] = s2[1]
    end
end

# The sums of `p1` and `p2` over the rows of channel `c` and all columns, by the `S` lanes of
# workgroup `c` and every `cstride`-th one after it, which split into lanes down the rows
# and lanes across the columns if the channel has fewer rows than lanes; then
# `finish!(c, s1, s2, args...)`.
@kernel inbounds=true unsafe_indices=true function batchnorm_channels_kernel!(
        finish!, args, @Const(p1), @Const(p2), Sp, C, cstride, ::Val{S}) where {S}
    t = @index(Local, Linear)
    c = @index(Group, Linear)
    A = eltype(p1)
    shared = @localmem NTuple{2,eltype(p1)} (S,)
    Si = min(S, prevpow(2, Sp))
    ti = (t - 1) % Si + 1
    tk = (t - 1) ÷ Si + 1
    while c <= C
        o = (c - 1) * Sp
        s1 = zero(A)
        s2 = zero(A)
        for k in tk:(S ÷ Si):size(p1, 2)
            @simd for i in ti:Si:Sp
                s1 += p1[o + i, k]
                s2 += p2[o + i, k]
            end
        end
        if S > 1
            s1, s2 = group_reduce(add_pairs, shared, (s1, s2), t, Val(S))
        end
        if t == 1
            finish!(c, s1, s2, args...)
        end
        c += cstride
    end
end

# The layout of `batchnorm_sums_kernel!`. On the CPU, tiles of 64 rows, which read the columns
# in runs of 256 bytes (in Float32), and a few workgroups per thread, taking blocks of rows,
# and blocks of columns only if the rows are too few, as the partial sums of a block of
# columns span all rows; on a GPU that of `fused_norm_params_kernel!`.
function batchnorm_sums_layout(backend, M, N)
    backend isa CPU || return fused_norm_params_layout(backend, M, N)
    R = 64
    n = should_use_spawn() ? 4 * Threads.nthreads(:default) : 1
    K = clamp(cld(n, cld(M, R)), 1, max(N ÷ 8, 1))
    return 1, R, cld(N, K), R * cld(M, R * cld(n, K))
end

# The sums over the elements of every channel, of `x` shifted by the first one or of `dy` and
# `dy * (x - μ)`, finished in every channel `c` by `finish!(c, s1, s2, args...)`.
function batchnorm_sums!(finish!, dy, x::AbstractArray{T}, μ, args...) where {T}
    X, Sp = batchnorm_rows(x)
    M, N = size(X)
    backend = get_backend(X)
    S, R, L, MR = batchnorm_sums_layout(backend, M, N)
    K = cld(N, L)
    p1 = similar(X, _stats_type(T), M, K)
    p2 = similar(p1)
    dY = dy === nothing ? nothing : reshape(dy, size(X))
    batchnorm_sums_kernel!(backend, S)(p1, p2, dY, X, μ, Sp, L, MR, Val(S), Val(R);
                                       ndrange=S * cld(M, MR) * K)
    C = M ÷ Sp
    S, G = fused_norm_layout(backend, Sp * K, C)
    batchnorm_channels_kernel!(backend, S)(finish!, args, p1, p2, Sp, C, G, Val(S); ndrange=S * G)
    return nothing
end

# The mean and the reciprocal deviation of channel `c` over its `n` elements, from the sums of
# `x` shifted by its first element, and the coefficients of `batchnorm_y`.
@inline function batchnorm_stats!(c, s1, s2, μ, rσ, coef, x, g, b, Sp, n, eps)
    m = eltype(μ)(@inbounds x[(c - 1) * Sp + 1, 1]) + s1 / n
    r = inv(sqrt(max(s2 - s1 * s1 / n, zero(s2)) / n + eps))
    @inbounds μ[c] = m
    @inbounds rσ[c] = r
    @inbounds coef[c] = (m, norm_scale(r, g, c), norm_shift(zero(r), b, c))
    return nothing
end

# The mean and the reciprocal deviation of every channel, and the coefficients of the output.
function fused_batchnorm_stats(g, b, x::AbstractArray{T,N}, eps) where {T,N}
    A = _stats_type(T)
    C = size(x, N-1)
    μ = similar(x, A, C)
    rσ = similar(x, A, C)
    coef = similar(x, NTuple{3,A}, C)
    X, Sp = batchnorm_rows(x)
    batchnorm_sums!(batchnorm_stats!, nothing, x, nothing, μ, rσ, coef, X, maybe_vec(g), maybe_vec(b),
                    Sp, A(length(x) ÷ C), A(eps))
    return μ, rσ, coef
end

# `y[i] = f(coef[c], xs[1][i], …)` for every element `i` of channel `c`, every work-item
# striding over the elements.
@kernel inbounds=true unsafe_indices=true function batchnorm_map_kernel!(
        f, y, coef, xs, Sp, C, stride)
    i = @index(Global, Linear)
    while i <= length(y)
        y[i] = f(coef[((i - 1) ÷ Sp) % C + 1], load_all(xs, i)...)
        i += stride
    end
end

@inline load_all(xs, i) = map(a -> @inbounds(a[i]), xs)

# `batchnorm_map_kernel!` on a GPU, about eight elements per lane; a broadcast on the CPU,
# where the elements of a channel take the inner dimension, unless there's one per sample.
function batchnorm_map!(f, y::AbstractArray{T,N}, coef, xs...) where {T,N}
    Sp = prod(d -> size(y, d), 1:N-2; init=1)
    C = size(y, N-1)
    backend = get_backend(y)
    if backend isa CPU
        if Sp == 1
            reshape(y, C, :) .= f.(coef, map(x -> reshape(x, C, :), xs)...)
        else
            reshape(y, Sp, C, :) .= f.(reshape(coef, 1, C, 1), map(x -> reshape(x, Sp, C, :), xs)...)
        end
    else
        G = 256 * cld(length(y), 256 * 8)
        batchnorm_map_kernel!(backend, 256)(f, y, coef, xs, Sp, C, G; ndrange=G)
    end
    return y
end

@inline batchnorm_y((μ, a, b), x) = (x - μ) * a + b

# The gradients of the scale and bias of channel `c`, from the sums of `dy` and
# `dy * (x - μ)` over its `n` elements, and the coefficients of `batchnorm_dx`.
@inline function batchnorm_grads!(c, s1, s2, dg, db, coef, μ, rσ, g, n)
    r = @inbounds rσ[c]
    norm_store!(dg, c, s2 * r)
    norm_store!(db, c, s1)
    a = norm_scale(r, g, c)
    @inbounds coef[c] = (μ[c], a, a * s1 / n, a * r * r * s2 / n)
    return nothing
end

# `(dg, db, dx)` from the statistics of `fused_batchnorm_stats`:
# `dx = g * rσ * (dy - mean(dy) - (x - μ) * rσ² * mean(dy * (x - μ)))`.
function ∇fused_batchnorm(g, b, x::AbstractArray{T,N}, dy, μ, rσ) where {T,N}
    dy = same_device(dy, x) ? dy : copyto!(similar(x, eltype(dy)), dy)
    C = size(x, N-1)
    dg = g === nothing ? nothing : similar(g)
    db = b === nothing ? nothing : similar(b)
    coef = similar(rσ, NTuple{4,eltype(rσ)}, C)
    batchnorm_sums!(batchnorm_grads!, dy, x, μ, maybe_vec(dg), maybe_vec(db), coef, μ, rσ, maybe_vec(g),
                    eltype(rσ)(length(x) ÷ C))
    return dg, db, batchnorm_map!(batchnorm_dx, similar(x), coef, x, dy)
end

@inline batchnorm_dx((μ, a, b, c), x, dy) = a * dy - b - c * (x - μ)
