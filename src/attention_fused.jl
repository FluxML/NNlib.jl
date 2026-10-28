# Fused scaled dot-product attention.
#
# The output is computed tile by tile, with the softmax taken online over the tiles of
# keys, so that the `(kv_len, q_len, nheads, batch_size)` attention weights are never
# materialized: memory grows linearly with the sequence length. On the CPU the tiles are
# walked by Julia tasks, on the GPU backends by a KernelAbstractions kernel.

const FusedAttentionFloat = Union{Float16,BFloat16,Float32,Float64}

# Half-precision inputs are accumulated in single precision.
attention_compute_type(::Type{T}) where {T} = sizeof(T) < 4 ? Float32 : T

# Whether the fused implementation computes `scaled_dot_product_attention` for these
# `(head_dim, nheads, seq_len, batch_size)` arrays. It needs dense arrays of one of the
# floating-point types above living on the same device, and no dropout: the weights
# `fdrop` would act on are never formed.
function fused_attention_applies(q::AA4{T}, k::AA4{T}, v::AA4{T}, bias, fdrop, mask) where {T<:FusedAttentionFloat}
    fdrop === identity || return false
    isempty(q) | isempty(k) | isempty(v) && return false
    fused_attention_arrays(q, k, v, bias, mask) || return false
    sz = (size(k, 3), size(q, 3), size(q, 2), size(q, 4))
    return attention_broadcasts(bias, sz) && attention_broadcasts(mask, sz) &&
           (bias === nothing || promote_type(T, eltype(bias)) === T)
end
fused_attention_applies(q, k, v, bias, fdrop, mask) = false

@non_differentiable fused_attention_applies(::Any...)

const MaybeArray{T} = Union{Nothing,Array{<:T}}
const MaybeGPUArray{T} = Union{Nothing,AbstractGPUArray{<:T}}

# `BFloat16` arithmetic is emulated on the CPU.
fused_attention_arrays(q::Array{T}, k::Array, v::Array, bias::MaybeArray{Real},
                       mask::Union{Nothing,Array{Bool},BitArray}) where {T} = T !== BFloat16

function fused_attention_arrays(q::AbstractGPUArray, k::AbstractGPUArray, v::AbstractGPUArray,
                                bias::MaybeGPUArray{Real}, mask::MaybeGPUArray{Bool})
    backend = get_backend(q)
    # the output has the head dimension of the values, the gradients that of the queries too
    return all(x -> x === nothing || get_backend(x) == backend, (k, v, bias, mask)) &&
           attention_tiling(backend, q, v) !== nothing && attention_tiling(backend, q, q) !== nothing
end

fused_attention_arrays(q, k, v, bias, mask) = false

# A `bias` or `mask` the kernels can index: every dimension is full or a singleton.
attention_broadcasts(x::Nothing, sz) = true
attention_broadcasts(x::AbstractArray, sz) =
    ndims(x) <= 4 && all(d -> size(x, d) == 1 || size(x, d) == sz[d], 1:4)

# The output of `scaled_dot_product_attention` for inputs satisfying
# `fused_attention_applies`, in the layout of `_scaled_dot_product_attention`, with
# `window` the band of `attention_band`. An array `lse` receives what the gradient needs
# of the softmax: the logarithm of the sum of the exponentials of the logits of each
# query, as `lse[h, i, b]`.
function fused_attention(q::AA4{T}, k::AA4{T}, v::AA4{T}, bias, mask, scale, is_causal::Bool,
                         q_offset, q_lengths, kv_lengths, segments, window; lse=nothing) where {T}
    o = similar(q, size(v, 1), size(q, 2), size(q, 3), size(q, 4))
    fused_attention!(get_backend(q), o, lse, q, k, v, attention_pad4(bias), attention_pad4(mask),
                     attention_vec_offsets(q_offset), attention_vec(q_lengths), attention_vec(kv_lengths),
                     attention_seqs(q, k, segments), attention_scale(attention_compute_type(T), q, scale),
                     is_causal, window)
    return o
end

# The sequences the kernels go through: the batch elements, or the segments of packed
# arrays, which end at `q_ends` and `kv_ends` along the sequence dimension of their only
# batch element.
attention_seqs(q, k, segments::Nothing) =
    (; q_ends = nothing, kv_ends = nothing, n = size(q, 4), maxlen_q = size(q, 3), maxlen_kv = size(k, 3))
attention_seqs(q, k, s::AttentionSegments) =
    (; s.q_ends, s.kv_ends, n = length(s.q_ends), maxlen_q = s.maxlen_q::Int, maxlen_kv = s.maxlen_kv::Int)

attention_scale(::Type{A}, q, scale::Nothing) where {A} = inv(sqrt(A(size(q, 1))))
attention_scale(::Type{A}, q, scale) where {A} = A(scale)

attention_pad4(x::Nothing) = nothing
attention_pad4(x::AbstractArray) = reshape(x, size(x, 1), size(x, 2), size(x, 3), size(x, 4))
attention_vec(x::Nothing) = nothing
attention_vec(x::AbstractArray) = vec(x)
# A zero offset is passed as `nothing`, with which the kernels compile as they would
# without offsets: keeping one in a register makes a gradient kernel spill.
attention_vec_offsets(x::Integer) = iszero(x) ? nothing : Int(x)
attention_vec_offsets(x::AbstractArray) = vec(x)

# The entry of a `bias` or `mask` for the key `j` and the query `i`.
@inline attention_sel(x, n) = ifelse(n == 1, 1, x)
@inline attention_at(a::Nothing, j, i, h, b) = nothing
@inline attention_at(a, j, i, h, b) = @inbounds a[attention_sel(j, size(a, 1)), attention_sel(i, size(a, 2)),
                                                  attention_sel(h, size(a, 3)), attention_sel(b, size(a, 4))]
@inline attention_len(l::Nothing, b, n) = n
@inline attention_len(l, b, n) = Int(@inbounds l[b])
# the offset of the queries of the batch element `b` under a causal mask
@inline attention_offset(o::Nothing, b) = 0
@inline attention_offset(o::Integer, b) = o
@inline attention_offset(o, b) = Int(@inbounds o[b])
# Within a band of `window` keys, the query `i` attends to the keys `j > i + off - window`:
# the first of them (or the first key, counting from zero or one), how many keys from `j0`
# on are before it, and whether it is before `j`.
@inline attention_window_first(window::Nothing, i, off) = 0
@inline attention_window_first(window, i, off) = i + off - window + 1
@inline attention_window_skip(window::Nothing, i, off, j0) = 0
@inline attention_window_skip(window, i, off, j0) = attention_window_first(window, i, off) - j0
@inline attention_in_window(window::Nothing, j, i, off) = true
@inline attention_in_window(window, j, i, off) = j >= attention_window_first(window, i, off)
# the last of the queries up to `imax` attending to the key `j`
@inline attention_window_last(window::Nothing, imax, j, off) = imax
@inline attention_window_last(window, imax, j, off) = min(imax, j - off + window - 1)
# the keys a query attends to at most
@inline attention_window_keys(window::Nothing, len) = len
@inline attention_window_keys(window, len) = min(window, len)
# The first position of the sequence `s` along the sequence dimension of an array of `n`
# positions, and how many it has; and the batch element of the arrays it is in.
@inline attention_span(ends::Nothing, s, n) = (0, n)
@inline function attention_span(ends, s, n)
    first = s == 1 ? 0 : min(Int(@inbounds ends[s - 1]), n)
    return first, min(Int(@inbounds ends[s]), n) - first
end
@inline attention_batch(ends::Nothing, s) = s
@inline attention_batch(ends, s) = 1
# A packed sequence shorter than the longest has blocks with no position to go through.
@inline attention_live(ends::Nothing, i0, len, jmax) = jmax
@inline attention_live(ends, i0, len, jmax) = ifelse(i0 < len, jmax, 0)

# The sequence `s` of an array as an array of its own, along the dimension of the
# positions: a view of a segment of packed arrays.
@inline attention_seq(x, ends::Nothing, s) = x
@inline attention_seq(x::Nothing, ends::AbstractVector, s) = nothing
@inline attention_seq(x::AbstractArray{<:Any,4}, ends::AbstractVector, s) = view(x, :, :, attention_range(ends, s), :)
@inline attention_seq(x::AbstractArray{<:Any,3}, ends::AbstractVector, s) = view(x, :, attention_range(ends, s), :)
@inline attention_range(ends, s) = ((s == 1 ? 0 : Int(ends[s - 1])) + 1):Int(ends[s])


#######################
# CPU
#######################

# The keys of a tile are packed transposed, so that the loops below run over contiguous
# keys, and two queries go through them together, sharing the loads.

# s[j] += Σ_e q[e] * kt[j, d+e-1]
@inline function attention_logits4!(s1, s2, kt, n, d, q1::NTuple{4,A}, q2::NTuple{4,A}) where {A}
    @inbounds @simd ivdep for j in 1:n
        k1, k2, k3, k4 = kt[j, d], kt[j, d + 1], kt[j, d + 2], kt[j, d + 3]
        s1[j] = muladd(q1[4], k4, muladd(q1[3], k3, muladd(q1[2], k2, muladd(q1[1], k1, s1[j]))))
        s2[j] = muladd(q2[4], k4, muladd(q2[3], k3, muladd(q2[2], k2, muladd(q2[1], k1, s2[j]))))
    end
end

@inline function attention_logits1!(s1, s2, kt, n, d, q1::A, q2::A) where {A}
    @inbounds @simd ivdep for j in 1:n
        s1[j] = muladd(q1, kt[j, d], s1[j])
        s2[j] = muladd(q2, kt[j, d], s2[j])
    end
end

# Σ_j p[j] * vt[j, d+e-1]
@inline function attention_dots4(p1, p2, vt, n, d)
    A = eltype(vt)
    a1 = a2 = a3 = a4 = b1 = b2 = b3 = b4 = zero(A)
    @inbounds @simd for j in 1:n
        x, y = p1[j], p2[j]
        v1, v2, v3, v4 = vt[j, d], vt[j, d + 1], vt[j, d + 2], vt[j, d + 3]
        a1 = muladd(x, v1, a1); a2 = muladd(x, v2, a2); a3 = muladd(x, v3, a3); a4 = muladd(x, v4, a4)
        b1 = muladd(y, v1, b1); b2 = muladd(y, v2, b2); b3 = muladd(y, v3, b3); b4 = muladd(y, v4, b4)
    end
    return (a1, a2, a3, a4), (b1, b2, b3, b4)
end

@inline function attention_dots1(p1, p2, vt, n, d)
    a = b = zero(eltype(vt))
    @inbounds @simd for j in 1:n
        a = muladd(p1[j], vt[j, d], a)
        b = muladd(p2[j], vt[j, d], b)
    end
    return a, b
end

# The keys are copied to a compact buffer first, and transposed from there: in `x` they
# are `head_dim * nheads` apart, often a multiple of the page size, so that walking them
# for each head dimension in turn evicts them from the cache as it goes.
function attention_pack!(xt::Matrix{A}, buf::Matrix{A}, x, h, j0, n, b) where {A}
    D = size(x, 1)
    @inbounds for j in 1:n
        @simd for d in 1:D
            buf[d, j] = A(x[d, h, j0 + j - 1, b])
        end
    end
    @inbounds for d in 1:D
        @simd for j in 1:n
            xt[j, d] = buf[d, j]
        end
    end
    return nothing
end

# The logits of the queries `i1` and `i2` against the `n` keys of a packed tile.
function attention_logits!(s1::Vector{A}, s2::Vector{A}, kt, n, q, h, i1, i2, b, scale::A) where {A}
    @inbounds @simd for j in 1:n
        s1[j] = zero(A)
        s2[j] = zero(A)
    end
    D = size(kt, 2)
    d = 1
    @inbounds while d + 3 <= D
        x = (A(q[d, h, i1, b]) * scale, A(q[d + 1, h, i1, b]) * scale,
             A(q[d + 2, h, i1, b]) * scale, A(q[d + 3, h, i1, b]) * scale)
        y = (A(q[d, h, i2, b]) * scale, A(q[d + 1, h, i2, b]) * scale,
             A(q[d + 2, h, i2, b]) * scale, A(q[d + 3, h, i2, b]) * scale)
        attention_logits4!(s1, s2, kt, n, d, x, y)
        d += 4
    end
    @inbounds while d <= D
        attention_logits1!(s1, s2, kt, n, d, A(q[d, h, i1, b]) * scale, A(q[d, h, i2, b]) * scale)
        d += 1
    end
    return nothing
end

@inline attention_bias!(s, n, bias::Nothing, j0, i, h, b) = nothing
@inline function attention_bias!(s::Vector{A}, n, bias::AbstractArray, j0, i, h, b) where {A}
    @inbounds for j in 1:n
        s[j] += A(attention_at(bias, j0 + j - 1, i, h, b))
    end
    return nothing
end

@inline attention_mask!(s, n, mask::Nothing, j0, i, h, b) = nothing
@inline function attention_mask!(s::Vector{A}, n, mask::AbstractArray, j0, i, h, b) where {A}
    @inbounds for j in 1:n
        s[j] = ifelse(attention_at(mask, j0 + j - 1, i, h, b), s[j], A(-Inf))
    end
    return nothing
end

# One step of the online softmax of the query `i`, held in the slot `ii` of the
# workspace: turns the logits `s[1:n]` of a tile of keys starting at `j0`, of which the
# query attends to those after the first `nskip` up to the `nvalid`-th, into weights
# relative to the running maximum `m[ii]`, and rescales what was accumulated under the
# previous one.
function attention_softmax!(s::Vector{A}, n, nskip, nvalid, bias, mask, j0, i, h, b, m, l, oacc, ii) where {A}
    attention_bias!(s, nvalid, bias, j0, i, h, b)
    attention_mask!(s, nvalid, mask, j0, i, h, b)
    @inbounds for j in (nvalid + 1):n
        s[j] = A(-Inf)
    end
    @inbounds for j in 1:min(nskip, n)
        s[j] = A(-Inf)
    end
    mt = A(-Inf)
    @inbounds @simd for j in 1:n
        mt = max(mt, s[j])
    end
    @inbounds begin
        mi = m[ii]
        if mt > mi
            a = _exp(mi - mt)
            l[ii] *= a
            @simd for d in axes(oacc, 1)
                oacc[d, ii] *= a
            end
            m[ii] = mi = mt
        end
        lt = zero(A)
        @simd for j in 1:n
            e = _exp(s[j] - mi)     # the `exp` of the activations, which vectorizes
            s[j] = e
            lt += e
        end
        l[ii] += lt
    end
    return nothing
end

# oacc[:, ii] += vt' * p1 and oacc[:, ii + 1] += vt' * p2
function attention_outputs!(oacc, ii, p1, p2, vt, n)
    Dv = size(vt, 2)
    d = 1
    @inbounds while d + 3 <= Dv
        a, c = attention_dots4(p1, p2, vt, n, d)
        oacc[d, ii] += a[1]; oacc[d + 1, ii] += a[2]; oacc[d + 2, ii] += a[3]; oacc[d + 3, ii] += a[4]
        oacc[d, ii + 1] += c[1]; oacc[d + 1, ii + 1] += c[2]; oacc[d + 2, ii + 1] += c[3]; oacc[d + 3, ii + 1] += c[4]
        d += 4
    end
    @inbounds while d <= Dv
        a, c = attention_dots1(p1, p2, vt, n, d)
        oacc[d, ii] += a
        oacc[d, ii + 1] += c
        d += 1
    end
    return nothing
end

# Keys per tile: a packed tile should stay in the L2 cache, and be long enough for the
# loops over it to be worth vectorizing.
attention_keys_per_tile(D::Int) = D <= 32 ? 256 : D <= 256 ? 128 : 64

function attention_workspace(::Type{A}, D::Int, Dv::Int, nq::Int) where {A}
    nk = attention_keys_per_tile(max(D, Dv))
    # one slot more than the queries: an unpaired query is paired with a copy of itself
    return (; kt = Matrix{A}(undef, nk, D), vt = Matrix{A}(undef, nk, Dv),
              buf = Matrix{A}(undef, max(D, Dv), nk),
              s1 = Vector{A}(undef, nk), s2 = Vector{A}(undef, nk),
              oacc = Matrix{A}(undef, Dv, nq + 1),
              m = Vector{A}(undef, nq + 1), l = Vector{A}(undef, nq + 1))
end

# The queries `i0:i1` of the head `h` of the batch element `b`, of which the first `qlen`
# are valid and attend to the first `kvlen` keys, up to `i + off` for the query `i` with
# a causal mask, from `i + off - window + 1` on within a band.
function attention_item!(o::AbstractArray{T,4}, lse, q, k, v, bias, mask, qlen, kvlen, off,
                         scale::A, is_causal::Bool, window, h, b, i0, i1, ws) where {T,A}
    (; kt, vt, buf, s1, s2, oacc, m, l) = ws
    hk = (h - 1) ÷ (size(q, 2) ÷ size(k, 2)) + 1
    jmax = is_causal ? min(kvlen, i1 + off) : kvlen
    nk = size(kt, 1)
    fill!(oacc, zero(A))
    fill!(m, -floatmax(A))
    fill!(l, zero(A))
    j0 = max(1, attention_window_first(window, i0, off))
    while j0 <= jmax
        nt = min(nk, jmax - j0 + 1)
        attention_pack!(kt, buf, k, hk, j0, nt, b)
        attention_pack!(vt, buf, v, hk, j0, nt, b)
        for i in i0:2:i1
            i2 = min(i + 1, i1)
            ii = i - i0 + 1
            # with a causal mask the earlier query of the pair attends to one key less
            n = is_causal ? min(nt, i2 + off - j0 + 1) : nt
            n <= 0 && continue
            n1 = is_causal ? min(nt, i + off - j0 + 1) : nt
            # within a band the later query of the pair skips one key more
            skip1, skip2 = attention_window_skip(window, i, off, j0), attention_window_skip(window, i2, off, j0)
            skip1 >= n1 && skip2 >= n && continue
            attention_logits!(s1, s2, kt, n, q, h, i, i2, b, scale)
            attention_softmax!(s1, n, skip1, n1, bias, mask, j0, i, h, b, m, l, oacc, ii)
            attention_softmax!(s2, n, skip2, n, bias, mask, j0, i2, h, b, m, l, oacc, ii + 1)
            attention_outputs!(oacc, ii, s1, s2, vt, n)
        end
        j0 += nt
    end
    @inbounds for i in i0:i1
        ii = i - i0 + 1
        if i <= qlen
            il = inv(l[ii])
            for d in axes(o, 1)
                o[d, h, i, b] = oacc[d, ii] * il
            end
        else
            for d in axes(o, 1)
                o[d, h, i, b] = zero(T)
            end
        end
        lse === nothing || (lse[h, i, b] = m[ii] + log(l[ii]))
    end
    return nothing
end

function fused_attention!(::CPU, o, lse, q, k, v, bias, mask, q_offset, q_lengths, kv_lengths, seqs,
                          scale::A, is_causal::Bool, window) where {A}
    H, Lq, B = size(q, 2), seqs.maxlen_q, seqs.n
    # a task for every 2^16 multiply-adds at least: spawning one costs a few microseconds
    work = Lq * attention_window_keys(window, seqs.maxlen_kv) * H * B * (size(q, 1) + size(v, 1))
    ntasks = should_use_spawn() ? min(Threads.nthreads(:default), cld(work, 2^16)) : 1
    # Queries per item: many, since each item packs every tile of keys it goes through,
    # but leaving several items to each task so that they balance the load.
    nq = clamp(2 * cld(Lq * H * B, 16 * ntasks), 32, 512)
    nqt = cld(Lq, nq)
    nitems = nqt * H * B
    next = Threads.Atomic{Int}(1)
    function run()
        ws = attention_workspace(A, size(q, 1), size(v, 1), nq)
        while true
            item = Threads.atomic_add!(next, 1)
            item > nitems && break
            # The sequences vary fastest: the items running together then write to
            # different pages of `o`, which they would otherwise all fault in together,
            # a row of `o` holding every head.
            s = (item - 1) % B + 1
            i0 = ((item - 1) ÷ B) % nqt * nq + 1
            h = (item - 1) ÷ (B * nqt) + 1
            qs, ks = attention_seq(q, seqs.q_ends, s), attention_seq(k, seqs.kv_ends, s)
            i0 > size(qs, 3) && continue
            attention_item!(attention_seq(o, seqs.q_ends, s), attention_seq(lse, seqs.q_ends, s), qs, ks,
                            attention_seq(v, seqs.kv_ends, s), bias, mask,
                            attention_len(q_lengths, s, size(qs, 3)), attention_len(kv_lengths, s, size(ks, 3)),
                            attention_offset(q_offset, s), scale, is_causal, window, h,
                            attention_batch(seqs.q_ends, s),
                            i0, min(i0 + nq - 1, size(qs, 3)), ws)
        end
        return nothing
    end
    if min(ntasks, nitems) <= 1
        run()
    else
        foreach(wait, [Threads.@spawn(run()) for _ in 1:min(ntasks, nitems)])
    end
    return o
end


#######################
# GPU
#######################

# A block of `T` threads owns `BQ` queries of one head of one batch element, and walks
# the keys in tiles of `BK`. For each tile:
#
#   1. logits  S = QᵀK   every thread accumulates the logits of `TM` queries × `TN` keys
#                         in registers, reading the operands from shared memory, where
#                         they are staged in slabs of `KD` head dimensions;
#   2. online softmax     the threads exchange the maxima of their rows through shared
#                         memory, and write the weights P to a shared tile;
#   3. output  O += P V   every thread accumulates `TD` head dimensions × `TQ` queries of
#                         the output in registers, for the whole kernel, reading P from
#                         the shared tile and V from slabs of `KV` keys.
#
# What makes it fast:
# - the registers of a thread hold a tile (8×8 where it fits), so that an FMA needs a
#   quarter of a load, and nothing of size `head_dim` is ever kept in them;
# - the rows and columns of a thread are interleaved with those of the others, and the
#   shared tiles padded, so that the threads of a warp fall in different banks;
# - the loads from global memory of a slab are issued together, and one slab ahead of
#   their use, so that their latency overlaps with the arithmetic on the current slab.
#
# The tiling is a `Val` of a named tuple `c`, which makes every size a compile-time
# constant. The closures below only capture arguments of the function they are written
# in: a captured variable that is reassigned would be boxed, and `inbounds=true` on the
# kernel does not reach them.

const ATTENTION_MAX_HEAD_DIM = 2048

# Bytes of workgroup-local memory the kernel may use on `backend`. The default is what
# every backend provides; an extension raises it where more is available.
attention_shmem_budget(backend) = 32 * 1024

function attention_tiling(T::Int, BQ::Int, BK::Int, Dv::Int; KD::Int=8, KV::Int=(BK > 64 ? 16 : 8))
    TQ = min(8, BQ)                 # output tile of a thread: TD head dimensions × TQ queries
    NQ = BQ ÷ TQ
    ND = T ÷ NQ
    TD = cld(Dv, ND)
    mn = BQ * BK ÷ T                # logits tile of a thread: TM queries × TN keys
    TM = min(8, BQ, mn)
    TN = mn ÷ TM
    NR, NC = BQ ÷ TM, BK ÷ TN
    RS = T ÷ KD
    PP = BQ + max(1, 32 ÷ NC)
    PQ = BQ + max(1, 32 ÷ KD)
    PK = BK + max(1, 32 ÷ KD)
    DVP = TD * ND
    return (; T, BQ, BK, TM, TN, TD, TQ, KD, KV, NR, NC, NQ, ND, RS, PP, PQ, PK, DVP,
              SLAB = max((PQ + PK) * KD, DVP * KV))
end

# floats of shared memory
attention_shmem(c) = c.PP * c.BK + c.SLAB + c.BQ

# The tiling for a value head dimension `Dv`, `nqueries` queries overall and `budget`
# floats of shared memory, or `nothing` if the kernel has none.
function attention_tiling(Dv::Int, nqueries::Int, budget::Int)
    Dv <= ATTENTION_MAX_HEAD_DIM || return nothing
    BQ = Dv <= 256 ? 64 : Dv <= 512 ? 32 : Dv <= 1024 ? 16 : 8
    # a small problem is split into more blocks, to occupy the device
    BQ == 64 && cld(nqueries, BQ) < 128 && (BQ = 32)
    T = Dv <= 64 ? 64 : Dv <= 128 ? 128 : 256
    if BQ == 64
        # 128 keys per tile where they fit: 8×8 logits per thread with twice the threads
        wide = attention_tiling(max(T, 128), BQ, 128, Dv)
        attention_shmem(wide) <= budget && return wide
    end
    # otherwise shrink until it fits: the slab of values first, then the tile of queries
    while true
        for KV in (8, 4, 2, 1)
            c = attention_tiling(T, BQ, 64, Dv; KV)
            attention_shmem(c) <= budget && return c
        end
        BQ > 8 || return nothing
        BQ ÷= 2
    end
end

attention_tiling(backend, q::AbstractArray{T}, v) where {T} =
    attention_tiling(size(v, 1), length(q) ÷ size(q, 1),
                     attention_shmem_budget(backend) ÷ sizeof(attention_compute_type(T)))

# acc[m, n] += a[m] * c[n], with `acc` stored column-major as a tuple
@inline @generated function attention_mma(acc::NTuple{MN,A}, a::NTuple{TM,A}, c::NTuple{TN,A}) where {MN,A,TM,TN}
    Expr(:tuple, [:(muladd(a[$m], c[$n], acc[$((n - 1) * TM + m)])) for n in 1:TN for m in 1:TM]...)
end

# maximum and sum over n of x[m, n]
@inline @generated function attention_rowmax(x::NTuple{MN,A}, ::Val{TM}) where {MN,A,TM}
    Expr(:tuple, [foldl((a, b) -> :(max($a, $b)), [:(x[$((n - 1) * TM + m)]) for n in 1:(MN ÷ TM)]) for m in 1:TM]...)
end
@inline @generated function attention_rowsum(x::NTuple{MN,A}, ::Val{TM}) where {MN,A,TM}
    Expr(:tuple, [foldl((a, b) -> :($a + $b), [:(x[$((n - 1) * TM + m)]) for n in 1:(MN ÷ TM)]) for m in 1:TM]...)
end

# x[e, m] * a[m]
@inline @generated function attention_colscale(x::NTuple{MN,A}, a::NTuple{TQ,A}) where {MN,A,TQ}
    TD = MN ÷ TQ
    Expr(:tuple, [:(x[$((m - 1) * TD + e)] * a[$m]) for m in 1:TQ for e in 1:TD]...)
end

@inline attention_allowed(m::Nothing) = true
@inline attention_allowed(m::Bool) = m
@inline attention_biased(s, b::Nothing) = s
@inline attention_biased(s::A, b) where {A} = s + A(b)

# The rows `r0 + p*RS` of the dimension `d` of a tile of `x` starting at `first`, into
# registers: zero from the row `Lx` on. The load is unconditional, at a clamped index.
@inline function attention_fetch_rows(x, scale::A, d, h, first, r0, b, D, Lx, ::Val{NB}, ::Val{RS}) where {A,NB,RS}
    ntuple(Val(cld(NB, RS))) do p
        r = r0 + (p - 1) * RS
        val = A(@inbounds x[min(d, D - 1) + 1, h, min(first + r, size(x, 3) - 1) + 1, b]) * scale
        ifelse((d < D) & (r < NB) & (first + r < Lx), val, zero(A))
    end
end

# ... and from there to `slab[off + row + P*dd]`
@inline function attention_put_rows!(slab, off, P, vals::NTuple{NP}, dd, r0, ::Val{NB}, ::Val{RS}) where {NP,NB,RS}
    ntuple(Val(NP)) do p
        r = r0 + (p - 1) * RS
        if r < NB
            @inbounds slab[off + r + P * dd + 1] = vals[p]
        end
        nothing
    end
    return nothing
end

# The dimensions `t + x*T` of the keys `j:j+KV-1` of `v`, into registers.
@inline function attention_fetch_dims(v, ::Type{A}, t, hk, j, b, jmax, ::Val{c}) where {A,c}
    Dv, Lv = size(v, 1), size(v, 3)
    ntuple(Val(c.KV * cld(c.DVP, c.T))) do x
        jj = (x - 1) % c.KV
        d = t + ((x - 1) ÷ c.KV) * c.T
        val = A(@inbounds v[min(d, Dv - 1) + 1, hk, min(j + jj, Lv - 1) + 1, b])
        ifelse((d < Dv) & (j + jj < jmax), val, zero(A))
    end
end

# ... and from there to `slab[d + DVP*jj]`
@inline function attention_put_dims!(slab, vals::NTuple{N}, t, ::Val{c}) where {N,c}
    ntuple(Val(N)) do x
        jj = (x - 1) % c.KV
        d = t + ((x - 1) ÷ c.KV) * c.T
        if d < c.DVP
            @inbounds slab[d + c.DVP * jj + 1] = vals[x]
        end
        nothing
    end
    return nothing
end

@inline function attention_logits_slab(acc, slab, tr, tc, ::Val{c}) where {c}
    for s in 0:(c.KD - 1)
        a = ntuple(m -> @inbounds(slab[tr + c.NR * (m - 1) + c.PQ * s + 1]), Val(c.TM))
        k = ntuple(n -> @inbounds(slab[c.PQ * c.KD + tc + c.NC * (n - 1) + c.PK * s + 1]), Val(c.TN))
        acc = attention_mma(acc, a, k)
    end
    return acc
end

@inline function attention_output_slab(oacc, slab, p_shm, ud, uq, jj0, ::Val{c}) where {c}
    for s in 0:(c.KV - 1)
        a = ntuple(e -> @inbounds(slab[ud + c.ND * (e - 1) + c.DVP * s + 1]), Val(c.TD))
        p = ntuple(m -> @inbounds(p_shm[uq + c.NQ * (m - 1) + c.PP * (jj0 + s) + 1]), Val(c.TQ))
        oacc = attention_mma(oacc, a, p)
    end
    return oacc
end

# the logits of a thread, with the bias added and `-Inf` at the keys not attended to
@inline function attention_masked(acc::NTuple{MN,A}, mask, bias, i0, j0, tr, tc, h, b, jmax, Lq, off, window,
                                  ::Val{causal}, ::Val{c}) where {MN,A,causal,c}
    ntuple(Val(MN)) do x
        i = i0 + tr + c.NR * ((x - 1) % c.TM)
        j = j0 + tc + c.NC * ((x - 1) ÷ c.TM)
        ok = (j < jmax) & (i < Lq) & (!causal | (j <= i + off)) & attention_in_window(window, j, i, off)
        ok && attention_allowed(attention_at(mask, j + 1, i + 1, h, b)) ?
            attention_biased(acc[x], attention_at(bias, j + 1, i + 1, h, b)) : A(-Inf)
    end
end

# One value for each row of a thread, in the first columns of the shared tile, where the
# other threads holding the row find it.
@inline function attention_put_rowstat!(p_shm, x::NTuple{TM}, tr, tc, ::Val{c}) where {TM,c}
    ntuple(Val(TM)) do m
        @inbounds p_shm[tr + c.NR * (m - 1) + c.PP * tc + 1] = x[m]
        nothing
    end
    return nothing
end

@inline function attention_rows_max(p_shm, mx::NTuple{TM,A}, tr, ::Val{c}) where {TM,A,c}
    ntuple(Val(TM)) do m
        r = tr + c.NR * (m - 1)
        x = mx[m]
        for cc in 0:(c.NC - 1)
            x = max(x, @inbounds p_shm[r + c.PP * cc + 1])
        end
        x
    end
end

@inline function attention_row_sum(p_shm, r, ::Type{A}, ::Val{c}) where {A,c}
    l = zero(A)
    for cc in 0:(c.NC - 1)
        l += @inbounds p_shm[r + c.PP * cc + 1]
    end
    return l
end

@inline attention_weights(sc::NTuple{MN,A}, mnew::NTuple{TM,A}) where {MN,A,TM} =
    ntuple(x -> exp(sc[x] - mnew[(x - 1) % TM + 1]), Val(MN))

@inline attention_rescale(mx::NTuple{TM,A}, mnew::NTuple{TM,A}) where {TM,A} =
    ntuple(m -> exp(mx[m] - mnew[m]), Val(TM))

@inline attention_fold(ls::NTuple{TM,A}, alpha::NTuple{TM,A}, psum::NTuple{TM,A}) where {TM,A} =
    ntuple(m -> muladd(ls[m], alpha[m], psum[m]), Val(TM))

@inline function attention_put_tile!(p_shm, pr::NTuple{MN}, tr, tc, ::Val{c}) where {MN,c}
    ntuple(Val(MN)) do x
        @inbounds p_shm[tr + c.NR * ((x - 1) % c.TM) + c.PP * (tc + c.NC * ((x - 1) ÷ c.TM)) + 1] = pr[x]
        nothing
    end
    return nothing
end

@inline function attention_put_stat!(stat, alpha::NTuple{TM}, tr, ::Val{c}) where {TM,c}
    ntuple(Val(TM)) do m
        @inbounds stat[tr + c.NR * (m - 1) + 1] = alpha[m]
        nothing
    end
    return nothing
end

@inline attention_get_stat(stat, uq, ::Val{c}) where {c} =
    ntuple(m -> @inbounds(stat[uq + c.NQ * (m - 1) + 1]), Val(c.TQ))

# the outputs of the queries `i0 + ...` of a sequence of `Lq` starting at `qb`
@inline function attention_write!(o, oacc::NTuple{MN,A}, p_shm, ud, uq, i0, qb, Lq, h, b, qlen, ::Val{c}) where {MN,A,c}
    Dv = size(o, 1)
    ntuple(Val(c.TQ)) do m
        @inline
        r = uq + c.NQ * (m - 1)
        i = i0 + r
        il = i < qlen ? inv(attention_row_sum(p_shm, r, A, Val(c))) : zero(A)
        ntuple(Val(c.TD)) do e
            @inline
            d = ud + c.ND * (e - 1)
            if (d < Dv) & (i < Lq)
                @inbounds o[d + 1, h, qb + i + 1, b] = i < qlen ? oacc[(m - 1) * c.TD + e] * il : zero(A)
            end
            nothing
        end
        nothing
    end
    return nothing
end

@inline attention_write_lse!(lse::Nothing, mx::NTuple, p_shm, tr, i0, qb, Lq, h, b, ::Val) = nothing
@inline function attention_write_lse!(lse, mx::NTuple{TM,A}, p_shm, tr, i0, qb, Lq, h, b, ::Val{c}) where {TM,A,c}
    ntuple(Val(TM)) do m
        @inline
        r = tr + c.NR * (m - 1)
        if i0 + r < Lq
            @inbounds lse[h, qb + i0 + r + 1, b] = mx[m] + log(attention_row_sum(p_shm, r, A, Val(c)))
        end
        nothing
    end
    return nothing
end

# The queries and the keys of the sequence `s` start at `qb` and `kb` in the batch element
# `b` of the arrays: in a block the positions `i0`, `j0` and their bounds count from there.
@kernel inbounds=true cpu=false unsafe_indices=true function attention_kernel!(
        o, lse, @Const(q), @Const(k), @Const(v), bias, mask, q_offset, window, q_lengths, kv_lengths,
        q_ends, kv_ends, scale::A, causal::Val, til::Val{c}) where {A,c}
    p_shm = @localmem A (c.PP * c.BK,)
    slab = @localmem A (c.SLAB,)
    stat = @localmem A (c.BQ,)

    t = @index(Local, Linear) - 1
    gq, h, s = @index(Group, NTuple)
    D = size(q, 1)
    qb, Lq = attention_span(q_ends, s, size(q, 3))
    kb, Lk = attention_span(kv_ends, s, size(k, 3))
    b = attention_batch(q_ends, s)
    hk = (h - 1) ÷ (size(q, 2) ÷ size(k, 2)) + 1
    i0 = (gq - 1) * c.BQ
    qlen = attention_len(q_lengths, s, Lq)
    off = attention_offset(q_offset, s)
    jmax = attention_live(q_ends, i0, Lq,
                          min(attention_len(kv_lengths, s, Lk), causal === Val(true) ? i0 + c.BQ + off : Lk))
    # within a band, from the tile of the first key of the first query on
    jfirst = (max(attention_window_first(window, i0, off), 0) ÷ c.BK) * c.BK

    tr, tc = t ÷ c.NC, t % c.NC       # logits tile: queries tr + NR*m, keys tc + NC*n
    ud, uq = t ÷ c.NQ, t % c.NQ       # output tile: dimensions ud + ND*e, queries uq + NQ*m
    dd, r0 = t % c.KD, t ÷ c.KD       # this thread's share of a slab

    mx = ntuple(_ -> -floatmax(A), Val(c.TM))
    ls = ntuple(_ -> zero(A), Val(c.TM))
    oacc = ntuple(_ -> zero(A), Val(c.TD * c.TQ))

    qv = attention_fetch_rows(q, scale, dd, h, qb + i0, r0, b, D, qb + Lq, Val(c.BQ), Val(c.RS))
    kv = attention_fetch_rows(k, one(A), dd, hk, kb + jfirst, r0, b, D, kb + jmax, Val(c.BK), Val(c.RS))
    vv = ntuple(_ -> zero(A), Val(c.KV * cld(c.DVP, c.T)))
    j0 = jfirst
    while j0 < jmax
        # ---- logits
        acc = ntuple(_ -> zero(A), Val(c.TM * c.TN))
        d0 = 0
        while d0 < D
            attention_put_rows!(slab, 0, c.PQ, qv, dd, r0, Val(c.BQ), Val(c.RS))
            attention_put_rows!(slab, c.PQ * c.KD, c.PK, kv, dd, r0, Val(c.BK), Val(c.RS))
            @synchronize()
            d0 += c.KD
            if d0 < D
                qv = attention_fetch_rows(q, scale, d0 + dd, h, qb + i0, r0, b, D, qb + Lq, Val(c.BQ), Val(c.RS))
                kv = attention_fetch_rows(k, one(A), d0 + dd, hk, kb + j0, r0, b, D, kb + jmax, Val(c.BK), Val(c.RS))
            else
                vv = attention_fetch_dims(v, A, t, hk, kb + j0, b, kb + jmax, til)
            end
            acc = attention_logits_slab(acc, slab, tr, tc, til)
            @synchronize()
        end

        # ---- online softmax
        sc = attention_masked(acc, mask, bias, i0, j0, tr, tc, h, b, jmax, Lq, off, window, causal, til)
        attention_put_rowstat!(p_shm, attention_rowmax(sc, Val(c.TM)), tr, tc, til)
        @synchronize()
        mnew = attention_rows_max(p_shm, mx, tr, til)
        alpha = attention_rescale(mx, mnew)
        pr = attention_weights(sc, mnew)
        ls = attention_fold(ls, alpha, attention_rowsum(pr, Val(c.TM)))
        mx = mnew
        @synchronize()
        attention_put_tile!(p_shm, pr, tr, tc, til)
        tc == 0 && attention_put_stat!(stat, alpha, tr, til)
        @synchronize()

        # ---- output
        oacc = attention_colscale(oacc, attention_get_stat(stat, uq, til))
        jj0 = 0
        while jj0 < c.BK
            attention_put_dims!(slab, vv, t, til)
            @synchronize()
            jj0 += c.KV
            if jj0 < c.BK
                vv = attention_fetch_dims(v, A, t, hk, kb + j0 + jj0, b, kb + jmax, til)
            elseif j0 + c.BK < jmax
                qv = attention_fetch_rows(q, scale, dd, h, qb + i0, r0, b, D, qb + Lq, Val(c.BQ), Val(c.RS))
                kv = attention_fetch_rows(k, one(A), dd, hk, kb + j0 + c.BK, r0, b, D, kb + jmax, Val(c.BK), Val(c.RS))
            end
            oacc = attention_output_slab(oacc, slab, p_shm, ud, uq, jj0 - c.KV, til)
            @synchronize()
        end
        j0 += c.BK
    end

    # ---- normalization
    attention_put_rowstat!(p_shm, ls, tr, tc, til)
    @synchronize()
    attention_write!(o, oacc, p_shm, ud, uq, i0, qb, Lq, h, b, qlen, til)
    tc == 0 && attention_write_lse!(lse, mx, p_shm, tr, i0, qb, Lq, h, b, til)
end

function fused_attention!(backend::Backend, o, lse, q, k, v, bias, mask, q_offset, q_lengths, kv_lengths, seqs,
                          scale::A, is_causal::Bool, window) where {A}
    d = attention_decode_tiling(backend, q, k, v, seqs)
    d === nothing || return attention_decode!(backend, d, o, lse, q, k, v, bias, mask, q_offset, q_lengths,
                                              kv_lengths, seqs, scale, is_causal, window)
    c = attention_tiling(backend, q, v)
    attention_kernel!(backend, (c.T, 1, 1))(o, lse, q, k, v, bias, mask, q_offset, window, q_lengths, kv_lengths,
        seqs.q_ends, seqs.kv_ends, scale, Val(is_causal), Val(c);
        ndrange=(c.T * cld(seqs.maxlen_q, c.BQ), size(q, 2), seqs.n))
    return o
end


# Decoding: few queries per sequence, as with a key-value cache. The kernel above gives a
# block `BQ` queries of one head, most of them idle with one query, and the block walks
# all the keys of the sequence: only `nheads × batch_size` blocks, each reading every
# tile of keys and values once for each query head sharing it. Here, as in
# Flash-Decoding, the rows of a sequence are its queries for each of the heads sharing a
# key/value head, `r = g + G*i` for the head `g` of the group and the query `i`. A block
# owns `R` of them, and a share of the keys: a split. Each split writes its output
# normalized by its own sum of exponentials, and the logarithm of that sum, which a
# second kernel combines.
#
# For each tile of `BK` keys:
#
#   1. logits  every thread holds `TRL` rows × `TN` keys, reading the queries and a tile
#              of keys from shared memory;
#   2. online softmax, the maximum of each row taken over the `NC` threads holding it
#              through shared memory, the weights written to a shared tile;
#   3. output  every thread holds `TD` head dimensions × `TRO` rows, reading the values
#              straight from global memory, a warp reading consecutive head dimensions.
#
# The keys and the values of the next tile are loaded during the current one.

# Rows of a sequence at most. Beyond, the kernel above is as fast where it has enough
# blocks: its tiles of queries fill up, and those of keys are read from the L2 cache.
const ATTENTION_DECODE_ROWS = 64

function attention_decode_tiling(D::Int, Dv::Int, G::Int, R::Int)
    T = R >= 32 ? 256 : 128
    BK = 32
    NC = 32                         # logits: NRL row slots × NC key slots
    NRL = T ÷ NC
    ND = min(T, nextpow(2, max(Dv, 32)))   # output: ND dimension slots × NRO row slots
    NRO = T ÷ ND
    return (; T, D, Dv, G, R, BK, NC, NRL, TRL = R ÷ NRL, TN = BK ÷ NC, ND, NRO, TRO = R ÷ NRO,
              TD = cld(Dv, ND), NKL = cld(D * BK, T), PK = D + iseven(D))
end

# floats of shared memory: queries, keys, weights, row statistics
attention_decode_shmem(c) = c.D * c.R + c.PK * c.BK + c.R * c.BK + c.R * c.NC + c.R

# The tiling of the decoding kernel for these arrays, or `nothing` where it does not apply.
function attention_decode_tiling(backend, q::AbstractArray{T}, k, v, seqs) where {T}
    D, Dv, G = size(q, 1), size(v, 1), size(q, 2) ÷ size(k, 2)
    G * seqs.maxlen_q <= ATTENTION_DECODE_ROWS || return nothing
    Dv <= 256 || return nothing
    budget = attention_shmem_budget(backend) ÷ sizeof(attention_compute_type(T))
    R = clamp(nextpow(2, G * seqs.maxlen_q), 4, 32)
    while R >= 4
        c = attention_decode_tiling(D, Dv, G, R)
        attention_decode_shmem(c) <= budget && return c
        R ÷= 2
    end
    return nothing
end

# Streaming multiprocessors (CUDA) or compute units of the device: an extension sets it.
attention_compute_units(backend) = 64

# Splits of the keys, where the blocks are fewer than the multiprocessors: a block for
# each multiprocessor at least, and up to eight while the splits stay long.
function attention_splits(backend, c, nblocks::Int, nkeys::Int)
    nsm = attention_compute_units(backend)
    nblocks >= nsm && return 1
    ntiles = cld(nkeys, c.BK)
    return max(1, min(cld(nsm, nblocks), ntiles ÷ 4), min(cld(8 * nsm, nblocks), ntiles ÷ 16))
end

function attention_decode!(backend::Backend, c, o, lse, q, k, v, bias, mask, q_offset, q_lengths, kv_lengths, seqs,
                           scale::A, is_causal::Bool, window) where {A}
    # the keys of a sequence the queries attend to at most
    nkeys = min(seqs.maxlen_kv, attention_window_keys(window, seqs.maxlen_kv) + seqs.maxlen_q - 1)
    nrb = cld(size(q, 2) ÷ size(k, 2) * seqs.maxlen_q, c.R)
    nsplit = attention_splits(backend, c, nrb * size(k, 2) * seqs.n, nkeys)
    if nsplit == 1
        oo, ll = o, lse
    else
        oo = similar(o, A, size(o, 1), size(o, 2), size(o, 3), nsplit, size(o, 4))
        ll = similar(o, A, size(o, 2), size(o, 3), nsplit, size(o, 4))
    end
    attention_decode_kernel!(backend, (c.T, 1, 1))(oo, ll, q, k, v, bias, mask, q_offset, window, q_lengths,
        kv_lengths, seqs.q_ends, seqs.kv_ends, scale, nsplit, Val(is_causal), Val(c);
        ndrange=(c.T * nsplit * nrb, size(k, 2), seqs.n))
    nsplit == 1 && return o
    attention_combine_kernel!(backend)(o, lse, oo, ll, q_lengths, seqs.q_ends, Val(8);
                                       ndrange=(size(o, 1) * size(o, 2), seqs.maxlen_q, seqs.n))
    return o
end

# the query and the head of the row `r` of the key/value head `hk`
@inline attention_decode_row(r, hk, ::Val{c}) where {c} = r ÷ c.G, (hk - 1) * c.G + r % c.G + 1

# The scaled queries of the rows `r0 + r` as `qs[d + D*r]`, zero from the row `nrows` on.
@inline function attention_decode_queries!(qs, q, scale::A, hk, qb, b, r0, nrows, t, til::Val{c}) where {A,c}
    ntuple(Val(cld(c.D * c.R, c.T))) do x
        @inline
        e = t + (x - 1) * c.T
        d, r = e % c.D, e ÷ c.D
        if r < c.R
            i, h = attention_decode_row(r0 + r, hk, til)
            val = A(@inbounds q[d + 1, h, qb + min(i, size(q, 3) - 1 - qb) + 1, b]) * scale
            @inbounds qs[e + 1] = ifelse(r0 + r < nrows, val, zero(A))
        end
        nothing
    end
    return nothing
end

# The tile of the keys `j:j+BK-1` (zero from `jlim` on), in registers, with the head
# dimensions of a key in consecutive threads ...
@inline function attention_decode_fetch_keys(k, ::Type{A}, hk, j, b, jlim, t, ::Val{c}) where {A,c}
    ntuple(Val(c.NKL)) do x
        e = t + (x - 1) * c.T
        d, jj = e % c.D, e ÷ c.D
        val = A(@inbounds k[d + 1, hk, min(j + jj, size(k, 3) - 1) + 1, b])
        ifelse(j + jj < jlim, val, zero(A))
    end
end

# ... and from there to `ks[d + PK*jj]`
@inline function attention_decode_put_keys!(ks, vals::NTuple{N}, t, ::Val{c}) where {N,c}
    ntuple(Val(N)) do x
        e = t + (x - 1) * c.T
        d, jj = e % c.D, e ÷ c.D
        if jj < c.BK
            @inbounds ks[d + c.PK * jj + 1] = vals[x]
        end
        nothing
    end
    return nothing
end

# The dimensions `ud + ND*e` of the values of the keys `j:j+BK-1`, as `vals[e + TD*jj]`.
@inline function attention_decode_fetch_values(v, ::Type{A}, hk, j, b, jlim, ud, ::Val{c}) where {A,c}
    ntuple(Val(c.BK * c.TD)) do x
        jj, e = (x - 1) ÷ c.TD, (x - 1) % c.TD
        d = ud + c.ND * e
        val = A(@inbounds v[min(d, c.Dv - 1) + 1, hk, min(j + jj, size(v, 3) - 1) + 1, b])
        ifelse((d < c.Dv) & (j + jj < jlim), val, zero(A))
    end
end

@inline function attention_decode_logits(qs, ks, tr, kc, ::Type{A}, ::Val{c}) where {A,c}
    acc = ntuple(_ -> zero(A), Val(c.TRL * c.TN))
    for d in 0:(c.D - 1)
        a = ntuple(m -> @inbounds(qs[d + c.D * (tr + c.NRL * (m - 1)) + 1]), Val(c.TRL))
        kk = ntuple(n -> @inbounds(ks[d + c.PK * (kc + c.NC * (n - 1)) + 1]), Val(c.TN))
        acc = attention_mma(acc, a, kk)
    end
    return acc
end

# the logits of a thread, with the bias added and `-Inf` at the keys not attended to
@inline function attention_decode_masked(acc::NTuple{MN,A}, mask, bias, j0, tr, kc, hk, b, jend, r0, nrows, off,
                                         window, ::Val{causal}, til::Val{c}) where {MN,A,causal,c}
    ntuple(Val(MN)) do x
        r = r0 + tr + c.NRL * ((x - 1) % c.TRL)
        j = j0 + kc + c.NC * ((x - 1) ÷ c.TRL)
        i, h = attention_decode_row(r, hk, til)
        ok = (j < jend) & (r < nrows) & (!causal | (j <= i + off)) & attention_in_window(window, j, i, off)
        ok && attention_allowed(attention_at(mask, j + 1, i + 1, h, b)) ?
            attention_biased(acc[x], attention_at(bias, j + 1, i + 1, h, b)) : A(-Inf)
    end
end

# one value for each row of a thread, as `pm[kc + NC*r]`
@inline function attention_decode_put_rowstat!(pm, x::NTuple{TRL}, tr, kc, ::Val{c}) where {TRL,c}
    ntuple(Val(TRL)) do m
        @inbounds pm[kc + c.NC * (tr + c.NRL * (m - 1)) + 1] = x[m]
        nothing
    end
    return nothing
end

@inline function attention_decode_rows_max(pm, mx::NTuple{TRL,A}, tr, ::Val{c}) where {TRL,A,c}
    ntuple(Val(TRL)) do m
        r = tr + c.NRL * (m - 1)
        x = mx[m]
        for cc in 0:(c.NC - 1)
            x = max(x, @inbounds pm[cc + c.NC * r + 1])
        end
        x
    end
end

@inline function attention_decode_put_weights!(pw, pr::NTuple{MN}, tr, kc, ::Val{c}) where {MN,c}
    ntuple(Val(MN)) do x
        r = tr + c.NRL * ((x - 1) % c.TRL)
        jj = kc + c.NC * ((x - 1) ÷ c.TRL)
        @inbounds pw[jj + c.BK * r + 1] = pr[x]
        nothing
    end
    return nothing
end

@inline function attention_decode_put_stat!(st, x::NTuple{TRL}, tr, ::Val{c}) where {TRL,c}
    ntuple(Val(TRL)) do m
        @inbounds st[tr + c.NRL * (m - 1) + 1] = x[m]
        nothing
    end
    return nothing
end

@inline attention_decode_get_stat(st, uo, ::Val{c}) where {c} =
    ntuple(m -> @inbounds(st[uo + c.NRO * (m - 1) + 1]), Val(c.TRO))

# oacc[e, m] += Σ_jj vals[e, jj] * pw[jj, uo + NRO*m], unrolled: `vals` is in registers
@generated function attention_decode_output(oacc::NTuple{MN,A}, pw, vals::NTuple{N,A}, uo,
                                                    ::Val{c}) where {MN,A,N,c}
    body = Any[Expr(:meta, :inline), Expr(:inbounds, true)]
    for jj in 0:(c.BK - 1)
        p = Expr(:tuple, [:(pw[$(jj + 1) + $(c.BK) * (uo + $(c.NRO * (m - 1)))]) for m in 1:c.TRO]...)
        a = Expr(:tuple, [:(vals[$(jj * c.TD + e)]) for e in 1:c.TD]...)
        push!(body, :(oacc = attention_mma(oacc, $a, $p)))
    end
    push!(body, Expr(:inbounds, :pop), :(return oacc))
    return Expr(:block, body...)
end

@inline function attention_decode_row_sum(pm, r, ::Type{A}, ::Val{c}) where {A,c}
    l = zero(A)
    for cc in 0:(c.NC - 1)
        l += @inbounds pm[cc + c.NC * r + 1]
    end
    return l
end

# The output and the `lse` of the rows of a thread: in the arrays of the outputs, or with
# a split, in those of the split, normalized by its own sum.
@inline attention_decode_store!(o::AbstractArray{<:Any,4}, d, h, pos, sp, b, x, i, qlen) =
    @inbounds o[d, h, pos, b] = ifelse(i < qlen, x, zero(x))
@inline attention_decode_store!(o::AbstractArray{<:Any,5}, d, h, pos, sp, b, x, i, qlen) =
    @inbounds o[d, h, pos, sp, b] = x
@inline attention_decode_store_lse!(lse::Nothing, h, pos, sp, b, m, l) = nothing
@inline attention_decode_store_lse!(lse::AbstractArray{<:Any,3}, h, pos, sp, b, m, l) =
    @inbounds lse[h, pos, b] = m + log(l)
@inline attention_decode_store_lse!(lse::AbstractArray{<:Any,4}, h, pos, sp, b, m::A, l) where {A} =
    @inbounds lse[h, pos, sp, b] = l > 0 ? m + log(l) : A(-Inf)
@inline attention_decode_scale(o::AbstractArray{<:Any,4}, l) = inv(l)
@inline attention_decode_scale(o::AbstractArray{<:Any,5}, l) = l > 0 ? inv(l) : zero(l)

@inline function attention_decode_write!(o, lse, oacc::NTuple{MN,A}, pm, st, ud, uo, hk, qb, sp, b, r0, nrows,
                                         qlen, til::Val{c}) where {MN,A,c}
    ntuple(Val(c.TRO)) do m
        @inline
        r = uo + c.NRO * (m - 1)
        i, h = attention_decode_row(r0 + r, hk, til)
        if r0 + r < nrows
            l = attention_decode_row_sum(pm, r, A, Val(c))
            il = attention_decode_scale(o, l)
            ntuple(Val(c.TD)) do e
                @inline
                d = ud + c.ND * (e - 1)
                d < c.Dv && attention_decode_store!(o, d + 1, h, qb + i + 1, sp, b, oacc[(m - 1) * c.TD + e] * il,
                                                    i, qlen)
                nothing
            end
            ud == 0 && attention_decode_store_lse!(lse, h, qb + i + 1, sp, b, @inbounds(st[r + 1]), l)
        end
        nothing
    end
    return nothing
end

@kernel inbounds=true cpu=false unsafe_indices=true function attention_decode_kernel!(
        o, lse, @Const(q), @Const(k), @Const(v), bias, mask, q_offset, window, q_lengths, kv_lengths,
        q_ends, kv_ends, scale::A, nsplit, causal::Val, til::Val{c}) where {A,c}
    qs = @localmem A (c.D * c.R,)
    ks = @localmem A (c.PK * c.BK,)
    pw = @localmem A (c.R * c.BK,)
    pm = @localmem A (c.R * c.NC,)
    st = @localmem A (c.R,)

    t = @index(Local, Linear) - 1
    gx, hk, s = @index(Group, NTuple)
    sp, r0 = (gx - 1) % nsplit + 1, (gx - 1) ÷ nsplit * c.R
    qb, Lq = attention_span(q_ends, s, size(q, 3))
    kb, Lk = attention_span(kv_ends, s, size(k, 3))
    b = attention_batch(q_ends, s)
    qlen = attention_len(q_lengths, s, Lq)
    off = attention_offset(q_offset, s)
    # the rows of the sequence, and the queries of those of the block
    nrows = c.G * Lq
    i0, i1 = r0 ÷ c.G, (min(r0 + c.R, nrows) - 1) ÷ c.G
    jmax = ifelse(r0 < nrows, min(attention_len(kv_lengths, s, Lk), causal === Val(true) ? i1 + off + 1 : Lk), 0)
    jfirst = max(attention_window_first(window, i0, off), 0)
    # the keys of the split, in whole tiles
    chunk = cld(cld(max(jmax - jfirst, 0), nsplit), c.BK) * c.BK
    jstart = jfirst + (sp - 1) * chunk
    jend = min(jstart + chunk, jmax)

    tr, kc = t ÷ c.NC, t % c.NC       # logits: rows tr + NRL*m, keys kc + NC*n
    ud, uo = t % c.ND, t ÷ c.ND       # output: dimensions ud + ND*e, rows uo + NRO*m

    mx = ntuple(_ -> -floatmax(A), Val(c.TRL))
    ls = ntuple(_ -> zero(A), Val(c.TRL))
    oacc = ntuple(_ -> zero(A), Val(c.TD * c.TRO))

    attention_decode_queries!(qs, q, scale, hk, qb, b, r0, nrows, t, til)
    kr = attention_decode_fetch_keys(k, A, hk, kb + jstart, b, kb + jend, t, til)
    vr = attention_decode_fetch_values(v, A, hk, kb + jstart, b, kb + jend, ud, til)
    attention_decode_put_keys!(ks, kr, t, til)
    @synchronize()
    j0 = jstart
    while j0 < jend
        # ---- logits
        acc = attention_decode_logits(qs, ks, tr, kc, A, til)
        sc = attention_decode_masked(acc, mask, bias, j0, tr, kc, hk, b, jend, r0, nrows, off, window, causal, til)
        attention_decode_put_rowstat!(pm, attention_rowmax(sc, Val(c.TRL)), tr, kc, til)
        @synchronize()
        if j0 + c.BK < jend
            kr = attention_decode_fetch_keys(k, A, hk, kb + j0 + c.BK, b, kb + jend, t, til)
        end

        # ---- online softmax
        mnew = attention_decode_rows_max(pm, mx, tr, til)
        alpha = attention_rescale(mx, mnew)
        pr = attention_weights(sc, mnew)
        ls = attention_fold(ls, alpha, attention_rowsum(pr, Val(c.TRL)))
        mx = mnew
        attention_decode_put_weights!(pw, pr, tr, kc, til)
        kc == 0 && attention_decode_put_stat!(st, alpha, tr, til)
        @synchronize()

        # ---- output
        oacc = attention_colscale(oacc, attention_decode_get_stat(st, uo, til))
        oacc = attention_decode_output(oacc, pw, vr, uo, til)
        if j0 + c.BK < jend
            attention_decode_put_keys!(ks, kr, t, til)
            vr = attention_decode_fetch_values(v, A, hk, kb + j0 + c.BK, b, kb + jend, ud, til)
        end
        @synchronize()
        j0 += c.BK
    end

    # ---- normalization
    attention_decode_put_rowstat!(pm, ls, tr, kc, til)
    kc == 0 && attention_decode_put_stat!(st, mx, tr, til)
    @synchronize()
    attention_decode_write!(o, lse, oacc, pm, st, ud, uo, hk, qb, sp, b, r0, nrows, qlen, til)
end

# The output and the `lse` of a query from those of the splits, taken `NS` at a time with
# an online maximum, so that their loads are issued together. A query attending to no key
# gets `0 / 0`, and an `lse` of `-Inf`.
@kernel inbounds=true cpu=false function attention_combine_kernel!(o, lse, @Const(os), @Const(ls), q_lengths, q_ends,
                                                                  ::Val{NS}) where {NS}
    dh, i, s = @index(Global, NTuple)
    Dv = size(o, 1)
    d, h = (dh - 1) % Dv + 1, (dh - 1) ÷ Dv + 1
    qb, Lq = attention_span(q_ends, s, size(o, 3))
    b = attention_batch(q_ends, s)
    pos = qb + i
    if i <= Lq
        A = eltype(os)
        n = size(ls, 3)
        m, x, l = -floatmax(A), zero(A), zero(A)
        for sp0 in 0:NS:(n - 1)
            ws = ntuple(u -> sp0 + u <= n ? ls[h, pos, min(sp0 + u, n), b] : A(-Inf), Val(NS))
            xs = ntuple(u -> os[d, h, pos, min(sp0 + u, n), b], Val(NS))
            m, x, l = attention_combine_step(m, x, l, ws, xs)
        end
        o[d, h, pos, b] = i <= attention_len(q_lengths, s, Lq) ? x / l : zero(A)
        d == 1 && attention_combine_lse!(lse, h, pos, b, m, l)
    end
end

@inline function attention_combine_step(m::A, x::A, l::A, ws::NTuple{N,A}, xs::NTuple{N,A}) where {N,A}
    mnew = max(m, maximum(ws))
    a = exp(m - mnew)
    es = ntuple(u -> exp(ws[u] - mnew), Val(N))
    return mnew, muladd(x, a, sum(ntuple(u -> es[u] * xs[u], Val(N)))), muladd(l, a, sum(es))
end

@inline attention_combine_lse!(lse::Nothing, h, pos, b, m, l) = nothing
@inline attention_combine_lse!(lse, h, pos, b, m, l) = @inbounds lse[h, pos, b] = m + log(l)
