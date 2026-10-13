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
    return all(x -> x === nothing || get_backend(x) == backend, (k, v, bias, mask)) &&
           size(v, 1) <= ATTENTION_MAX_HEAD_DIM
end

fused_attention_arrays(q, k, v, bias, mask) = false

# A `bias` or `mask` the kernels can index: every dimension is full or a singleton.
attention_broadcasts(x::Nothing, sz) = true
attention_broadcasts(x::AbstractArray, sz) =
    ndims(x) <= 4 && all(d -> size(x, d) == 1 || size(x, d) == sz[d], 1:4)

# The output of `scaled_dot_product_attention` for inputs satisfying
# `fused_attention_applies`, in the layout of `_scaled_dot_product_attention`.
function fused_attention(q::AA4{T}, k::AA4{T}, v::AA4{T}, bias, mask, scale, is_causal::Bool,
                         q_lengths, kv_lengths) where {T}
    A = attention_compute_type(T)
    s = scale === nothing ? inv(sqrt(A(size(q, 1)))) : A(scale)
    o = similar(q, size(v, 1), size(q, 2), size(q, 3), size(q, 4))
    fused_attention!(get_backend(q), o, q, k, v, attention_pad4(bias), attention_pad4(mask),
                     attention_vec(q_lengths), attention_vec(kv_lengths), s, is_causal)
    return o
end

attention_pad4(x::Nothing) = nothing
attention_pad4(x::AbstractArray) = reshape(x, size(x, 1), size(x, 2), size(x, 3), size(x, 4))
attention_vec(x::Nothing) = nothing
attention_vec(x::AbstractArray) = vec(x)

# The entry of a `bias` or `mask` for the key `j` and the query `i`.
@inline attention_sel(x, n) = ifelse(n == 1, 1, x)
@inline attention_at(a::Nothing, j, i, h, b) = nothing
@inline attention_at(a, j, i, h, b) = @inbounds a[attention_sel(j, size(a, 1)), attention_sel(i, size(a, 2)),
                                                  attention_sel(h, size(a, 3)), attention_sel(b, size(a, 4))]
@inline attention_len(l::Nothing, b, n) = n
@inline attention_len(l, b, n) = Int(@inbounds l[b])


#######################
# CPU
#######################

# `exp` for arguments in `[attention_exp_floor, 0]`, without the branches that keep
# `Base.exp` from vectorizing. The relative error is below 2 ulp.
@inline function attention_exp(x::Float32)
    n = round(x * 1.442695f0)
    r = muladd(n, -0.693359375f0, x)
    r = muladd(n, 2.12194440f-4, r)
    p = evalpoly(r, (5.0000001201f-1, 1.6666665459f-1, 4.1665795894f-2, 8.3334519073f-3,
                     1.3981999507f-3, 1.9875691500f-4))
    y = muladd(p * r, r, r) + 1f0
    return y * reinterpret(Float32, (unsafe_trunc(Int32, n) + Int32(127)) << Int32(23))
end

@inline function attention_exp(x::Float64)
    n = round(x * 1.4426950408889634)
    r = muladd(n, -0.6931471803691238, x)
    r = muladd(n, -1.9082149292705877e-10, r)
    y = evalpoly(r, (1.0, 1.0, 0.5, 0.16666666666666666, 0.041666666666666664,
                     0.008333333333333333, 0.001388888888888889, 0.0001984126984126984,
                     2.48015873015873e-5, 2.7557319223985893e-6, 2.755731922398589e-7,
                     2.505210838544172e-8, 2.08767569878681e-9))
    return y * reinterpret(Float64, (unsafe_trunc(Int64, n) + 1023) << 52)
end

attention_exp_floor(::Type{Float32}) = -87f0
attention_exp_floor(::Type{Float64}) = -708.0

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

function attention_pack!(xt::Matrix{A}, x, h, j0, n, b) where {A}
    @inbounds for d in axes(x, 1)
        @simd for j in 1:n
            xt[j, d] = A(x[d, h, j0 + j - 1, b])
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
# query attends to the first `nvalid`, into weights relative to the running maximum
# `m[ii]`, and rescales what was accumulated under the previous one.
function attention_softmax!(s::Vector{A}, n, nvalid, bias, mask, j0, i, h, b, m, l, oacc, ii) where {A}
    attention_bias!(s, nvalid, bias, j0, i, h, b)
    attention_mask!(s, nvalid, mask, j0, i, h, b)
    @inbounds for j in (nvalid + 1):n
        s[j] = A(-Inf)
    end
    mt = A(-Inf)
    @inbounds @simd for j in 1:n
        mt = max(mt, s[j])
    end
    @inbounds begin
        mi = m[ii]
        if mt > mi
            a = exp(mi - mt)
            l[ii] *= a
            @simd for d in axes(oacc, 1)
                oacc[d, ii] *= a
            end
            m[ii] = mi = mt
        end
        low = attention_exp_floor(A)
        lt = zero(A)
        @simd for j in 1:n
            x = s[j] - mi
            e = ifelse(x < low, zero(A), attention_exp(max(x, low)))
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
              s1 = Vector{A}(undef, nk), s2 = Vector{A}(undef, nk),
              oacc = Matrix{A}(undef, Dv, nq + 1),
              m = Vector{A}(undef, nq + 1), l = Vector{A}(undef, nq + 1))
end

# The queries `i0:i1` of the head `h` of the batch element `b`.
function attention_item!(o::AbstractArray{T,4}, q, k, v, bias, mask, q_lengths, kv_lengths,
                         scale::A, is_causal::Bool, h, b, i0, i1, ws) where {T,A}
    (; kt, vt, s1, s2, oacc, m, l) = ws
    hk = (h - 1) ÷ (size(q, 2) ÷ size(k, 2)) + 1
    qlen = attention_len(q_lengths, b, size(q, 3))
    kvlen = attention_len(kv_lengths, b, size(k, 3))
    jmax = is_causal ? min(kvlen, i1) : kvlen
    nk = size(kt, 1)
    fill!(oacc, zero(A))
    fill!(m, -floatmax(A))
    fill!(l, zero(A))
    j0 = 1
    while j0 <= jmax
        nt = min(nk, jmax - j0 + 1)
        attention_pack!(kt, k, hk, j0, nt, b)
        attention_pack!(vt, v, hk, j0, nt, b)
        for i in i0:2:i1
            i2 = min(i + 1, i1)
            ii = i - i0 + 1
            # with a causal mask the earlier query of the pair attends to one key less
            n = is_causal ? min(nt, i2 - j0 + 1) : nt
            n <= 0 && continue
            n1 = is_causal ? min(nt, i - j0 + 1) : nt
            attention_logits!(s1, s2, kt, n, q, h, i, i2, b, scale)
            attention_softmax!(s1, n, n1, bias, mask, j0, i, h, b, m, l, oacc, ii)
            attention_softmax!(s2, n, n, bias, mask, j0, i2, h, b, m, l, oacc, ii + 1)
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
    end
    return nothing
end

function fused_attention!(::CPU, o, q, k, v, bias, mask, q_lengths, kv_lengths, scale::A,
                          is_causal::Bool) where {A}
    H, Lq, B = size(q, 2), size(q, 3), size(q, 4)
    # a task for every 2^16 multiply-adds at least: spawning one costs a few microseconds
    work = Lq * size(k, 3) * H * B * (size(q, 1) + size(v, 1))
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
            i0 = ((item - 1) % nqt) * nq + 1
            h = ((item - 1) ÷ nqt) % H + 1
            b = (item - 1) ÷ (nqt * H) + 1
            attention_item!(o, q, k, v, bias, mask, q_lengths, kv_lengths, scale, is_causal,
                            h, b, i0, min(i0 + nq - 1, Lq), ws)
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
# floats of shared memory.
function attention_tiling(Dv::Int, nqueries::Int, budget::Int)
    BQ = Dv <= 256 ? 64 : Dv <= 512 ? 32 : Dv <= 1024 ? 16 : 8
    # a small problem is split into more blocks, to occupy the device
    BQ == 64 && cld(nqueries, BQ) < 128 && (BQ = 32)
    T = Dv <= 64 ? 64 : Dv <= 128 ? 128 : 256
    # 128 keys per tile where they fit: 8×8 logits per thread with twice the threads
    wide = attention_tiling(max(T, 128), BQ, 128, Dv)
    BQ == 64 && attention_shmem(wide) <= budget && return wide
    c = attention_tiling(T, BQ, 64, Dv)
    while attention_shmem(c) > budget && c.BQ > 8
        c = attention_tiling(T, c.BQ ÷ 2, 64, Dv)
    end
    return c
end

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
@inline function attention_masked(acc::NTuple{MN,A}, mask, bias, i0, j0, tr, tc, h, b, jmax, Lq,
                                  ::Val{causal}, ::Val{c}) where {MN,A,causal,c}
    ntuple(Val(MN)) do x
        i = i0 + tr + c.NR * ((x - 1) % c.TM)
        j = j0 + tc + c.NC * ((x - 1) ÷ c.TM)
        ok = (j < jmax) & (i < Lq) & (!causal | (j <= i))
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

@inline function attention_write!(o, oacc::NTuple{MN,A}, p_shm, ud, uq, i0, h, b, qlen, ::Val{c}) where {MN,A,c}
    Dv, Lq = size(o, 1), size(o, 3)
    ntuple(Val(c.TQ)) do m
        r = uq + c.NQ * (m - 1)
        i = i0 + r
        il = i < qlen ? inv(attention_row_sum(p_shm, r, A, Val(c))) : zero(A)
        ntuple(Val(c.TD)) do e
            d = ud + c.ND * (e - 1)
            if (d < Dv) & (i < Lq)
                @inbounds o[d + 1, h, i + 1, b] = i < qlen ? oacc[(m - 1) * c.TD + e] * il : zero(A)
            end
            nothing
        end
        nothing
    end
    return nothing
end

@kernel inbounds=true cpu=false unsafe_indices=true function attention_kernel!(
        o, @Const(q), @Const(k), @Const(v), bias, mask, q_lengths, kv_lengths,
        scale::A, causal::Val, til::Val{c}) where {A,c}
    p_shm = @localmem A (c.PP * c.BK,)
    slab = @localmem A (c.SLAB,)
    stat = @localmem A (c.BQ,)

    t = @index(Local, Linear) - 1
    gq, h, b = @index(Group, NTuple)
    D, Lq = size(q, 1), size(q, 3)
    hk = (h - 1) ÷ (size(q, 2) ÷ size(k, 2)) + 1
    i0 = (gq - 1) * c.BQ
    qlen = attention_len(q_lengths, b, Lq)
    jmax = min(attention_len(kv_lengths, b, size(k, 3)), causal === Val(true) ? i0 + c.BQ : size(k, 3))

    tr, tc = t ÷ c.NC, t % c.NC       # logits tile: queries tr + NR*m, keys tc + NC*n
    ud, uq = t ÷ c.NQ, t % c.NQ       # output tile: dimensions ud + ND*e, queries uq + NQ*m
    dd, r0 = t % c.KD, t ÷ c.KD       # this thread's share of a slab

    mx = ntuple(_ -> -floatmax(A), Val(c.TM))
    ls = ntuple(_ -> zero(A), Val(c.TM))
    oacc = ntuple(_ -> zero(A), Val(c.TD * c.TQ))

    qv = attention_fetch_rows(q, scale, dd, h, i0, r0, b, D, Lq, Val(c.BQ), Val(c.RS))
    kv = attention_fetch_rows(k, one(A), dd, hk, 0, r0, b, D, jmax, Val(c.BK), Val(c.RS))
    vv = ntuple(_ -> zero(A), Val(c.KV * cld(c.DVP, c.T)))
    j0 = 0
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
                qv = attention_fetch_rows(q, scale, d0 + dd, h, i0, r0, b, D, Lq, Val(c.BQ), Val(c.RS))
                kv = attention_fetch_rows(k, one(A), d0 + dd, hk, j0, r0, b, D, jmax, Val(c.BK), Val(c.RS))
            else
                vv = attention_fetch_dims(v, A, t, hk, j0, b, jmax, til)
            end
            acc = attention_logits_slab(acc, slab, tr, tc, til)
            @synchronize()
        end

        # ---- online softmax
        sc = attention_masked(acc, mask, bias, i0, j0, tr, tc, h, b, jmax, Lq, causal, til)
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
                vv = attention_fetch_dims(v, A, t, hk, j0 + jj0, b, jmax, til)
            elseif j0 + c.BK < jmax
                qv = attention_fetch_rows(q, scale, dd, h, i0, r0, b, D, Lq, Val(c.BQ), Val(c.RS))
                kv = attention_fetch_rows(k, one(A), dd, hk, j0 + c.BK, r0, b, D, jmax, Val(c.BK), Val(c.RS))
            end
            oacc = attention_output_slab(oacc, slab, p_shm, ud, uq, jj0 - c.KV, til)
            @synchronize()
        end
        j0 += c.BK
    end

    # ---- normalization
    attention_put_rowstat!(p_shm, ls, tr, tc, til)
    @synchronize()
    attention_write!(o, oacc, p_shm, ud, uq, i0, h, b, qlen, til)
end

function fused_attention!(backend::GPU, o, q, k, v, bias, mask, q_lengths, kv_lengths, scale::A,
                          is_causal::Bool) where {A}
    H, Lq, B = size(q, 2), size(q, 3), size(q, 4)
    c = attention_tiling(size(v, 1), Lq * H * B, attention_shmem_budget(backend) ÷ sizeof(A))
    attention_kernel!(backend, (c.T, 1, 1))(o, q, k, v, bias, mask, q_lengths, kv_lengths,
        scale, Val(is_causal), Val(c); ndrange=(c.T * cld(Lq, c.BQ), H, B))
    return o
end
