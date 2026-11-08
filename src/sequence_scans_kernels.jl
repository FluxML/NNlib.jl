# Kernels of the chunked scans for the GPU backends, and of their steps for decoding.
#
# As in mamba_ssm and flash-linear-attention, each kernel computes tiles of a product of
# the matrices of a chunk, for all the chunks of all the heads at once. The masks and the
# decays are applied to the operands as they are loaded, so that the only `(chunk_size,
# chunk_size)` matrices stored are the products shared by a group of heads, as `Cᵀ B`.
#
# A block of `T` threads computes a `BI × BJ` tile of outputs, each thread `TM × TN` of
# them, at the rows `TM * ti + m` and columns `TN * tj + n`: the products read them from
# shared memory four at a time, the pitches being multiples of 4. The operands are loaded
# in slabs of `BK` from a function `f(m, k)` of the row (or column) `m` and the index `k`
# summed over, zero outside their bounds, into shared memory as `slab[m + P * k]`.
# Consecutive threads load consecutive `m` or `k`, whichever is contiguous in memory.

function scan_tiling(BI::Int, BJ::Int, BK::Int, TM::Int; T::Int=256)
    NR = BI ÷ TM
    NC = T ÷ NR
    TN = BJ ÷ NC
    pad = 4
    return (; T, BI, BJ, BK, TM, TN, NR, NC, PI=BI + pad, PJ=BJ + pad)
end

@inline function scan_fetch(f, m0, k0, t, ::Val{BM}, ::Val{BK}, ::Val{T}, ::Val{mfast}) where {BM,BK,T,mfast}
    ntuple(Val(BM * BK ÷ T)) do q
        @inline
        e = t + (q - 1) * T
        m, k = mfast ? (e % BM, e ÷ BM) : (e ÷ BK, e % BK)
        f(m0 + m, k0 + k)
    end
end

@inline function scan_put!(slab, vals::NTuple{Q}, t, P, ::Val{BM}, ::Val{BK}, ::Val{T}, ::Val{mfast}) where {Q,BM,BK,T,mfast}
    ntuple(Val(Q)) do q
        @inline
        e = t + (q - 1) * T
        m, k = mfast ? (e % BM, e ÷ BM) : (e ÷ BK, e % BK)
        @inbounds slab[m + P * k + 1] = vals[q]
        nothing
    end
    return nothing
end

# An operand already in shared memory, the element `(m, k)` at `buf[off + m * SM + k * SK + 1]`,
# which has to be defined (zero past the bounds) for the slabs the loop reads.
struct ScanShared{SM,SK,B}
    buf::B
    off::Int
end

ScanShared(buf, off, ::Val{SM}, ::Val{SK}) where {SM,SK} = ScanShared{SM,SK,typeof(buf)}(buf, off)

@inline scan_stage(f, m0, k0, t, bm::Val, bk::Val, T::Val, mf::Val) = scan_fetch(f, m0, k0, t, bm, bk, T, mf)
@inline scan_stage(::ScanShared, m0, k0, t, ::Val, ::Val, ::Val, ::Val) = nothing
@inline scan_put!(slab, ::Nothing, t, P, ::Val, ::Val, ::Val, ::Val) = nothing

# where the products read the slab of an operand: (buffer, offset, stride of m, stride of k)
@inline scan_source(f, slab, P, m0, k0) = (slab, 0, 1, P)
@inline scan_source(f::ScanShared{SM,SK}, slab, P, m0, k0) where {SM,SK} = (f.buf, f.off + m0 * SM + k0 * SK, SM, SK)

# acc[m, n] += Σ_k a[TM*ti + m, k] * b[TN*tj + n, k] over `nk` columns of the slabs
@inline function scan_mma(acc, (ab, ao, am, ak), (bb, bo, bm, bk), ti, tj, ::Val{nk}, ::Val{c}) where {nk,c}
    for kk in 0:(nk - 1)
        a = ntuple(m -> @inbounds(ab[ao + (c.TM * ti + m - 1) * am + ak * kk + 1]), Val(c.TM))
        b = ntuple(n -> @inbounds(bb[bo + (c.TN * tj + n - 1) * bm + bk * kk + 1]), Val(c.TN))
        acc = attention_mma(acc, a, b)
    end
    return acc
end

# acc[m, n] += Σ_{kbeg ≤ k < kend} fa(i0 + TM*ti + m, k) * fb(j0 + TN*tj + n, k), each of
# `fa` and `fb` a function loaded slab by slab into `As` and `Bs`, or a `ScanShared`
@inline function scan_mainloop(acc, As, Bs, fa, fb, kbeg, kend, i0, j0, t, ti, tj,
                               ::Val{ma}, ::Val{mb}, ::Val{c}) where {ma,mb,c}
    k0 = kbeg
    k0 < kend || return acc
    va = scan_stage(fa, i0, k0, t, Val(c.BI), Val(c.BK), Val(c.T), Val(ma))
    vb = scan_stage(fb, j0, k0, t, Val(c.BJ), Val(c.BK), Val(c.T), Val(mb))
    while k0 < kend
        scan_put!(As, va, t, c.PI, Val(c.BI), Val(c.BK), Val(c.T), Val(ma))
        scan_put!(Bs, vb, t, c.PJ, Val(c.BJ), Val(c.BK), Val(c.T), Val(mb))
        @synchronize()
        sa, sb = scan_source(fa, As, c.PI, i0, k0), scan_source(fb, Bs, c.PJ, j0, k0)
        k0 += c.BK
        if k0 < kend
            va = scan_stage(fa, i0, k0, t, Val(c.BI), Val(c.BK), Val(c.T), Val(ma))
            vb = scan_stage(fb, j0, k0, t, Val(c.BJ), Val(c.BK), Val(c.T), Val(mb))
        end
        acc = scan_mma(acc, sa, sb, ti, tj, Val(c.BK), Val(c))
        @synchronize()
    end
    return acc
end

# the tiling of the kernels over the matrices of the chunks
scan_tiles() = scan_tiling(64, 64, 16, 4)

@inline scan_zeros(::Type{A}, ::Val{c}) where {A,c} = ntuple(_ -> zero(A), Val(c.TM * c.TN))

# f(i, j, acc[m, n]) at the rows i = i0 + TM*ti + m < ni and the columns j = j0 + TN*tj + n < nj
@inline function scan_foreach(f, acc, i0, j0, ti, tj, ni, nj, ::Val{c}) where {c}
    ntuple(Val(c.TM * c.TN)) do x
        @inline
        i = i0 + c.TM * ti + (x - 1) % c.TM
        j = j0 + c.TN * tj + (x - 1) ÷ c.TM
        (i < ni) & (j < nj) && f(i, j, acc[x])
        nothing
    end
    return nothing
end

@inline scan_scale(f, acc::NTuple{MN,A}, i0, j0, ti, tj, ::Val{c}) where {MN,A,c} =
    ntuple(Val(MN)) do x
        @inline
        f(i0 + c.TM * ti + (x - 1) % c.TM, j0 + c.TN * tj + (x - 1) ÷ c.TM, acc[x])
    end

# The element `(r, c)` of a matrix in the dense array `x`, at `x[o + r * sr + c * sc + 1]`,
# in the compute type `A`, zero unless `r < nr` and `c < nc`: the load is at a clamped index.
@inline function scan_ld(::Type{A}, x, o, r, c, sr, sc, nr, nc) where {A}
    v = A(@inbounds x[o + min(r, nr - 1) * sr + min(c, nc - 1) * sc + 1])
    return ifelse((r < nr) & (c < nc), v, zero(A))
end

# The offset of the token `t` (zero-based) of the head `hd` in `x`, of size
# (features, nheads, seq_len, batch_size), and the distance between tokens.
@inline tok_off(x, hd, t, b) = size(x, 1) * ((hd - 1) + size(x, 2) * (t + size(x, 3) * (b - 1)))
@inline tok_stride(x) = size(x, 1) * size(x, 2)

# The offset of the matrix of the chunk `ch` of the head `hd` in `x`, of size
# (rows, columns, chunks, nheads, batch_size).
@inline chunk_off(x, ch, hd, b) =
    size(x, 1) * size(x, 2) * ((ch - 1) + size(x, 3) * ((hd - 1) + size(x, 4) * (b - 1)))

# The tile `τ` (zero-based) of the lower triangle, row tiles first: (row tile, column tile).
@inline function scan_lower_tile(τ)
    r = 0
    while ((r + 1) * (r + 2)) ÷ 2 <= τ
        r += 1
    end
    return r, τ - (r * (r + 1)) ÷ 2
end

@inline scan_heads(hb, h) = ((hb - 1) % h + 1, (hb - 1) ÷ h + 1)

# Per position `s` of the chunk, into `w[s + 1]`: f(s) for s < L
@inline function scan_weights!(w, f, t, L, ::Val{T}) where {T}
    s = t
    while s < L
        @inbounds w[s + 1] = f(s)
        s += T
    end
    return nothing
end

@inline scan_at(w, k, L) = @inbounds w[min(k, L - 1) + 1]

# the longest chunk the kernels take
const L_MAX_SCAN = 256


#######################
# ssd_scan
#######################

# out[i, j, c, gg, b] = Σ_f X[f, gg, c0 + i, b] * Y[f, gg, c0 + j, b] for j ≤ i on tiles
# of the lower triangle: `Cᵀ B` of the chunks of a group of heads, or `kᵀ k`.
@kernel inbounds=true cpu=false unsafe_indices=true function scan_gram_kernel!(
        out::AbstractArray{A}, @Const(X), @Const(Y), L::Int, ::Val{c}) where {A,c}
    As = @localmem A (c.PI * c.BK,)
    Bs = @localmem A (c.PJ * c.BK,)
    t = @index(Local, Linear) - 1
    τ, ch, gb = @index(Group, NTuple)
    gg, b = scan_heads(gb, size(X, 2))
    ti, tj = t % c.NR, t ÷ c.NR
    it, jt = scan_lower_tile(τ - 1)
    i0, j0 = it * c.BI, jt * c.BJ
    c0 = (ch - 1) * L
    F, nv = size(X, 1), min(size(X, 3) - c0, L)
    ox, oy, sx = tok_off(X, gg, c0, b), tok_off(Y, gg, c0, b), tok_stride(X)
    fa = @inline (m, k) -> scan_ld(A, X, ox, k, m, 1, sx, F, nv)
    fb = @inline (m, k) -> scan_ld(A, Y, oy, k, m, 1, sx, F, nv)
    acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, F, i0, j0, t, ti, tj,
                        Val(false), Val(false), Val(c))
    oo = chunk_off(out, ch, gg, b)
    scan_foreach(acc, i0, j0, ti, tj, L, L, Val(c)) do i, j, v
        @inline
        @inbounds out[oo + i + L * j + 1] = v
    end
end

function scan_gram!(out, X, Y, L)
    c = scan_tiles()
    nt = cld(L, c.BI)
    scan_gram_kernel!(get_backend(out), c.T)(out, X, Y, L, Val(c);
        ndrange=(c.T * (nt * (nt + 1)) ÷ 2, size(out, 3), size(out, 4) * size(out, 5)))
    return out
end

# out[i, j, c, h, b] = Σ_s X[i, h, c0 + s, b] * w[s] * Y[j, gg, c0 + s, b], the states
# written by the chunks: `w[s] = dt[s] * exp(cum[L] - cum[s])` in the forward pass (X the
# inputs, Y = B), and `w[s] = exp(cum[s])` for the gradient of the states at their starts
# (X the cotangents of the outputs, Y = C).
@kernel inbounds=true cpu=false unsafe_indices=true function ssd_chunk_state_kernel!(
        out::AbstractArray{A}, @Const(X), @Const(dt), @Const(cum), @Const(Y), L::Int,
        ::Val{forward}, ::Val{c}) where {A,forward,c}
    As = @localmem A (c.PI * c.BK,)
    Bs = @localmem A (c.PJ * c.BK,)
    w = @localmem A (L_MAX_SCAN,)
    t = @index(Local, Linear) - 1
    τ, ch, hb = @index(Group, NTuple)
    P, h, n = size(X, 1), size(X, 2), size(X, 3)
    N = size(Y, 1)
    hd, b = scan_heads(hb, h)
    gg = (hd - 1) ÷ (h ÷ size(Y, 2)) + 1
    ti, tj = t % c.NR, t ÷ c.NR
    nti = cld(P, c.BI)
    i0, j0 = ((τ - 1) % nti) * c.BI, ((τ - 1) ÷ nti) * c.BJ
    c0 = (ch - 1) * L
    nv = min(n - c0, L)
    last = cum[hd, L, ch, b]
    scan_weights!(w, @inline(s -> @inbounds forward ? (s < nv ? A(dt[hd, c0 + s + 1, b]) : zero(A)) *
                                    exp(last - cum[hd, s + 1, ch, b]) :
                                    exp(cum[hd, s + 1, ch, b])), t, L, Val(c.T))
    @synchronize()
    ox, oy, sx, sy = tok_off(X, hd, c0, b), tok_off(Y, gg, c0, b), tok_stride(X), tok_stride(Y)
    fa = @inline (m, k) -> scan_ld(A, X, ox, m, k, 1, sx, P, nv) * scan_at(w, k, L)
    fb = @inline (m, k) -> scan_ld(A, Y, oy, m, k, 1, sy, N, nv)
    acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, L, i0, j0, t, ti, tj,
                        Val(true), Val(true), Val(c))
    oo = chunk_off(out, ch, hd, b)
    scan_foreach(acc, i0, j0, ti, tj, P, N, Val(c)) do i, j, v
        @inline
        @inbounds out[oo + i + P * j + 1] = v
    end
end

function ssd_chunk_state!(out, X, dt, cum, Y, L, forward::Bool)
    c = scan_tiles()
    nt = cld(size(out, 1), c.BI) * cld(size(out, 2), c.BJ)
    ssd_chunk_state_kernel!(get_backend(out), c.T)(out, X, dt, cum, Y, L, Val(forward), Val(c);
        ndrange=(c.T * nt, size(out, 3), size(out, 4) * size(out, 5)))
    return out
end

# The states at the starts of the chunks, in place of the states `S` written by them,
# and after the last into `final`: s = exp(cum[L]) * s + S[c]. Backwards, with `dS` the
# cotangents of the states at the starts of the chunks, in place of their part `S` of the
# cotangents of the states before them, and `dots[k, h, c, b]` the sums over the elements
# `k` of `dS[c] * states[c]`, to which the decay of the chunk contributes. The chunks are
# read `SCAN_BATCH` at a time: one at a time, each load would wait for the previous store.
@kernel inbounds=true cpu=false function ssd_state_passing_kernel!(
        S::AbstractArray{A}, final, @Const(initial), cum, dots, @Const(states), L::Int,
        ::Val{forward}) where {A,forward}
    shared = @localmem A (256,)
    e, hd, b = @index(Global, NTuple)
    t = @index(Local, Linear)
    k = @index(Group, NTuple)[1]
    M, nc = size(S, 1) * size(S, 2), size(S, 3)
    live = e <= M
    o = M * nc * ((hd - 1) + size(S, 4) * (b - 1)) + min(e, M)
    oe = M * ((hd - 1) + size(S, 4) * (b - 1)) + min(e, M)
    s = initial === nothing || !live ? zero(A) : A(initial[oe])
    c0 = 0
    while c0 < nc
        ws = ssd_passing_batch(S, o, M, c0, nc, forward, live, Val(SCAN_BATCH))
        ss = forward ? ws : ssd_passing_batch(states, o, M, c0, nc, forward, live, Val(SCAN_BATCH))
        for q in 1:SCAN_BATCH
            if c0 + q <= nc
                ch = forward ? c0 + q : nc + 1 - c0 - q
                decay = exp(cum[hd, L, ch, b])
                if live
                    S[o + M * (ch - 1)] = s
                end
                if !forward
                    d = group_reduce(+, shared, s * ss[q], t, Val(256))
                    if t == 1
                        dots[k, hd, ch, b] = d * decay
                    end
                end
                s = muladd(decay, s, ws[q])
            end
        end
        c0 += SCAN_BATCH
    end
    if live
        final[oe] = s
    end
end

const SCAN_BATCH = 8

# the chunks `c0 + 1:B` of `x` in the order of the pass (zero for `live = false`)
@inline ssd_passing_batch(x, o, M, c0, nc, forward, live, ::Val{B}) where {B} =
    ntuple(Val(B)) do q
        @inline
        ch = forward ? c0 + q : nc + 1 - c0 - q
        v = @inbounds x[o + M * (clamp(ch, 1, nc) - 1)]
        ifelse(live & (c0 + q <= nc), v, zero(v))
    end

function ssd_state_passing!(S, final, initial, cum, L; dots=nothing, states=nothing)
    M = size(S, 1) * size(S, 2)
    ssd_state_passing_kernel!(get_backend(S), (256, 1, 1))(S, final, initial, cum, dots, states, L,
        Val(dots === nothing); ndrange=(256 * cld(M, 256), size(S, 4), size(S, 5)))
    return S
end

# Forwards, the outputs
#   y[p, l] = Σ_{s ≤ l} x[p, s] dt[s] CB[l, s] exp(cum[l] - cum[s])
#             + Σ_n S[p, n] C[n, l] exp(cum[l]) + D * x[p, l]
# from the inputs `X = x` and the states `S` at the start of the chunk. Backwards, from
# the cotangents `X = dy` of the outputs and `S = dS` of the states after the chunk,
#   g[p, s] = Σ_{l ≥ s} dy[p, l] CB[l, s] exp(cum[l] - cum[s])
#             + Σ_n dS[p, n] B[n, s] exp(cum[L] - cum[s])
# with `dx = dt * g + D * dy`, and sums over `p` for the gradients of `dt` and `D`, into
# `uw[q, k, hd, t, b]` for the tile `k` of `p`: of `x * g` over the first part of `g` (q = 1)
# and the whole (2), of `dy * x` (3) and of `dy * y` (4), `y` the outputs.
@kernel inbounds=true cpu=false unsafe_indices=true function ssd_chunk_scan_kernel!(
        out, uw, @Const(X), @Const(x), @Const(y), @Const(dt), cum::AbstractArray{A}, @Const(CB),
        @Const(Y), @Const(S), D, L::Int, ::Val{forward}, ::Val{c}) where {A,forward,c}
    As = @localmem A (c.PI * c.BK,)
    Bs = @localmem A (c.PJ * c.BK,)
    cs = @localmem A (L_MAX_SCAN,)
    ws = @localmem A (L_MAX_SCAN,)
    red = @localmem A (c.NR * c.BJ,)
    t = @index(Local, Linear) - 1
    τ, ch, hb = @index(Group, NTuple)
    P, h, n = size(X, 1), size(X, 2), size(X, 3)
    N = size(Y, 1)
    hd, b = scan_heads(hb, h)
    gg = (hd - 1) ÷ (h ÷ size(Y, 2)) + 1
    ti, tj = t % c.NR, t ÷ c.NR
    nti = cld(P, c.BI)
    it = (τ - 1) % nti
    i0, j0 = it * c.BI, ((τ - 1) ÷ nti) * c.BJ
    c0 = (ch - 1) * L
    nv = min(n - c0, L)
    last = cum[hd, L, ch, b]
    scan_weights!(cs, @inline(s -> @inbounds cum[hd, s + 1, ch, b]), t, L, Val(c.T))
    # forwards dt, backwards the decays to the end of the chunk
    scan_weights!(ws, @inline(s -> @inbounds forward ? (s < nv ? A(dt[hd, c0 + s + 1, b]) : zero(A)) :
                                     exp(last - cum[hd, s + 1, ch, b])), t, L, Val(c.T))
    @synchronize()
    ox, oy, sx, sy = tok_off(X, hd, c0, b), tok_off(Y, gg, c0, b), tok_stride(X), tok_stride(Y)
    ocb, os = chunk_off(CB, ch, gg, b), chunk_off(S, ch, hd, b)
    dh = D === nothing ? zero(A) : A(D[hd])
    acc = scan_zeros(A, Val(c))
    if forward
        fa = @inline (m, k) -> scan_ld(A, X, ox, m, k, 1, sx, P, nv) * scan_at(ws, k, L)
        fb = @inline (m, k) -> ifelse((k <= m) & (m < L),
                                      A(@inbounds CB[ocb + min(m, L - 1) + L * min(k, L - 1) + 1]) *
                                      exp(scan_at(cs, m, L) - scan_at(cs, k, L)), zero(A))
        acc = scan_mainloop(acc, As, Bs, fa, fb, 0, min(L, j0 + c.BJ), i0, j0, t, ti, tj,
                            Val(true), Val(true), Val(c))
        # exp(cum) in place of the cumulative sums
        scan_weights!(cs, @inline(s -> @inbounds exp(cs[s + 1])), t, L, Val(c.T))
        @synchronize()
        ga = @inline (m, k) -> scan_ld(A, S, os, m, k, 1, P, P, N)
        gb = @inline (m, k) -> scan_ld(A, Y, oy, k, m, 1, sy, N, nv) * scan_at(cs, m, L)
        acc = scan_mainloop(acc, As, Bs, ga, gb, 0, N, i0, j0, t, ti, tj, Val(true), Val(false), Val(c))
        scan_foreach(acc, i0, j0, ti, tj, P, nv, Val(c)) do i, j, v
            @inline
            @inbounds out[ox + i + sx * j + 1] = v + dh * A(@inbounds X[ox + i + sx * j + 1])
        end
    else
        fa = @inline (m, k) -> scan_ld(A, X, ox, m, k, 1, sx, P, nv)
        fb = @inline (m, k) -> ifelse((m <= k) & (k < L),
                                      A(@inbounds CB[ocb + min(k, L - 1) + L * min(m, L - 1) + 1]) *
                                      exp(scan_at(cs, k, L) - scan_at(cs, m, L)), zero(A))
        acc = scan_mainloop(acc, As, Bs, fa, fb, j0, L, i0, j0, t, ti, tj, Val(true), Val(false), Val(c))
        xg = @inline (i, j, v) -> v * scan_ld(A, x, ox, i, j, 1, sx, P, nv)
        ssd_column_dots!(xg, uw, 1, acc, red, hd, b, c0, nv, it, i0, j0, t, ti, tj, Val(c))
        ga = @inline (m, k) -> scan_ld(A, S, os, m, k, 1, P, P, N)
        gb = @inline (m, k) -> scan_ld(A, Y, oy, k, m, 1, sy, N, nv) * scan_at(ws, m, L)
        acc = scan_mainloop(acc, As, Bs, ga, gb, 0, N, i0, j0, t, ti, tj, Val(true), Val(false), Val(c))
        ssd_column_dots!(xg, uw, 2, acc, red, hd, b, c0, nv, it, i0, j0, t, ti, tj, Val(c))
        dyx = @inline (i, j, v) -> scan_ld(A, X, ox, i, j, 1, sx, P, nv) * scan_ld(A, x, ox, i, j, 1, sx, P, nv)
        ssd_column_dots!(dyx, uw, 3, acc, red, hd, b, c0, nv, it, i0, j0, t, ti, tj, Val(c))
        dyy = @inline (i, j, v) -> scan_ld(A, X, ox, i, j, 1, sx, P, nv) * scan_ld(A, y, ox, i, j, 1, sx, P, nv)
        ssd_column_dots!(dyy, uw, 4, acc, red, hd, b, c0, nv, it, i0, j0, t, ti, tj, Val(c))
        scan_foreach(acc, i0, j0, ti, tj, P, nv, Val(c)) do i, j, v
            @inline
            @inbounds out[ox + i + sx * j + 1] = v * A(@inbounds dt[hd, c0 + j + 1, b]) +
                                                 dh * A(@inbounds X[ox + i + sx * j + 1])
        end
    end
end

# uw[q, it, hd, c0 + j, b] = Σ_i f(i, j, acc[i, j]) over the rows `i` of the tile
@inline function ssd_column_dots!(f, uw, q, acc::NTuple{MN,A}, red, hd, b, c0, nv, it, i0, j0, t, ti, tj,
                                  ::Val{c}) where {MN,A,c}
    ntuple(Val(c.TN)) do nn
        @inline
        jl = c.TN * tj + nn - 1
        s = zero(A)
        for m in 1:c.TM
            s += f(i0 + c.TM * ti + m - 1, j0 + jl, acc[(nn - 1) * c.TM + m])
        end
        @inbounds red[ti + c.NR * jl + 1] = s
        nothing
    end
    @synchronize()
    if t < c.BJ
        tot = zero(A)
        for r in 0:(c.NR - 1)
            tot += @inbounds red[r + c.NR * t + 1]
        end
        if j0 + t < nv
            @inbounds uw[q, it + 1, hd, c0 + j0 + t + 1, b] = tot
        end
    end
    @synchronize()
    return nothing
end

function ssd_chunk_scan!(out, uw, X, x, y, dt, cum, CB, Y, S, D, L, forward::Bool)
    c = scan_tiles()
    nt = cld(size(X, 1), c.BI) * cld(L, c.BJ)
    ssd_chunk_scan_kernel!(get_backend(out), c.T)(out, uw, X, x, y, dt, cum, CB, Y, S, D, L,
        Val(forward), Val(c); ndrange=(c.T * nt, size(cum, 3), size(X, 2) * size(X, 4)))
    return out
end

ssd_scan_tiles(P) = cld(P, scan_tiles().BI)

# The cotangent of `CB` of a group of heads,
#   dCB[l, s] = Σ_hd Σ_p dy[p, hd, l] x[p, hd, s] dt[hd, s] exp(cum[hd, l] - cum[hd, s])
# for s ≤ l, zero above. The heads of a group are split in parts of `hs`, each summed into
# its own `out[:, :, :, :, :, part]`.
@kernel inbounds=true cpu=false unsafe_indices=true function ssd_dcb_kernel!(
        out, @Const(dy), @Const(x), @Const(dt), cum::AbstractArray{A}, L::Int, hs::Int, ::Val{c}) where {A,c}
    As = @localmem A (c.PI * c.BK,)
    Bs = @localmem A (c.PJ * c.BK,)
    cs = @localmem A (L_MAX_SCAN,)
    ws = @localmem A (L_MAX_SCAN,)
    t = @index(Local, Linear) - 1
    τ, ch, gbs = @index(Group, NTuple)
    P, h, n = size(x, 1), size(x, 2), size(x, 3)
    G, nb = size(out, 4), size(out, 5)
    gb, part = scan_heads(gbs, G * nb)
    gg, b = scan_heads(gb, G)
    r = h ÷ G
    ti, tj = t % c.NR, t ÷ c.NR
    it, jt = scan_lower_tile(τ - 1)
    i0, j0 = it * c.BI, jt * c.BJ
    c0 = (ch - 1) * L
    nv = min(n - c0, L)
    sx = tok_stride(x)
    tot = scan_zeros(A, Val(c))
    for hl in ((part - 1) * hs + 1):min(r, part * hs)
        hd = (gg - 1) * r + hl
        scan_weights!(cs, @inline(s -> @inbounds cum[hd, s + 1, ch, b]), t, L, Val(c.T))
        scan_weights!(ws, @inline(s -> @inbounds s < nv ? A(dt[hd, c0 + s + 1, b]) : zero(A)), t, L, Val(c.T))
        ox = tok_off(x, hd, c0, b)
        fa = @inline (m, k) -> scan_ld(A, dy, ox, k, m, 1, sx, P, nv)
        fb = @inline (m, k) -> scan_ld(A, x, ox, k, m, 1, sx, P, nv)
        acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, P, i0, j0, t, ti, tj,
                            Val(false), Val(false), Val(c))
        tot = ssd_dcb_add(tot, acc, cs, ws, i0, j0, ti, tj, L, Val(c))
        @synchronize()
    end
    oo = chunk_off(out, ch, gg, b) + (part - 1) * L * L * size(out, 3) * G * nb
    scan_foreach(tot, i0, j0, ti, tj, L, L, Val(c)) do i, j, v
        @inline
        @inbounds out[oo + i + L * j + 1] = v
    end
end

@inline ssd_dcb_add(tot::NTuple{MN,A}, acc, cs, ws, i0, j0, ti, tj, L, ::Val{c}) where {MN,A,c} =
    ntuple(Val(MN)) do x
        @inline
        i = i0 + c.TM * ti + (x - 1) % c.TM
        j = j0 + c.TN * tj + (x - 1) ÷ c.TM
        w = scan_at(ws, j, L) * exp(scan_at(cs, i, L) - scan_at(cs, j, L))
        ifelse((j <= i) & (i < L), muladd(acc[x], w, tot[x]), tot[x])
    end

function ssd_dcb(dy, x, dt::AbstractArray{T}, cum, L, g) where {T}
    c = scan_tiles()
    nt = cld(L, c.BI)
    nc, b = size(cum, 3), size(x, 4)
    hs, np = scan_head_parts(get_backend(x), (nt * (nt + 1)) ÷ 2 * nc * g * b, size(x, 2) ÷ g)
    out = similar(x, T, L, L, nc, g, b, np)
    ssd_dcb_kernel!(get_backend(out), c.T)(out, dy, x, dt, cum, L, hs, Val(c);
        ndrange=(c.T * (nt * (nt + 1)) ÷ 2, nc, g * b * np))
    return np == 1 ? reshape(out, L, L, nc, g, b) : reshape(sum(out; dims=6), L, L, nc, g, b)
end

# The heads of a group in parts of `hs`, `np` of them, so that the kernels over groups have
# blocks for all the compute units even with few groups.
function scan_head_parts(backend, blocks, r)
    np = clamp(cld(2 * attention_compute_units(backend), blocks), 1, r)
    hs = cld(r, np)
    return hs, cld(r, hs)
end

# The cotangents of `B` (`dB = true`) and `C`, for a group of heads:
#   dB[n, s] = Σ_{l ≥ s} C[n, l] dCB[l, s] + Σ_hd Σ_p dS[p, n] x[p, s] dt[s] exp(cum[L] - cum[s])
#   dC[n, l] = Σ_{s ≤ l} B[n, s] dCB[l, s] + Σ_hd Σ_p S[p, n] dy[p, l] exp(cum[l])
# with `Z` = C or B, `W` = dS or S, `X` = x or dy, the heads split in parts as for `dCB`.
@kernel inbounds=true cpu=false unsafe_indices=true function ssd_dbc_kernel!(
        out, @Const(Z), @Const(dCB), @Const(W), @Const(X), @Const(dt), cum::AbstractArray{A},
        L::Int, hs::Int, ::Val{dB}, ::Val{c}) where {A,dB,c}
    As = @localmem A (c.PI * c.BK,)
    Bs = @localmem A (c.PJ * c.BK,)
    ws = @localmem A (L_MAX_SCAN,)
    t = @index(Local, Linear) - 1
    τ, ch, gbs = @index(Group, NTuple)
    P, h, n = size(X, 1), size(X, 2), size(X, 3)
    N, G, nb = size(Z, 1), size(Z, 2), size(Z, 4)
    gb, part = scan_heads(gbs, G * nb)
    gg, b = scan_heads(gb, G)
    r = h ÷ G
    ti, tj = t % c.NR, t ÷ c.NR
    nti = cld(N, c.BI)
    i0, j0 = ((τ - 1) % nti) * c.BI, ((τ - 1) ÷ nti) * c.BJ
    c0 = (ch - 1) * L
    nv = min(n - c0, L)
    oz, sz, sx = tok_off(Z, gg, c0, b), tok_stride(Z), tok_stride(X)
    ocb = chunk_off(dCB, ch, gg, b)
    fa = @inline (m, k) -> scan_ld(A, Z, oz, m, k, 1, sz, N, nv)
    acc = scan_zeros(A, Val(c))
    if dB
        fb = @inline (m, k) -> ifelse((m <= k) & (k < L),
                                      A(@inbounds dCB[ocb + min(k, L - 1) + L * min(m, L - 1) + 1]), zero(A))
        acc = scan_mainloop(acc, As, Bs, fa, fb, part == 1 ? j0 : L, L, i0, j0, t, ti, tj,
                            Val(true), Val(false), Val(c))
    else
        fb = @inline (m, k) -> ifelse((k <= m) & (m < L),
                                      A(@inbounds dCB[ocb + min(m, L - 1) + L * min(k, L - 1) + 1]), zero(A))
        acc = scan_mainloop(acc, As, Bs, fa, fb, 0, part == 1 ? min(L, j0 + c.BJ) : 0, i0, j0, t, ti, tj,
                            Val(true), Val(true), Val(c))
    end
    for hl in ((part - 1) * hs + 1):min(r, part * hs)
        hd = (gg - 1) * r + hl
        last = cum[hd, L, ch, b]
        scan_weights!(ws, @inline(s -> @inbounds dB ? (s < nv ? A(dt[hd, c0 + s + 1, b]) : zero(A)) *
                                    exp(last - cum[hd, s + 1, ch, b]) :
                                    exp(cum[hd, s + 1, ch, b])), t, L, Val(c.T))
        @synchronize()
        ow, ox = chunk_off(W, ch, hd, b), tok_off(X, hd, c0, b)
        ga = @inline (m, k) -> scan_ld(A, W, ow, k, m, 1, P, P, N)
        gb = @inline (m, k) -> scan_ld(A, X, ox, k, m, 1, sx, P, nv) * scan_at(ws, m, L)
        acc = scan_mainloop(acc, As, Bs, ga, gb, 0, P, i0, j0, t, ti, tj, Val(false), Val(false), Val(c))
    end
    oo = oz + (part - 1) * length(Z)
    scan_foreach(acc, i0, j0, ti, tj, N, nv, Val(c)) do i, j, v
        @inline
        @inbounds out[oo + i + sz * j + 1] = v
    end
end

function ssd_dbc(Z, dCB, W, X, dt, cum::AbstractArray{T}, L, dB::Bool) where {T}
    c = scan_tiles()
    nt = cld(size(Z, 1), c.BI) * cld(L, c.BJ)
    nc, g = size(cum, 3), size(Z, 2)
    hs, np = scan_head_parts(get_backend(Z), nt * nc * g * size(Z, 4), size(X, 2) ÷ g)
    out = similar(Z, np == 1 ? eltype(Z) : T, size(Z)..., np)
    ssd_dbc_kernel!(get_backend(out), c.T)(out, Z, dCB, W, X, dt, cum, L, hs, Val(dB), Val(c);
        ndrange=(c.T * nt, nc, g * size(Z, 4) * np))
    np == 1 && return reshape(out, size(Z))
    return copyto!(similar(Z), reshape(sum(out; dims=5), size(Z)))
end

# cum[hd, l, c, b]: the cumulative sums of `dt * A` within the chunks of `L` tokens
ssd_chunk_cumsum(dt, A, L) = cumsum(reshape(pad_tokens(dt .* A, L), size(dt, 1), L, :, size(dt, 3)); dims=2)

# `ssd_chunked` with the kernels, for dense arrays, `dt` masked and of the compute type
# `T`, as `A` and `D`.
function ssd_kernels(x, dt, A, B, C, D, state, L)
    y, final, _ = ssd_kernels_forward(x, dt, A, B, C, D, state, L, eltype(x))
    return y, final
end

# ... with the outputs of type `Y`, and what the gradient needs
function ssd_kernels_forward(x, dt::AbstractArray{T}, A, B, C, D, state, L, ::Type{Y}) where {T,Y}
    P, h, n, b = size(x)
    N, g = size(B, 1), size(B, 2)
    nc = cld(n, L)
    cum = ssd_chunk_cumsum(dt, A, L)
    CB = scan_gram!(similar(x, T, L, L, nc, g, b), C, B, L)
    S = ssd_chunk_state!(similar(x, T, P, N, nc, h, b), x, dt, cum, B, L, true)
    final = similar(x, T, P, N, h, b)
    ssd_state_passing!(S, final, state, cum, L)
    y = similar(x, Y, P, h, n, b)
    ssd_chunk_scan!(y, nothing, x, x, nothing, dt, cum, CB, C, S, D, L, true)
    return y, final, (; cum, CB, S)
end

function ∇ssd_kernels(dy, dfinal, y, x, dt::AbstractArray{T}, A, B, C, D, state, L, saved) where {T}
    P, h, n, b = size(x)
    N = size(B, 1)
    (; cum, CB, S) = saved
    nc = size(cum, 3)
    # the cotangents of the states at the ends of the chunks, from those of the outputs
    dS = ssd_chunk_state!(similar(S), dy, dt, cum, C, L, false)
    dots = similar(S, cld(P * N, 256), h, nc, b)
    dstate = similar(S, P, N, h, b)
    ssd_state_passing!(dS, dstate, dfinal, cum, L; dots, states=S)
    uw = similar(S, 4, ssd_scan_tiles(P), h, n, b)
    dx = ssd_chunk_scan!(similar(x), uw, dy, x, y, dt, cum, CB, B, dS, D, L, false)
    dCB = ssd_dcb(dy, x, dt, cum, L, size(B, 2))
    dB = ssd_dbc(C, dCB, dS, x, dt, cum, L, true)
    dC = ssd_dbc(B, dCB, S, dy, dt, cum, L, false)
    # the gradients of `dt` and `A`, through the cumulative sums
    uw = sum(uw; dims=2)
    u, gx, xy, yy = (reshape(uw[q:q, :, :, :, :], h, n, b) for q in 1:4)
    dcum = D === nothing ? yy .- dt .* gx : yy .- reshape(D, h) .* xy .- dt .* gx
    dlast = reshape(sum(dots; dims=1), h, nc, b) .+
            reshape(sum(reshape(pad_tokens(dt .* (gx .- u), L), h, L, nc, b); dims=2), h, nc, b)
    da = reverse_chunk_cumsum(dcum, L) .+ repeat_chunks(dlast, L, n)
    ddt = gx .+ reshape(A, h) .* da
    dA = vec(sum(dt .* da; dims=(2, 3)))
    dD = D === nothing ? NoTangent() : vec(sum(xy; dims=(2, 3)))
    return dx, ddt, dA, dB, dC, dD, state === nothing ? NoTangent() : dstate
end

pad_tokens(a, L) = (pad = mod(-size(a, 2), L); pad == 0 ? a : pad_zeros(a, (0, pad); dims=2))

# Σ_{l ≥ t} d[l] over the positions `l` of the chunk of `t`
function reverse_chunk_cumsum(d, L)
    h, n, b = size(d)
    r = reverse_cumsum(reshape(pad_tokens(d, L), h, L, :, b); dims=2)
    return reshape(r, h, :, b)[:, 1:n, :]
end

# `x` (h, nc, b) repeated over the `L` positions of each chunk, the first `n`
repeat_chunks(x, L, n) = reshape(repeat(reshape(x, size(x, 1), 1, size(x, 2), size(x, 3)); inner=(1, L, 1, 1)),
                                 size(x, 1), :, size(x, 3))[:, 1:n, :]


#######################
# gated_delta_rule
#######################

# the longest chunk of the kernels of the gated delta rule
const L_MAX_DELTA = 64

# For the chunk `ch` of the head `hd`: the inverse `T` of `I + A`, the strictly lower
#   A[j, i] = beta[j] * exp(cum[j] - cum[i]) * KK[j, i],    KK[j, i] = k[:, j] ⋅ k[:, i],
# into `Tm` unless it is `nothing`, and the writes of the values and of the keys,
#   u[:, j] = Σ_i T[j, i] * beta[i] * v[:, i],    w[:, j] = Σ_i T[j, i] * beta[i] * exp(cum[i]) * k[:, i].
# `T` is found row by row, in place of `A` in shared memory.
@kernel inbounds=true cpu=false unsafe_indices=true function delta_prepare_kernel!(
        Tm, u, w, @Const(KK), @Const(k), @Const(v), @Const(beta), cum::AbstractArray{A}, L::Int,
        ::Val{c}) where {A,c}
    LP = L_MAX_DELTA + 1
    Ts = @localmem A ((L_MAX_DELTA + 1) * L_MAX_DELTA,)
    As = @localmem A (c.PI * c.BK,)
    cs = @localmem A (L_MAX_DELTA,)
    bs = @localmem A (L_MAX_DELTA,)
    t = @index(Local, Linear) - 1
    ch, hd, b = @index(Group, NTuple)
    dk, hk, n = size(k, 1), size(k, 2), size(k, 3)
    dv, h = size(v, 1), size(v, 2)
    kh = (hd - 1) ÷ (h ÷ hk) + 1
    ti, tj = t % c.NR, t ÷ c.NR
    c0 = (ch - 1) * L
    nv = min(n - c0, L)
    scan_weights!(cs, @inline(s -> @inbounds cum[hd, s + 1, ch, b]), t, L, Val(c.T))
    scan_weights!(bs, @inline(s -> @inbounds s < nv ? A(beta[hd, c0 + s + 1, b]) : zero(A)), t, L, Val(c.T))
    @synchronize()
    okk = chunk_off(KK, ch, kh, b)
    e = t
    while e < LP * L_MAX_DELTA
        j, i = e % LP, e ÷ LP
        jj, ii = min(j, L - 1), min(i, L - 1)
        a = bs[jj + 1] * exp(cs[jj + 1] - cs[ii + 1]) * A(KK[okk + jj + L * ii + 1])
        Ts[e + 1] = ifelse((i < j) & (j < L), a, zero(A))
        e += c.T
    end
    @synchronize()
    delta_tril_inverse!(Ts, t, Val(LP), Val(c.T))
    if Tm !== nothing
        ot = chunk_off(Tm, ch, hd, b)
        e = t
        while e < L_MAX_DELTA * L_MAX_DELTA
            j, i = e % L_MAX_DELTA, e ÷ L_MAX_DELTA
            if (j < L) & (i < L)
                Tm[ot + j + L * i + 1] = Ts[j + LP * i + 1]
            end
            e += c.T
        end
    end
    tr = ScanShared(Ts, 0, Val(1), Val(L_MAX_DELTA + 1))
    ov, sv, ok, sk = tok_off(v, hd, c0, b), tok_stride(v), tok_off(k, kh, c0, b), tok_stride(k)
    ou, ow = chunk_off(u, ch, hd, b), chunk_off(w, ch, hd, b)
    for i0 in 0:c.BI:(dv - 1)
        fv = @inline (m, kk) -> scan_ld(A, v, ov, m, kk, 1, sv, dv, nv) * scan_at(bs, kk, L)
        acc = scan_mainloop(scan_zeros(A, Val(c)), As, nothing, fv, tr, 0, L, i0, 0, t, ti, tj,
                            Val(true), Val(true), Val(c))
        scan_foreach(acc, i0, 0, ti, tj, dv, L, Val(c)) do i, j, x
            @inline
            @inbounds u[ou + i + dv * j + 1] = x
        end
    end
    for i0 in 0:c.BI:(dk - 1)
        fk = @inline (m, kk) -> scan_ld(A, k, ok, m, kk, 1, sk, dk, nv) * scan_at(bs, kk, L) *
                                exp(scan_at(cs, kk, L))
        acc = scan_mainloop(scan_zeros(A, Val(c)), As, nothing, fk, tr, 0, L, i0, 0, t, ti, tj,
                            Val(true), Val(true), Val(c))
        scan_foreach(acc, i0, 0, ti, tj, dk, L, Val(c)) do i, j, x
            @inline
            @inbounds w[ow + i + dk * j + 1] = x
        end
    end
end

# The inverse of `I + A` in place of the strictly lower `A`, of size `L_MAX_DELTA`, in
# `Ts[j + LP * i + 1]`, by blocks of 16: the diagonal blocks are inverted row by row, then
# each row of blocks below the first from those above it,
#   T[I, J] = -T[I, I] * Σ_{J ≤ K < I} A[I, K] * T[K, J].
@inline function delta_tril_inverse!(Ts, t, ::Val{LP}, ::Val{T}) where {LP,T}
    blk, col = t ÷ 16, t % 16
    base = 16 * blk
    for r in 1:15
        j = base + r
        x = zero(eltype(Ts))
        if (t < L_MAX_DELTA) & (col < r)
            x = -Ts[j + LP * (base + col) + 1]
            for i in (col + 1):(r - 1)
                x = muladd(-Ts[j + LP * (base + i) + 1], Ts[base + i + LP * (base + col) + 1], x)
            end
        end
        @synchronize()
        if (t < L_MAX_DELTA) & (col < r)
            Ts[j + LP * (base + col) + 1] = x
        end
        @synchronize()
    end
    if t < L_MAX_DELTA
        Ts[t + LP * t + 1] = one(eltype(Ts))
    end
    @synchronize()
    for I in 1:(L_MAX_DELTA ÷ 16 - 1)
        r0 = 16 * I
        # R[j, i] = Σ_{i ≤ k < r0} A[j, k] * T[k, i] for the rows `j` of the block row
        vals = ntuple(Val(cld(16 * (L_MAX_DELTA - 16), T))) do q
            @inline
            e = t + (q - 1) * T
            j, i = r0 + e % 16, e ÷ 16
            x = zero(eltype(Ts))
            if e < 16 * r0
                for k in i:(r0 - 1)
                    x = muladd(Ts[j + LP * k + 1], Ts[k + LP * i + 1], x)
                end
            end
            x
        end
        @synchronize()
        delta_put_block_row!(Ts, vals, t, r0, Val(LP), Val(T))
        @synchronize()
        # T[j, i] = -Σ_{r0 ≤ r ≤ j} T[j, r] * R[r, i]
        vals = ntuple(Val(cld(16 * (L_MAX_DELTA - 16), T))) do q
            @inline
            e = t + (q - 1) * T
            j, i = r0 + e % 16, e ÷ 16
            x = zero(eltype(Ts))
            if e < 16 * r0
                for r in r0:j
                    x = muladd(-Ts[j + LP * r + 1], Ts[r + LP * i + 1], x)
                end
            end
            x
        end
        @synchronize()
        delta_put_block_row!(Ts, vals, t, r0, Val(LP), Val(T))
        @synchronize()
    end
    return nothing
end

@inline function delta_put_block_row!(Ts, vals::NTuple{Q}, t, r0, ::Val{LP}, ::Val{T}) where {Q,LP,T}
    ntuple(Val(Q)) do q
        @inline
        e = t + (q - 1) * T
        if e < 16 * r0
            Ts[r0 + e % 16 + LP * (e ÷ 16) + 1] = vals[q]
        end
        nothing
    end
    return nothing
end

delta_prepare_shmem(c) = (L_MAX_DELTA + 1) * L_MAX_DELTA + c.PI * c.BK + 2 * L_MAX_DELTA

function delta_prepare!(Tm, u, w, KK, k, v, beta, cum, L)
    c = scan_tiles()
    delta_prepare_kernel!(get_backend(u), c.T)(Tm, u, w, KK, k, v, beta, cum, L, Val(c);
        ndrange=(c.T * size(u, 3), size(u, 4), size(u, 5)))
    return u, w
end

# The states at the starts of the chunks into `states`, after the last into `final`, from
# `initial`. In a chunk, the writes `u` become `U = u - S * w` in place, and then
#   S = exp(cum[L]) * S + Σ_j exp(cum[L] - cum[j]) * U[:, j] * k[:, j]ᵀ.
# A block carries the rows `d0 .+ (0:BI-1)` of the state of a head, in shared memory.
@kernel inbounds=true cpu=false unsafe_indices=true function delta_recurrence_kernel!(
        states, final, @Const(initial), u, @Const(w), @Const(k), cum::AbstractArray{A}, L::Int,
        ::Val{dkp}, ::Val{c}, ::Val{c2}) where {A,dkp,c,c2}
    PS = c.BI + 4
    Ss = @localmem A ((c.BI + 4) * dkp,)
    Us = @localmem A ((c.BI + 4) * L_MAX_DELTA,)
    Bs = @localmem A (max(c.PJ, c2.PJ) * c.BK,)
    ws = @localmem A (L_MAX_DELTA,)
    t = @index(Local, Linear) - 1
    dtile, hd, b = @index(Group, NTuple)
    d0 = (dtile - 1) * c.BI
    dk, hk, n = size(k, 1), size(k, 2), size(k, 3)
    dv, nc, h = size(u, 1), size(u, 3), size(u, 4)
    kh = (hd - 1) ÷ (h ÷ hk) + 1
    ti, tj = t % c.NR, t ÷ c.NR
    e = t
    while e < PS * dkp
        d, f = e % PS, e ÷ PS
        Ss[e + 1] = initial === nothing ? zero(A) :
                    ifelse((d < c.BI) & (d0 + d < dv) & (f < dk),
                           A(initial[min(d0 + d, dv - 1) + 1, min(f, dk - 1) + 1, hd, b]), zero(A))
        e += c.T
    end
    e = t
    while e < PS * L_MAX_DELTA
        Us[e + 1] = zero(A)
        e += c.T
    end
    @synchronize()
    sop, uop = ScanShared(Ss, 0, Val(1), Val(c.BI + 4)), ScanShared(Us, 0, Val(1), Val(c.BI + 4))
    nd = min(c.BI, dv - d0)
    sk = tok_stride(k)
    for ch in 1:nc
        c0 = (ch - 1) * L
        nv = min(n - c0, L)
        last = cum[hd, L, ch, b]
        scan_weights!(ws, @inline(s -> @inbounds exp(last - cum[hd, s + 1, ch, b])), t, L, Val(c.T))
        delta_store_rows!(states, chunk_off(states, ch, hd, b), Ss, d0, nd, dv, dk, PS, t, Val(c.T), Val(c.BI))
        ow, ou = chunk_off(w, ch, hd, b), chunk_off(u, ch, hd, b)
        fw = @inline (m, kk) -> scan_ld(A, w, ow, kk, m, 1, dk, dk, L)
        acc = scan_mainloop(scan_zeros(A, Val(c)), nothing, Bs, sop, fw, 0, dk, 0, 0, t, ti, tj,
                            Val(true), Val(false), Val(c))
        scan_foreach(acc, 0, 0, ti, tj, nd, L, Val(c)) do i, j, x
            @inline
            U = @inbounds u[ou + d0 + i + dv * j + 1] - x
            @inbounds u[ou + d0 + i + dv * j + 1] = U
            @inbounds Us[i + PS * j + 1] = U * ws[j + 1]
        end
        @synchronize()
        decay = exp(last)
        okt = tok_off(k, kh, c0, b)
        ti2, tj2 = t % c2.NR, t ÷ c2.NR
        for f0 in 0:c2.BJ:(dk - 1)
            fk = @inline (m, kk) -> scan_ld(A, k, okt, m, kk, 1, sk, dk, nv)
            acc = scan_mainloop(scan_zeros(A, Val(c2)), nothing, Bs, uop, fk, 0, L, 0, f0, t, ti2, tj2,
                                Val(true), Val(true), Val(c2))
            scan_foreach(acc, 0, f0, ti2, tj2, c2.BI, dk, Val(c2)) do i, f, x
                @inline
                @inbounds Ss[i + PS * f + 1] = muladd(decay, Ss[i + PS * f + 1], x)
            end
        end
        @synchronize()
    end
    delta_store_rows!(final, dv * dk * ((hd - 1) + h * (b - 1)), Ss, d0, nd, dv, dk, PS, t, Val(c.T), Val(c.BI))
end

# x[o + d0 + d + dv * f + 1] = Ss[d + PS * f + 1] for the `nd` rows `d < BV` of the block
@inline function delta_store_rows!(x, o, Ss, d0, nd, dv, dk, PS, t, ::Val{T}, ::Val{BV}) where {T,BV}
    e = t
    while e < BV * dk
        d, f = e % BV, e ÷ BV
        if d < nd
            @inbounds x[o + d0 + d + dv * f + 1] = Ss[d + PS * f + 1]
        end
        e += T
    end
    return nothing
end

# The tilings of the two products of the recurrence (`reverse` for the gradient) for `BV`
# rows of the state in a block, the most that fit in `budget` floats, and `dk` rounded up
# to the slabs.
function delta_recurrence_tiling(dk, budget; reverse::Bool=false)
    dkp = cld(dk, 16) * 16
    for (BV, T, TM) in ((32, 128, 4), (16, 128, 2), (8, 64, 2))
        c = scan_tiling(BV, 64, 16, TM; T)
        c2 = scan_tiling(BV, 128, 16, 2TM; T)
        shmem = (BV + 4) * (dkp + L_MAX_DELTA) + max(c.PJ, c2.PJ) * c.BK + L_MAX_DELTA
        reverse && (shmem += L_MAX_DELTA + max(c.NR * c.BJ, c.T) + c2.PI * c2.BK)
        shmem <= budget && return c, c2, dkp
    end
    return nothing
end

function delta_recurrence!(states, final, initial, u, w, k, cum, L)
    backend = get_backend(u)
    c, c2, dkp = delta_recurrence_tiling(size(k, 1), attention_shmem_budget(backend) ÷ sizeof(eltype(u)))
    delta_recurrence_kernel!(backend, c.T)(states, final, initial, u, w, k, cum, L, Val(dkp), Val(c), Val(c2);
        ndrange=(c.T * cld(size(u, 1), c.BI), size(u, 4), size(u, 5)))
    return states
end

# The outputs, scaled by `scale`:
#   o[:, l] = Σ_{j ≤ l} exp(cum[l] - cum[j]) * QK[l, j] * U[:, j] + exp(cum[l]) * S * q[:, l]
# with `S` the state at the start of the chunk and `QK[l, j] = q[:, l] ⋅ k[:, j]`.
@kernel inbounds=true cpu=false unsafe_indices=true function delta_output_kernel!(
        o, @Const(U), @Const(QK), @Const(states), @Const(q), cum::AbstractArray{A}, scale, L::Int,
        ::Val{c}) where {A,c}
    As = @localmem A (c.PI * c.BK,)
    Bs = @localmem A (c.PJ * c.BK,)
    cs = @localmem A (L_MAX_DELTA,)
    t = @index(Local, Linear) - 1
    τ, ch, hb = @index(Group, NTuple)
    dk, hk, n = size(q, 1), size(q, 2), size(q, 3)
    dv, h = size(o, 1), size(o, 2)
    hd, b = scan_heads(hb, h)
    kh = (hd - 1) ÷ (h ÷ hk) + 1
    ti, tj = t % c.NR, t ÷ c.NR
    i0 = (τ - 1) * c.BI
    c0 = (ch - 1) * L
    nv = min(n - c0, L)
    scan_weights!(cs, @inline(s -> @inbounds cum[hd, s + 1, ch, b]), t, L, Val(c.T))
    @synchronize()
    ou, oqk, os = chunk_off(U, ch, hd, b), chunk_off(QK, ch, kh, b), chunk_off(states, ch, hd, b)
    oq, sq = tok_off(q, kh, c0, b), tok_stride(q)
    fa = @inline (m, kk) -> scan_ld(A, U, ou, m, kk, 1, dv, dv, L)
    fb = @inline (m, kk) -> ifelse((kk <= m) & (m < L),
                                   A(@inbounds QK[oqk + min(m, L - 1) + L * min(kk, L - 1) + 1]) *
                                   exp(scan_at(cs, m, L) - scan_at(cs, kk, L)), zero(A))
    acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, L, i0, 0, t, ti, tj,
                        Val(true), Val(true), Val(c))
    scan_weights!(cs, @inline(s -> @inbounds exp(cs[s + 1])), t, L, Val(c.T))
    @synchronize()
    ga = @inline (m, kk) -> scan_ld(A, states, os, m, kk, 1, dv, dv, dk)
    gb = @inline (m, kk) -> scan_ld(A, q, oq, kk, m, 1, sq, dk, nv) * scan_at(cs, m, L)
    acc = scan_mainloop(acc, As, Bs, ga, gb, 0, dk, i0, 0, t, ti, tj, Val(true), Val(false), Val(c))
    oo, so = tok_off(o, hd, c0, b), tok_stride(o)
    scan_foreach(acc, i0, 0, ti, tj, dv, nv, Val(c)) do i, j, x
        @inline
        @inbounds o[oo + i + so * j + 1] = scale * x
    end
end

function delta_output!(o, U, QK, states, q, cum, scale, L)
    c = scan_tiles()
    delta_output_kernel!(get_backend(o), c.T)(o, U, QK, states, q, cum, scale, L, Val(c);
        ndrange=(c.T * cld(size(o, 1), c.BI), size(U, 3), size(o, 2) * size(o, 4)))
    return o
end

# `gated_delta_chunked` with the kernels, for dense arrays, `g` and `beta` masked and of the
# compute type `T`.
function delta_kernels(q, k, v, g, beta, scale, state, L)
    o, final, _ = delta_kernels_forward(q, k, v, g, beta, scale, state, L, eltype(v), Val(false))
    return o, final
end

# ... with the outputs of type `Y`, and with `keep` what the gradient needs
function delta_kernels_forward(q, k, v, g::AbstractArray{T}, beta, scale, state, L, ::Type{Y},
                               ::Val{keep}) where {T,Y,keep}
    dk, hk, n, b = size(q)
    dv, h = size(v, 1), size(v, 2)
    nc = cld(n, L)
    cum = cumsum(reshape(pad_tokens(g, L), h, L, nc, b); dims=2)
    QK = scan_gram!(similar(q, T, L, L, nc, hk, b), q, k, L)
    KK = scan_gram!(similar(q, T, L, L, nc, hk, b), k, k, L)
    Tm = keep ? similar(q, T, L, L, nc, h, b) : nothing
    U, W = similar(q, T, dv, L, nc, h, b), similar(q, T, dk, L, nc, h, b)
    delta_prepare!(Tm, U, W, KK, k, v, beta, cum, L)
    states, final = similar(q, T, dv, dk, nc, h, b), similar(q, T, dv, dk, h, b)
    delta_recurrence!(states, final, state, U, W, k, cum, L)
    o = similar(v, Y, dv, h, n, b)
    delta_output!(o, U, QK, states, q, cum, scale, L)
    return o, final, (; cum, QK, KK, Tm, U, W, states)
end

# The gradient. With `dO = scale * Δo` and, in a chunk, `F[j] = exp(cum[L] - cum[j])`,
# `E[l] = exp(cum[l])`, `S0` and `S1` the states at its start and end:
#   dU  = dO Mᵀ + dS1 K F,   M[j, l] = QK[l, j] exp(cum[l] - cum[j]) for j ≤ l   (dU from o: dUl)
#   dS0 = exp(cum[L]) dS1 + dO E Qᵀ - dU Wᵀ                                         (backwards over the chunks)
#   dW  = -S0ᵀ dU,   dG = (Uᵀ dO) ∘ M / QK,   dT = dUᵀ V B + dWᵀ K B E,   dA = -Tᵀ dT Tᵀ
#   dq  = S0ᵀ dO E + K dG,   dv = dU T B,
#   dk  = Q dGᵀ + dS1ᵀ U F + dW T B E + K (dKK + dKKᵀ),   dKK[j, i] = dA[j, i] beta[j] exp(cum[j] - cum[i])
# and the gradients of `cum` and `beta` from sums over the dimensions of the heads, which
# the kernels accumulate into `dpos[q, tile, hd, t, b]`:
#   1: U ⋅ dU,  2: U ⋅ dUl,  3: k ⋅ (dW T),  4: v ⋅ (dU T),
#   5, 6: the sums of the rows and columns of dA ∘ A,  7: Σ_i dA[j, i] exp(cum[j] - cum[i]) KK[j, i].

# dUl, the part of the cotangent of the writes from the outputs, and `dpos[2, ...]`
@kernel inbounds=true cpu=false unsafe_indices=true function delta_dul_kernel!(
        dU, dpos, @Const(dO), @Const(U), @Const(QK), cum::AbstractArray{A}, scale, L::Int, ::Val{c}) where {A,c}
    As = @localmem A (c.PI * c.BK,)
    Bs = @localmem A (c.PJ * c.BK,)
    cs = @localmem A (L_MAX_DELTA,)
    red = @localmem A (c.NR * c.BJ,)
    t = @index(Local, Linear) - 1
    τ, ch, hb = @index(Group, NTuple)
    dv, h, n = size(dO, 1), size(dO, 2), size(dO, 3)
    hd, b = scan_heads(hb, h)
    kh = (hd - 1) ÷ (h ÷ size(QK, 4)) + 1
    ti, tj = t % c.NR, t ÷ c.NR
    it = τ - 1
    i0 = it * c.BI
    c0 = (ch - 1) * L
    nv = min(n - c0, L)
    scan_weights!(cs, @inline(s -> @inbounds cum[hd, s + 1, ch, b]), t, L, Val(c.T))
    @synchronize()
    odo, sdo = tok_off(dO, hd, c0, b), tok_stride(dO)
    oqk, ou = chunk_off(QK, ch, kh, b), chunk_off(U, ch, hd, b)
    fa = @inline (m, kk) -> scan_ld(A, dO, odo, m, kk, 1, sdo, dv, nv) * scale
    fb = @inline (m, kk) -> ifelse((m <= kk) & (kk < L),
                                   A(@inbounds QK[oqk + min(kk, L - 1) + L * min(m, L - 1) + 1]) *
                                   exp(scan_at(cs, kk, L) - scan_at(cs, m, L)), zero(A))
    acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, L, i0, 0, t, ti, tj,
                        Val(true), Val(false), Val(c))
    scan_foreach(acc, i0, 0, ti, tj, dv, L, Val(c)) do i, j, x
        @inline
        @inbounds dU[ou + i + dv * j + 1] = x
    end
    uu = @inline (i, j, x) -> x * scan_ld(A, U, ou, i, j, 1, dv, dv, L)
    ssd_column_dots!(uu, dpos, 2, acc, red, hd, b, c0, nv, it, i0, 0, t, ti, tj, Val(c))
end

# Backwards over the chunks, from the cotangent `dS` of the final state (or zero): dU in
# place of dUl, the cotangents of the states after the chunks into `dstates`, of the
# initial one into `dinitial`, `dots[k, hd, c, b]` the sums over the block `k` of rows of
# `dS1 ∘ S0` times the decay of the chunk, and `dpos[1, ...]`.
@kernel inbounds=true cpu=false unsafe_indices=true function delta_reverse_kernel!(
        dU, dstates, dinitial, dots, dpos, @Const(dfinal), @Const(dO), @Const(states), @Const(U),
        @Const(w), @Const(q), @Const(k), cum::AbstractArray{A}, scale, L::Int,
        ::Val{dkp}, ::Val{c}, ::Val{c2}) where {A,dkp,c,c2}
    PS = c.BI + 4
    Ss = @localmem A ((c.BI + 4) * dkp,)
    Us = @localmem A ((c.BI + 4) * L_MAX_DELTA,)
    Bs = @localmem A (max(c.PJ, c2.PJ) * c.BK,)
    ws = @localmem A (L_MAX_DELTA,)
    es = @localmem A (L_MAX_DELTA,)
    red = @localmem A (max(c.NR * c.BJ, c.T),)
    As2 = @localmem A (c2.PI * c2.BK,)
    t = @index(Local, Linear) - 1
    dtile, hd, b = @index(Group, NTuple)
    d0 = (dtile - 1) * c.BI
    dk, hk, n = size(k, 1), size(k, 2), size(k, 3)
    dv, nc, h = size(U, 1), size(U, 3), size(U, 4)
    kh = (hd - 1) ÷ (h ÷ hk) + 1
    ti, tj = t % c.NR, t ÷ c.NR
    ti2, tj2 = t % c2.NR, t ÷ c2.NR
    e = t
    while e < PS * dkp
        d, f = e % PS, e ÷ PS
        Ss[e + 1] = dfinal === nothing ? zero(A) :
                    ifelse((d < c.BI) & (d0 + d < dv) & (f < dk),
                           A(dfinal[min(d0 + d, dv - 1) + 1, min(f, dk - 1) + 1, hd, b]), zero(A))
        e += c.T
    end
    e = t
    while e < PS * L_MAX_DELTA
        Us[e + 1] = zero(A)
        e += c.T
    end
    @synchronize()
    sop, uop = ScanShared(Ss, 0, Val(1), Val(c.BI + 4)), ScanShared(Us, 0, Val(1), Val(c.BI + 4))
    nd = min(c.BI, dv - d0)
    sk, sq, sdo = tok_stride(k), tok_stride(q), tok_stride(dO)
    for cc in 1:nc
        ch = nc + 1 - cc
        c0 = (ch - 1) * L
        nv = min(n - c0, L)
        last = cum[hd, L, ch, b]
        decay = exp(last)
        scan_weights!(ws, @inline(s -> @inbounds exp(last - cum[hd, s + 1, ch, b])), t, L, Val(c.T))
        scan_weights!(es, @inline(s -> @inbounds exp(cum[hd, s + 1, ch, b])), t, L, Val(c.T))
        os = chunk_off(states, ch, hd, b)
        delta_store_rows!(dstates, os, Ss, d0, nd, dv, dk, PS, t, Val(c.T), Val(c.BI))
        x = zero(A)
        e = t
        while e < c.BI * dk
            d, f = e % c.BI, e ÷ c.BI
            if d < nd
                x = muladd(Ss[d + PS * f + 1], A(states[os + d0 + d + dv * f + 1]), x)
            end
            e += c.T
        end
        x = group_reduce(+, red, x, t + 1, Val(c.T))
        if t == 0
            dots[dtile, hd, ch, b] = x * decay
        end
        okt, ou = tok_off(k, kh, c0, b), chunk_off(U, ch, hd, b)
        fk = @inline (m, kk) -> scan_ld(A, k, okt, kk, m, 1, sk, dk, nv) * scan_at(ws, m, L)
        acc = scan_mainloop(scan_zeros(A, Val(c)), nothing, Bs, sop, fk, 0, dk, 0, 0, t, ti, tj,
                            Val(true), Val(false), Val(c))
        acc = scan_scale(acc, 0, 0, ti, tj, Val(c)) do i, j, y
            @inline
            ifelse((i < nd) & (j < L), y + @inbounds(dU[ou + d0 + min(i, nd - 1) + dv * min(j, L - 1) + 1]), zero(A))
        end
        scan_foreach(acc, 0, 0, ti, tj, nd, L, Val(c)) do i, j, y
            @inline
            @inbounds dU[ou + d0 + i + dv * j + 1] = y
            @inbounds Us[i + PS * j + 1] = y
        end
        uu = @inline (i, j, y) -> y * scan_ld(A, U, ou + d0, i, j, 1, dv, nd, L)
        ssd_column_dots!(uu, dpos, 1, acc, red, hd, b, c0, nv, dtile - 1, 0, 0, t, ti, tj, Val(c))
        oq, odo, ow = tok_off(q, kh, c0, b), tok_off(dO, hd, c0, b), chunk_off(w, ch, hd, b)
        for f0 in 0:c2.BJ:(dk - 1)
            fa = @inline (m, kk) -> scan_ld(A, dO, odo + d0, m, kk, 1, sdo, nd, nv) * (scale * scan_at(es, kk, L))
            fq = @inline (m, kk) -> scan_ld(A, q, oq, m, kk, 1, sq, dk, nv)
            acc2 = scan_mainloop(scan_zeros(A, Val(c2)), As2, Bs, fa, fq, 0, L, 0, f0, t, ti2, tj2,
                                 Val(true), Val(true), Val(c2))
            fw = @inline (m, kk) -> -scan_ld(A, w, ow, m, kk, 1, dk, dk, L)
            acc2 = scan_mainloop(acc2, nothing, Bs, uop, fw, 0, L, 0, f0, t, ti2, tj2,
                                 Val(true), Val(true), Val(c2))
            scan_foreach(acc2, 0, f0, ti2, tj2, c2.BI, dk, Val(c2)) do i, f, y
                @inline
                @inbounds Ss[i + PS * f + 1] = muladd(decay, Ss[i + PS * f + 1], y)
            end
        end
        @synchronize()
    end
    if dinitial !== nothing
        delta_store_rows!(dinitial, dv * dk * ((hd - 1) + h * (b - 1)), Ss, d0, nd, dv, dk, PS, t, Val(c.T), Val(c.BI))
    end
end

# The rest of the gradient, chunk by chunk, in phases that leave their results in global
# memory for the next: dW, dG, dT (into `s1`), Y = dT Tᵀ (into `s2`), dA = -Tᵀ Y with dKK
# and the sums of `dpos[5:7, ...]`, then dq and dk of each head of values and dv.
@kernel inbounds=true cpu=false unsafe_indices=true function delta_chunk_grad_kernel!(
        dq, dk, dv, dW, dG, dKK, s1, s2, dpos, @Const(dU), @Const(dO), @Const(states), @Const(dstates),
        @Const(U), @Const(Tm), @Const(KK), @Const(q), @Const(k), @Const(v), @Const(beta),
        cum::AbstractArray{A}, scale, L::Int, ::Val{c}) where {A,c}
    As = @localmem A (c.PI * c.BK,)
    Bs = @localmem A (c.PJ * c.BK,)
    cs = @localmem A (L_MAX_DELTA,)
    bs = @localmem A (L_MAX_DELTA,)
    es = @localmem A (L_MAX_DELTA,)
    ws = @localmem A (L_MAX_DELTA,)
    red = @localmem A (c.NR * c.BJ,)
    t = @index(Local, Linear) - 1
    ch, hd, b = @index(Group, NTuple)
    Dk, hk, n = size(k, 1), size(k, 2), size(k, 3)
    Dv, h = size(v, 1), size(v, 2)
    kh = (hd - 1) ÷ (h ÷ hk) + 1
    ti, tj = t % c.NR, t ÷ c.NR
    c0 = (ch - 1) * L
    nv = min(n - c0, L)
    last = cum[hd, L, ch, b]
    scan_weights!(cs, @inline(s -> @inbounds cum[hd, s + 1, ch, b]), t, L, Val(c.T))
    scan_weights!(es, @inline(s -> @inbounds exp(cum[hd, s + 1, ch, b])), t, L, Val(c.T))
    scan_weights!(ws, @inline(s -> @inbounds exp(last - cum[hd, s + 1, ch, b])), t, L, Val(c.T))
    scan_weights!(bs, @inline(s -> @inbounds s < nv ? A(beta[hd, c0 + s + 1, b]) : zero(A)), t, L, Val(c.T))
    @synchronize()
    os, odu, ou, ot = chunk_off(states, ch, hd, b), chunk_off(dU, ch, hd, b), chunk_off(U, ch, hd, b), chunk_off(Tm, ch, hd, b)
    odw, odg, okk = chunk_off(dW, ch, hd, b), chunk_off(dG, ch, hd, b), chunk_off(KK, ch, kh, b)
    odo, sdo = tok_off(dO, hd, c0, b), tok_stride(dO)
    ok, sk, oq, sq = tok_off(k, kh, c0, b), tok_stride(k), tok_off(q, kh, c0, b), tok_stride(q)
    ov, sv = tok_off(v, hd, c0, b), tok_stride(v)
    # dW = -S0ᵀ dU
    for i0 in 0:c.BI:(Dk - 1)
        fa = @inline (m, kk) -> scan_ld(A, states, os, kk, m, 1, Dv, Dv, Dk)
        fb = @inline (m, kk) -> scan_ld(A, dU, odu, kk, m, 1, Dv, Dv, L)
        acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, Dv, i0, 0, t, ti, tj, Val(false), Val(false), Val(c))
        scan_foreach(acc, i0, 0, ti, tj, Dk, L, Val(c)) do i, j, x
            @inline
            @inbounds dW[odw + i + Dk * j + 1] = -x
        end
    end
    # dG[j, l] = Σ_d U[d, j] dO[d, l] exp(cum[l] - cum[j]) for j ≤ l
    fa = @inline (m, kk) -> scan_ld(A, U, ou, kk, m, 1, Dv, Dv, L)
    fb = @inline (m, kk) -> scan_ld(A, dO, odo, kk, m, 1, sdo, Dv, nv) * scale
    acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, Dv, 0, 0, t, ti, tj, Val(false), Val(false), Val(c))
    scan_foreach(acc, 0, 0, ti, tj, L, L, Val(c)) do j, l, x
        @inline
        @inbounds dG[odg + j + L * l + 1] = ifelse(j <= l, x * exp(cs[l + 1] - cs[min(j, l) + 1]), zero(A))
    end
    @synchronize()
    # dT[j, i] = Σ_d dU[d, j] v[d, i] beta[i] + Σ_f dW[f, j] k[f, i] beta[i] exp(cum[i])
    fa = @inline (m, kk) -> scan_ld(A, dU, odu, kk, m, 1, Dv, Dv, L)
    fb = @inline (m, kk) -> scan_ld(A, v, ov, kk, m, 1, sv, Dv, nv) * scan_at(bs, m, L)
    acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, Dv, 0, 0, t, ti, tj, Val(false), Val(false), Val(c))
    ga = @inline (m, kk) -> scan_ld(A, dW, odw, kk, m, 1, Dk, Dk, L)
    gb = @inline (m, kk) -> scan_ld(A, k, ok, kk, m, 1, sk, Dk, nv) * scan_at(bs, m, L) * scan_at(es, m, L)
    acc = scan_mainloop(acc, As, Bs, ga, gb, 0, Dk, 0, 0, t, ti, tj, Val(false), Val(false), Val(c))
    o1 = chunk_off(s1, ch, hd, b)
    scan_foreach(acc, 0, 0, ti, tj, L, L, Val(c)) do j, i, x
        @inline
        @inbounds s1[o1 + j + L * i + 1] = x
    end
    @synchronize()
    # Y = dT Tᵀ
    fa = @inline (m, kk) -> scan_ld(A, s1, o1, m, kk, 1, L, L, L)
    fb = @inline (m, kk) -> scan_ld(A, Tm, ot, m, kk, 1, L, L, L)
    acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, L, 0, 0, t, ti, tj, Val(true), Val(true), Val(c))
    o2 = chunk_off(s2, ch, hd, b)
    scan_foreach(acc, 0, 0, ti, tj, L, L, Val(c)) do j, i, x
        @inline
        @inbounds s2[o2 + j + L * i + 1] = x
    end
    @synchronize()
    # dA = -Tᵀ Y, strictly lower
    fa = @inline (m, kk) -> scan_ld(A, Tm, ot, kk, m, 1, L, L, L)
    fb = @inline (m, kk) -> scan_ld(A, s2, o2, kk, m, 1, L, L, L)
    acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, L, 0, 0, t, ti, tj, Val(false), Val(false), Val(c))
    odkk = chunk_off(dKK, ch, hd, b)
    scan_foreach(acc, 0, 0, ti, tj, L, L, Val(c)) do j, i, x
        @inline
        da = ifelse(i < j, -x, zero(A))
        dec = @inbounds exp(cs[j + 1] - cs[min(i, j) + 1])
        kkji = ifelse(i < j, A(@inbounds KK[okk + j + L * i + 1]), zero(A))
        dkk = da * @inbounds(bs[j + 1]) * dec
        @inbounds dKK[odkk + j + L * i + 1] = dkk
        @inbounds s1[o1 + j + L * i + 1] = dkk * kkji
        @inbounds s2[o2 + j + L * i + 1] = da * dec * kkji
    end
    @synchronize()
    if t < nv
        zr, zc, br = zero(A), zero(A), zero(A)
        for i in 0:(L - 1)
            zr += s1[o1 + t + L * i + 1]
            zc += s1[o1 + i + L * t + 1]
            br += s2[o2 + t + L * i + 1]
        end
        dpos[5, 1, hd, c0 + t + 1, b] = zr
        dpos[6, 1, hd, c0 + t + 1, b] = zc
        dpos[7, 1, hd, c0 + t + 1, b] = br
    end
    @synchronize()
    odq, sdq = tok_off(dq, hd, c0, b), tok_stride(dq)
    for i0 in 0:c.BI:(Dk - 1)
        it = i0 ÷ c.BI
        # dq = S0ᵀ dO E + K dG
        fa = @inline (m, kk) -> scan_ld(A, states, os, kk, m, 1, Dv, Dv, Dk)
        fb = @inline (m, kk) -> scan_ld(A, dO, odo, kk, m, 1, sdo, Dv, nv) * (scale * scan_at(es, m, L))
        acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, Dv, i0, 0, t, ti, tj, Val(false), Val(false), Val(c))
        ga = @inline (m, kk) -> scan_ld(A, k, ok, m, kk, 1, sk, Dk, nv)
        gb = @inline (m, kk) -> scan_ld(A, dG, odg, kk, m, 1, L, L, L)
        acc = scan_mainloop(acc, As, Bs, ga, gb, 0, L, i0, 0, t, ti, tj, Val(true), Val(false), Val(c))
        scan_foreach(acc, i0, 0, ti, tj, Dk, nv, Val(c)) do f, l, x
            @inline
            @inbounds dq[odq + f + sdq * l + 1] = x
        end
        # dk = Q dGᵀ + dS1ᵀ U F + K (dKK + dKKᵀ) + dW T B E
        fa = @inline (m, kk) -> scan_ld(A, q, oq, m, kk, 1, sq, Dk, nv)
        fb = @inline (m, kk) -> scan_ld(A, dG, odg, m, kk, 1, L, L, L)
        acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, L, i0, 0, t, ti, tj, Val(true), Val(true), Val(c))
        fa = @inline (m, kk) -> scan_ld(A, dstates, os, kk, m, 1, Dv, Dv, Dk)
        fb = @inline (m, kk) -> scan_ld(A, U, ou, kk, m, 1, Dv, Dv, L) * scan_at(ws, m, L)
        acc = scan_mainloop(acc, As, Bs, fa, fb, 0, Dv, i0, 0, t, ti, tj, Val(false), Val(false), Val(c))
        fa = @inline (m, kk) -> scan_ld(A, k, ok, m, kk, 1, sk, Dk, nv)
        fb = @inline (m, kk) -> scan_ld(A, dKK, odkk, kk, m, 1, L, L, L) + scan_ld(A, dKK, odkk, m, kk, 1, L, L, L)
        acc = scan_mainloop(acc, As, Bs, fa, fb, 0, L, i0, 0, t, ti, tj, Val(true), Val(false), Val(c))
        fa = @inline (m, kk) -> scan_ld(A, dW, odw, m, kk, 1, Dk, Dk, L)
        fb = @inline (m, kk) -> scan_ld(A, Tm, ot, kk, m, 1, L, L, L)
        accw = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, L, i0, 0, t, ti, tj, Val(true), Val(false), Val(c))
        kw = @inline (f, i, x) -> x * scan_ld(A, k, ok, f, i, 1, sk, Dk, nv)
        ssd_column_dots!(kw, dpos, 3, accw, red, hd, b, c0, nv, it, i0, 0, t, ti, tj, Val(c))
        accw = scan_scale(accw, i0, 0, ti, tj, Val(c)) do f, i, x
            @inline
            x * scan_at(bs, i, L) * scan_at(es, i, L)
        end
        scan_foreach(map(+, acc, accw), i0, 0, ti, tj, Dk, nv, Val(c)) do f, i, x
            @inline
            @inbounds dk[odq + f + sdq * i + 1] = x
        end
    end
    # dv = dU T B
    for i0 in 0:c.BI:(Dv - 1)
        fa = @inline (m, kk) -> scan_ld(A, dU, odu, m, kk, 1, Dv, Dv, L)
        fb = @inline (m, kk) -> scan_ld(A, Tm, ot, kk, m, 1, L, L, L)
        acc = scan_mainloop(scan_zeros(A, Val(c)), As, Bs, fa, fb, 0, L, i0, 0, t, ti, tj, Val(true), Val(false), Val(c))
        vu = @inline (d, i, x) -> x * scan_ld(A, v, ov, d, i, 1, sv, Dv, nv)
        ssd_column_dots!(vu, dpos, 4, acc, red, hd, b, c0, nv, i0 ÷ c.BI, i0, 0, t, ti, tj, Val(c))
        scan_foreach(acc, i0, 0, ti, tj, Dv, nv, Val(c)) do d, i, x
            @inline
            @inbounds dv[ov + d + sv * i + 1] = x * bs[i + 1]
        end
    end
end

function ∇delta_kernels(dO, dfinal, o, q, k, v, g::AbstractArray{T}, beta, scale, state, L, saved) where {T}
    Dk, hk, n, b = size(q)
    Dv, h = size(v, 1), size(v, 2)
    (; cum, QK, KK, Tm, U, W, states) = saved
    nc = size(cum, 3)
    backend = get_backend(q)
    c = scan_tiles()
    nt = max(cld(Dv, c.BI), cld(Dk, c.BI))
    dpos = fill!(similar(U, 7, nt, h, n, b), 0)
    dU = similar(U)
    delta_dul_kernel!(backend, c.T)(dU, dpos, dO, U, QK, cum, scale, L, Val(c);
        ndrange=(c.T * cld(Dv, c.BI), nc, h * b))
    cr, c2, dkp = delta_recurrence_tiling(Dk, attention_shmem_budget(backend) ÷ sizeof(T); reverse=true)
    ndt = cld(Dv, cr.BI)
    dpos1 = fill!(similar(U, 1, ndt, h, n, b), 0)
    dstates = similar(states)
    dinitial = state === nothing ? nothing : similar(states, Dv, Dk, h, b)
    dots = similar(U, ndt, h, nc, b)
    delta_reverse_kernel!(backend, cr.T)(dU, dstates, dinitial, dots, dpos1, dfinal, dO, states, U, W, q, k,
        cum, scale, L, Val(dkp), Val(cr), Val(c2); ndrange=(cr.T * ndt, h, b))
    dqh, dkh, dv = similar(q, T, Dk, h, n, b), similar(q, T, Dk, h, n, b), similar(v)
    dW, dG, dKK, s1, s2 = similar(W), similar(Tm), similar(Tm), similar(Tm), similar(Tm)
    delta_chunk_grad_kernel!(backend, c.T)(dqh, dkh, dv, dW, dG, dKK, s1, s2, dpos, dU, dO, states, dstates,
        U, Tm, KK, q, k, v, beta, cum, scale, L, Val(c); ndrange=(c.T * nc, h, b))
    p = sum(dpos; dims=2)
    pos(i) = reshape(p[i:i, :, :, :, :], h, n, b)
    uu, uul, kwt, vut, zr, zc, ba = pos(1) .+ reshape(sum(dpos1; dims=2), h, n, b), pos(2), pos(3), pos(4), pos(5), pos(6), pos(7)
    ecum = reshape(cum, h, :, b)[:, 1:n, :]
    edc = exp.(ecum)
    dcum = reshape(sum(dO .* o; dims=1), h, n, b) .- uu .+ beta .* edc .* kwt .+ zr .- zc
    dlast = reshape(sum(dots; dims=1), h, nc, b) .+
            reshape(sum(reshape(pad_tokens(uu .- uul, L), h, L, nc, b); dims=2), h, nc, b)
    dg = reverse_chunk_cumsum(dcum, L) .+ repeat_chunks(dlast, L, n)
    dbeta = vut .+ edc .* kwt .+ ba
    r = h ÷ hk
    dq = r == 1 ? dqh : reshape(sum(reshape(dqh, Dk, r, hk, n, b); dims=2), Dk, hk, n, b)
    dk = r == 1 ? dkh : reshape(sum(reshape(dkh, Dk, r, hk, n, b); dims=2), Dk, hk, n, b)
    return dq, dk, dv, dg, dbeta, dinitial === nothing ? NoTangent() : dinitial
end

function rrule(::typeof(delta_kernels), q, k, v, g::AbstractArray{T}, beta, scale, state, L) where {T}
    o, final, saved = delta_kernels_forward(q, k, v, g, beta, scale, state, L, T, Val(true))
    out = eltype(o) == eltype(v) ? o : eltype(v).(o)
    function delta_kernels_pullback(Δ)
        Δ = unthunk(Δ)
        Δ isa AbstractZero && return (NoTangent(), ntuple(_ -> Δ, 5)..., NoTangent(), Δ, NoTangent())
        dO = scan_cotangent(o, T, unthunk(Δ[1]))
        dO = dO === nothing ? fill!(similar(o), 0) : dO
        dfinal = scan_cotangent(final, T, unthunk(Δ[2]))
        dq, dk, dv, dg, dbeta, dstate = ∇delta_kernels(dO, dfinal, o, q, k, v, g, beta, scale, state, L, saved)
        return NoTangent(), dq, dk, dv, dg, dbeta, NoTangent(), dstate, NoTangent()
    end
    return (out, final), delta_kernels_pullback
end

function delta_kernels_apply(q, k, v, g, beta, state)
    scan_kernels_apply(q, k, v, g, beta, state) || return false
    backend = get_backend(q)
    T = promote_type(Float32, eltype(q), eltype(k), eltype(v), eltype(g), eltype(beta))
    budget = attention_shmem_budget(backend) ÷ sizeof(T)
    return delta_prepare_shmem(scan_tiles()) <= budget &&
           delta_recurrence_tiling(size(k, 1), budget; reverse=true) !== nothing
end

@non_differentiable delta_kernels_apply(::Any...)

# Whether the kernels compute with these arrays (or `nothing`s): arrays of floating-point
# numbers on one GPU backend, which the kernels take dense (`scan_dense`).
function scan_kernels_apply(x, arrays...)
    all(a -> a === nothing || (a isa AnyGPUArray && eltype(a) <: FusedAttentionFloat), (x, arrays...)) || return false
    backend = get_backend(x)
    return all(a -> a === nothing || get_backend(a) == backend, arrays)
end

@non_differentiable scan_kernels_apply(::Any...)

scan_dense(x) = x isa DenseArray || x === nothing ? x : copy(x)

# Whether the kernels can be called where this is: outside of automatic differentiation,
# or under ChainRules, which has the rule below.
scan_kernels_differentiable(x) = !within_gradient(x)

rrule(::typeof(scan_kernels_differentiable), x) = true, _ -> (NoTangent(), NoTangent())

# A cotangent the kernels can index, or `nothing` for a zero one.
scan_cotangent(x, ::Type{T}, Δ::AbstractZero) where {T} = nothing
scan_cotangent(x::AbstractGPUArray, ::Type{T}, Δ::AbstractGPUArray{<:Real}) where {T} = Δ
scan_cotangent(x, ::Type{T}, Δ) where {T} = copyto!(similar(x, T, size(Δ)), unthunk(Δ))

function rrule(::typeof(ssd_kernels), x, dt::AbstractArray{T}, A, B, C, D, state, L) where {T}
    y, final, saved = ssd_kernels_forward(x, dt, A, B, C, D, state, L, T)
    out = eltype(y) == eltype(x) ? y : eltype(x).(y)
    function ssd_kernels_pullback(Δ)
        Δ = unthunk(Δ)
        Δ isa AbstractZero && return (NoTangent(), ntuple(_ -> Δ, 7)..., NoTangent())
        dy = scan_cotangent(y, T, unthunk(Δ[1]))
        dy = dy === nothing ? fill!(similar(y), 0) : dy
        dfinal = scan_cotangent(final, T, unthunk(Δ[2]))
        dx, ddt, dA, dB, dC, dD, dstate = ∇ssd_kernels(dy, dfinal, y, x, dt, A, B, C, D, state, L, saved)
        return NoTangent(), dx, ddt, dA, dB, dC, dD, dstate, NoTangent()
    end
    return (out, final), ssd_kernels_pullback
end


#######################
# Decoding
#######################

# One token, as when decoding: a block computes `STEP_ROWS` rows of the state of a head,
# which it reads and writes once. Its thread `(i, j)` holds the elements `j + STEP_PARTS * m`
# of the row `i`, and the sums along the rows add the parts through shared memory.
const STEP_ROWS, STEP_PARTS = 32, 8

# the longest rows of the states (`state_size`, `head_k_dim`) the step kernels take
const STEP_MAX_COLUMNS = 256

step_kernels_apply(ncolumns, arrays...) = ncolumns <= STEP_MAX_COLUMNS && scan_kernels_apply(arrays...)

@non_differentiable step_kernels_apply(::Any...)

# whether the token of the batch element `b` is past the end of its sequence
@inline step_masked(::Nothing, b) = false
@inline step_masked(len, b) = @inbounds(len[b]) == 0

# the elements `j + STEP_PARTS * m` of a row of `n`, at `x[o + st * f + 1]` for the element
# `f`, in the type `A`, zero past `n` (and for `x = nothing`)
@inline step_row(::Type{A}, x, o, st, j, n, ::Val{NF}) where {A,NF} =
    ntuple(Val(NF)) do m
        @inline
        f = j + STEP_PARTS * (m - 1)
        v = A(@inbounds x[o + st * min(f, n - 1) + 1])
        ifelse(f < n, v, zero(A))
    end
@inline step_row(::Type{A}, ::Nothing, o, st, j, n, ::Val{NF}) where {A,NF} = ntuple(_ -> zero(A), Val(NF))

@inline function step_store!(x, vals::NTuple{NF}, o, st, j, n) where {NF}
    ntuple(Val(NF)) do m
        @inline
        f = j + STEP_PARTS * (m - 1)
        f < n && (@inbounds x[o + st * f + 1] = vals[m])
        nothing
    end
    return nothing
end

# (`map` and `sum` of tuples of 32 elements or more are not unrolled)
@generated function step_dot(a::NTuple{NF,A}, b::NTuple{NF,A}) where {NF,A}
    ex = :(zero(A))
    for m in 1:NF
        ex = :(muladd(a[$m], b[$m], $ex))
    end
    return Expr(:block, Expr(:meta, :inline), ex)
end

# the sums of the values `v` over the parts of the row `i`, on each of its threads
@inline function step_sum(red, v::NTuple{NV,A}, t, i) where {NV,A}
    ntuple(Val(NV)) do q
        @inline
        @inbounds red[t + STEP_ROWS * STEP_PARTS * (q - 1) + 1] = v[q]
        nothing
    end
    @synchronize()
    return ntuple(Val(NV)) do q
        @inline
        s = zero(A)
        for r in 0:(STEP_PARTS - 1)
            s += @inbounds red[i + STEP_ROWS * r + STEP_ROWS * STEP_PARTS * (q - 1) + 1]
        end
        s
    end
end

@kernel inbounds=true cpu=false unsafe_indices=true function ssd_step_kernel!(
        y, final::AbstractArray{A}, @Const(initial), @Const(x), @Const(dt), @Const(Ah), @Const(B),
        @Const(C), D, len, ::Val{NF}) where {A,NF}
    red = @localmem A (STEP_ROWS * STEP_PARTS,)
    t = @index(Local, Linear) - 1
    rb, hd, b = @index(Group, NTuple)
    P, h = size(x, 1), size(x, 2)
    N, g = size(B, 1), size(B, 2)
    gg = (hd - 1) ÷ (h ÷ g) + 1
    i, j = t % STEP_ROWS, t ÷ STEP_ROWS
    p = (rb - 1) * STEP_ROWS + i
    pc = min(p, P - 1)
    Δ = ifelse(step_masked(len, b), zero(A), A(dt[hd, 1, b]))
    xp = A(x[pc + 1, hd, 1, b])
    os, ob = P * N * ((hd - 1) + h * (b - 1)) + pc, N * ((gg - 1) + g * (b - 1))
    s0 = step_row(A, initial, os, P, j, N, Val(NF))
    Bv, Cv = step_row(A, B, ob, 1, j, N, Val(NF)), step_row(A, C, ob, 1, j, N, Val(NF))
    decay, w = exp(Δ * A(Ah[hd])), Δ * xp
    s = ntuple(m -> muladd(decay, s0[m], w * Bv[m]), Val(NF))
    yp, = step_sum(red, (step_dot(s, Cv),), t, i)
    if p < P
        if j == 0
            y[p + 1, hd, 1, b] = D === nothing ? yp : muladd(A(D[hd]), xp, yp)
        end
        step_store!(final, s, os, P, j, N)
    end
end

# `ssd_scan` of one token of dense arrays, with `final` of the compute type `T`
function ssd_step_kernel(::Type{T}, x, dt, A, B, C, D, state, lengths) where {T}
    P, h, _, b = size(x)
    N = size(B, 1)
    y, final = similar(x), similar(x, T, P, N, h, b)
    len = lengths === nothing ? nothing : sequence_lengths(x, lengths, 1, b)
    ssd_step_kernel!(get_backend(x), (STEP_ROWS * STEP_PARTS, 1, 1))(y, final, state, x, dt, A, B, C, D, len,
        Val(cld(N, STEP_PARTS)); ndrange=(STEP_ROWS * STEP_PARTS * cld(P, STEP_ROWS), h, b))
    return y, final
end

# With `S` the state before the token, decayed by `e = exp(g)`, and the write
# `u = beta * (v - e * S k)`, the state after it is `e * S + u kᵀ` and the output
# `scale * (e * S q + (k ⋅ q) * u)`.
@kernel inbounds=true cpu=false unsafe_indices=true function delta_step_kernel!(
        o, final::AbstractArray{A}, @Const(initial), @Const(q), @Const(k), @Const(v), @Const(g),
        @Const(beta), scale, len, ::Val{NF}) where {A,NF}
    red = @localmem A (3 * STEP_ROWS * STEP_PARTS,)
    t = @index(Local, Linear) - 1
    rb, hd, b = @index(Group, NTuple)
    dk, hk = size(k, 1), size(k, 2)
    dv, h = size(v, 1), size(v, 2)
    kh = (hd - 1) ÷ (h ÷ hk) + 1
    i, j = t % STEP_ROWS, t ÷ STEP_ROWS
    d = (rb - 1) * STEP_ROWS + i
    dc = min(d, dv - 1)
    masked = step_masked(len, b)
    e = ifelse(masked, one(A), exp(A(g[hd, 1, b])))
    β = ifelse(masked, zero(A), A(beta[hd, 1, b]))
    os, ok = dv * dk * ((hd - 1) + h * (b - 1)) + dc, dk * ((kh - 1) + hk * (b - 1))
    s = step_row(A, initial, os, dv, j, dk, Val(NF))
    kv, qv = step_row(A, k, ok, 1, j, dk, Val(NF)), step_row(A, q, ok, 1, j, dk, Val(NF))
    sk, sq, kq = step_sum(red, (step_dot(s, kv), step_dot(s, qv), step_dot(kv, qv)), t, i)
    u = β * (A(v[dc + 1, hd, 1, b]) - e * sk)
    if d < dv
        if j == 0
            o[d + 1, hd, 1, b] = scale * muladd(e, sq, kq * u)
        end
        step_store!(final, ntuple(m -> muladd(e, s[m], u * kv[m]), Val(NF)), os, dv, j, dk)
    end
end

# `gated_delta_rule` of one token of dense arrays, with `final` of the compute type of `scale`
function delta_step_kernel(q, k, v, g, beta, scale::T, state, lengths) where {T}
    dk, _, _, b = size(q)
    dv, h = size(v, 1), size(v, 2)
    o, final = similar(v), similar(v, T, dv, dk, h, b)
    len = lengths === nothing ? nothing : sequence_lengths(q, lengths, 1, b)
    delta_step_kernel!(get_backend(q), (STEP_ROWS * STEP_PARTS, 1, 1))(o, final, state, q, k, v, g, beta,
        scale, len, Val(cld(dk, STEP_PARTS)); ndrange=(STEP_ROWS * STEP_PARTS * cld(dv, STEP_ROWS), h, b))
    return o, final
end

# `causal_conv1d` of one token, a thread for each channel: its inputs `[state x]`
@kernel inbounds=true cpu=false function causal_conv_step_kernel!(
        y::AbstractArray{Y}, new_state::AbstractArray{S}, @Const(x), @Const(w), bias, @Const(state), len,
        ::Val{K}) where {Y,S,K}
    c, b = @index(Global, NTuple)
    xs = ntuple(Val(K)) do j
        @inline
        j < K ? (state === nothing ? zero(S) : S(state[c, j, b])) : S(x[c, 1, b])
    end
    yc = sum(ntuple(j -> Y(w[j, c]) * Y(xs[j]), Val(K)))
    y[c, 1, b] = bias === nothing ? yc : yc + Y(bias[c])
    shift = !step_masked(len, b)
    ntuple(Val(K - 1)) do j
        @inline
        new_state[c, j, b] = ifelse(shift, xs[j + 1], xs[j])
        nothing
    end
end

function causal_conv_step(x, w, bias, state, lengths)
    C, _, b = size(x)
    K = size(w, 1)
    S = state === nothing ? eltype(x) : promote_type(eltype(x), eltype(state))
    Y = promote_type(S, eltype(w), bias === nothing ? S : eltype(bias))
    y, new_state = similar(x, Y, C, 1, b), similar(x, S, C, K - 1, b)
    len = lengths === nothing ? nothing : sequence_lengths(x, lengths, 1, b)
    causal_conv_step_kernel!(get_backend(x), (256, 1))(y, new_state, x, w, bias, state, len, Val(K);
        ndrange=(C, b))
    return y, new_state
end
