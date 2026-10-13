# Gradient of the fused scaled dot-product attention.
#
# A query `i` has the logits `s[j] = scale * ⟨q[i], k[j]⟩ + bias[j, i]`, the weights
# `p = softmax(s)` and the output `o[i] = Σ_j p[j] * v[j]`. A cotangent `do[i]` of the
# output then gives
#
#   dv[j] += p[j] * do[i]
#   ds[j]  = p[j] * (⟨do[i], v[j]⟩ - ⟨do[i], o[i]⟩)
#   dq[i] += scale * ds[j] * k[j]
#   dk[j] += scale * ds[j] * q[i]
#
# The weights are recomputed tile by tile from the `lse` of each query, which the
# forward pass saves, so that here too nothing of size `kv_len × q_len` is formed. The
# result does not depend on how the work is scheduled: no update is atomic.

# The gradients of `fused_attention` with respect to `q`, `k`, `v` and `scale`, for the
# cotangent `dout` of its output `o`.
function fused_attention_grad(dout, o, lse, q::AA4{T}, k::AA4{T}, v::AA4{T}, bias, mask, scale,
                              is_causal::Bool, q_lengths, kv_lengths) where {T}
    A = attention_compute_type(T)
    s = attention_scale(A, q, scale)
    # `dq` is left without its factor `scale` when the gradient of the latter is needed
    dq, dk, dv = fused_attention_grad(get_backend(q), dout, o, lse, q, k, v, attention_pad4(bias),
                                      attention_pad4(mask), attention_vec(q_lengths),
                                      attention_vec(kv_lengths), s, scale === nothing ? s : one(A),
                                      is_causal)
    dscale = NoTangent()
    if scale !== nothing
        dscale = mapreduce(attention_wide_prod, +, q, dq)
        dq .*= s
    end
    return dq, dk, dv, dscale
end

@inline attention_wide(x::T) where {T} = attention_compute_type(T)(x)
@inline attention_wide_prod(x, y) = attention_wide(x) * attention_wide(y)

# A cotangent the kernels can index: on the CPU any array, as the `Fill` of a `sum`.
attention_cotangent(o::Array, dout::AbstractArray{<:Real,4}) = dout
attention_cotangent(o::AbstractGPUArray, dout::AbstractGPUArray{<:Real,4}) = dout
attention_cotangent(o, dout) = copyto!(similar(o), dout)

function rrule(::typeof(fused_attention), q::AA4{T}, k::AA4{T}, v::AA4{T}, bias, mask, scale,
               is_causal::Bool, q_lengths, kv_lengths) where {T}
    bias === nothing || throw(ArgumentError(
        "The fused attention has no gradient with respect to a bias: differentiate the \
         materialized implementation, as `scaled_dot_product_attention` does."))
    lse = similar(q, attention_compute_type(T), size(q, 2), size(q, 3), size(q, 4))
    o = fused_attention(q, k, v, bias, mask, scale, is_causal, q_lengths, kv_lengths; lse)
    function fused_attention_pullback(Δ)
        Δ = unthunk(Δ)
        Δ isa AbstractZero &&
            return (NoTangent(), Δ, Δ, Δ, NoTangent(), NoTangent(), Δ, NoTangent(), NoTangent(), NoTangent())
        dq, dk, dv, dscale = fused_attention_grad(attention_cotangent(o, Δ), o, lse, q, k, v, bias,
                                                  mask, scale, is_causal, q_lengths, kv_lengths)
        return (NoTangent(), dq, dk, dv, NoTangent(), NoTangent(), dscale, NoTangent(), NoTangent(), NoTangent())
    end
    return o, fused_attention_pullback
end

# Whether `fused_attention` can be called where this is: outside of automatic
# differentiation, or under a backend with a rule for it, that is ChainRules, above, and
# Enzyme, in its extension. The rules have no gradient with respect to a bias.
fused_attention_differentiable(q, bias) =
    !within_gradient(q) || (bias === nothing && _within_autodiff(Val(:Enzyme)))

rrule(::typeof(fused_attention_differentiable), q, bias) =
    bias === nothing, _ -> (NoTangent(), NoTangent(), NoTangent())


#######################
# CPU
#######################

# A range of queries and a range of keys make a pair, which adds to the gradients of
# both. Two pairs with a range in common would race, so the pairs are taken in rounds:
# those of a round share no range, and run in parallel. A pair is visited once, and
# gives its part of the three gradients from the same weights.

# The pairs of a round: its `x`-th one, counting from zero, is the range `t` of queries
# with the range `u` of keys.
@inline attention_pair(x, round, nrq, nrk) =
    nrq <= nrk ? (x, (x + round) % nrk) : ((x + round) % nrq, x)

# Four queries go through a packed tile of keys together. For each of them the logits
# `s` become `ds` and `g[j] = ⟨do, v[j]⟩` becomes the weights, over the `n` keys the
# four are computed for, of which the query attends to the first `nvalid`.
function attention_grad_weights!(s::Vector{A}, g::Vector{A}, n, nvalid, bias, mask, j0, i, h, b,
                                 lse::A, delta::A) where {A}
    attention_bias!(s, nvalid, bias, j0, i, h, b)
    attention_mask!(s, nvalid, mask, j0, i, h, b)
    @inbounds for j in (max(nvalid, 0) + 1):n
        s[j] = A(-Inf)
    end
    low = attention_exp_floor(A)
    @inbounds @simd for j in 1:n
        x = s[j] - lse
        p = ifelse(x < low, zero(A), attention_exp(max(x, low)))
        s[j] = p * (g[j] - delta)
        g[j] = p
    end
    return nothing
end

# xt[j, d] += Σ_c w[c][j] * a[d, c]
function attention_rank!(xt::Matrix{A}, w::NTuple{4,Vector{A}}, n, a::Matrix{A}) where {A}
    w1, w2, w3, w4 = w
    Dx = size(xt, 2)
    d = 1
    @inbounds while d + 3 <= Dx
        a1 = (a[d, 1], a[d + 1, 1], a[d + 2, 1], a[d + 3, 1])
        a2 = (a[d, 2], a[d + 1, 2], a[d + 2, 2], a[d + 3, 2])
        a3 = (a[d, 3], a[d + 1, 3], a[d + 2, 3], a[d + 3, 3])
        a4 = (a[d, 4], a[d + 1, 4], a[d + 2, 4], a[d + 3, 4])
        @simd ivdep for j in 1:n
            x1, x2, x3, x4 = w1[j], w2[j], w3[j], w4[j]
            xt[j, d] = muladd(x4, a4[1], muladd(x3, a3[1], muladd(x2, a2[1], muladd(x1, a1[1], xt[j, d]))))
            xt[j, d + 1] = muladd(x4, a4[2], muladd(x3, a3[2], muladd(x2, a2[2], muladd(x1, a1[2], xt[j, d + 1]))))
            xt[j, d + 2] = muladd(x4, a4[3], muladd(x3, a3[3], muladd(x2, a2[3], muladd(x1, a1[3], xt[j, d + 2]))))
            xt[j, d + 3] = muladd(x4, a4[4], muladd(x3, a3[4], muladd(x2, a2[4], muladd(x1, a1[4], xt[j, d + 3]))))
        end
        d += 4
    end
    @inbounds while d <= Dx
        b1, b2, b3, b4 = a[d, 1], a[d, 2], a[d, 3], a[d, 4]
        @simd ivdep for j in 1:n
            xt[j, d] = muladd(w4[j], b4, muladd(w3[j], b3, muladd(w2[j], b2, muladd(w1[j], b1, xt[j, d]))))
        end
        d += 1
    end
    return nothing
end

# key by key, for the reason `attention_pack!` goes through a buffer
function attention_unpack_add!(dx::AbstractArray{A,4}, xt::Matrix{A}, h, j0, n, b) where {A}
    @inbounds for j in 1:n
        @simd ivdep for d in axes(dx, 1)
            dx[d, h, j0 + j - 1, b] += xt[j, d]
        end
    end
    return nothing
end

function attention_grad_workspace(::Type{A}, D::Int, Dv::Int, nq::Int) where {A}
    nk = attention_keys_per_tile(max(D, Dv))
    vecs() = ntuple(_ -> Vector{A}(undef, nk), 4)
    return (; kt = Matrix{A}(undef, nk, D), vt = Matrix{A}(undef, nk, Dv),
              buf = Matrix{A}(undef, max(D, Dv), nk),
              dkt = Matrix{A}(undef, nk, D), dvt = Matrix{A}(undef, nk, Dv),
              s = vecs(), g = vecs(), qs = Matrix{A}(undef, D, 4), dos = Matrix{A}(undef, Dv, 4),
              # three slots more than the queries: an incomplete four is filled with idle ones
              dqacc = Matrix{A}(undef, D, nq + 3), delta = Vector{A}(undef, nq))
end

# The slot `c` of a four: the query `i` if it takes part in the gradient, else idle.
function attention_grad_query!(s::Vector{A}, g::Vector{A}, qs, dos, c, n, nvalid, active::Bool,
                               q, dout, bias, mask, j0, i, h, b, scale::A, lse, delta) where {A}
    @inbounds if active
        attention_grad_weights!(s, g, n, nvalid, bias, mask, j0, i, h, b, lse, delta)
        for d in axes(qs, 1)
            qs[d, c] = A(q[d, h, i, b]) * scale
        end
        for d in axes(dos, 1)
            dos[d, c] = A(dout[d, h, i, b])
        end
    else
        for j in 1:n
            s[j] = zero(A)
            g[j] = zero(A)
        end
        for d in axes(qs, 1)
            qs[d, c] = zero(A)
        end
        for d in axes(dos, 1)
            dos[d, c] = zero(A)
        end
    end
    return nothing
end

# The pair of the queries `i0:i1` and the keys `ja:jb`, in the head `h` of the batch
# element `b`.
function attention_grad_item!(dq, dk, dv, dout, o, lse, q, k, v, bias, mask, q_lengths, kv_lengths,
                              scale::A, qscale::A, is_causal::Bool, h, b, i0, i1, ja, jb, ws) where {A}
    (; kt, vt, buf, dkt, dvt, s, g, qs, dos, dqacc, delta) = ws
    hk = (h - 1) ÷ (size(q, 2) ÷ size(k, 2)) + 1
    # the output is zero at the padded queries
    i1 = min(i1, attention_len(q_lengths, b, size(q, 3)))
    jmax = min(jb, attention_len(kv_lengths, b, size(k, 3)))
    is_causal && (jmax = min(jmax, i1))
    (i0 > i1 || ja > jmax) && return nothing
    nk = size(kt, 1)
    fill!(dqacc, zero(A))
    @inbounds for i in i0:i1
        x = zero(A)
        @simd for d in axes(o, 1)
            x = muladd(A(dout[d, h, i, b]), A(o[d, h, i, b]), x)
        end
        delta[i - i0 + 1] = x
    end
    j0 = ja
    while j0 <= jmax
        nt = min(nk, jmax - j0 + 1)
        attention_pack!(kt, buf, k, hk, j0, nt, b)
        attention_pack!(vt, buf, v, hk, j0, nt, b)
        fill!(dkt, zero(A))
        fill!(dvt, zero(A))
        for i in (is_causal ? max(i0, j0) : i0):4:i1
            i2, i3, i4 = min(i + 1, i1), min(i + 2, i1), min(i + 3, i1)
            # with a causal mask the earlier queries of the four attend to fewer keys
            n = is_causal ? min(nt, i4 - j0 + 1) : nt
            attention_logits!(s[1], s[2], kt, n, q, h, i, i2, b, scale)
            attention_logits!(s[3], s[4], kt, n, q, h, i3, i4, b, scale)
            attention_logits!(g[1], g[2], vt, n, dout, h, i, i2, b, one(A))
            attention_logits!(g[3], g[4], vt, n, dout, h, i3, i4, b, one(A))
            for c in 1:4
                ic = min(i + c - 1, i1)
                # a query attending to no key, whose output is NaN, takes no part
                active = (i + c - 1 <= i1) && @inbounds(lse[h, ic, b]) != A(-Inf)
                attention_grad_query!(s[c], g[c], qs, dos, c, n, is_causal ? min(nt, ic - j0 + 1) : nt,
                                      active, q, dout, bias, mask, j0, ic, h, b, scale,
                                      @inbounds(lse[h, ic, b]), @inbounds(delta[ic - i0 + 1]))
            end
            ii = i - i0 + 1
            attention_outputs!(dqacc, ii, s[1], s[2], kt, n)
            attention_outputs!(dqacc, ii + 2, s[3], s[4], kt, n)
            attention_rank!(dkt, s, n, qs)
            attention_rank!(dvt, g, n, dos)
        end
        attention_unpack_add!(dk, dkt, hk, j0, nt, b)
        attention_unpack_add!(dv, dvt, hk, j0, nt, b)
        j0 += nt
    end
    @inbounds for i in i0:i1
        @simd for d in axes(dq, 1)
            dq[d, h, i, b] += dqacc[d, i - i0 + 1] * qscale
        end
    end
    return nothing
end

# The pairs of a round, for the key/value head `hk` of each batch element: `next` hands
# them out to the tasks.
function attention_grad_round!(next, round, ws, dq, dk, dv, dout, o, lse, q, k, v, bias, mask,
                               q_lengths, kv_lengths, scale, qscale, is_causal, (; rq, rk, nrq, nrk, nq))
    Lq, Lk, Hk = size(q, 3), size(k, 3), size(k, 2)
    r = size(q, 2) ÷ Hk
    nx = min(nrq, nrk)
    nitems = nx * Hk * size(q, 4)
    while true
        item = Threads.atomic_add!(next, 1)
        item > nitems && break
        t, u = attention_pair((item - 1) % nx, round, nrq, nrk)
        hk = ((item - 1) ÷ nx) % Hk + 1
        b = (item - 1) ÷ (nx * Hk) + 1
        ia, ib = t * rq + 1, min((t + 1) * rq, Lq)
        ja, jb = u * rk + 1, min((u + 1) * rk, Lk)
        is_causal && ja > ib && continue
        # the query heads sharing these keys, and the queries in tiles
        for h in ((hk - 1) * r + 1):(hk * r), i0 in ia:nq:ib
            attention_grad_item!(dq, dk, dv, dout, o, lse, q, k, v, bias, mask, q_lengths, kv_lengths,
                                 scale, qscale, is_causal, h, b, i0, min(i0 + nq - 1, ib), ja, jb, ws)
        end
    end
    return nothing
end

function fused_attention_grad(::CPU, dout, o, lse, q::AA4{T}, k, v, bias, mask, q_lengths,
                              kv_lengths, scale::A, qscale::A, is_causal::Bool) where {T,A}
    D, Dv, Lq, Lk = size(q, 1), size(v, 1), size(q, 3), size(k, 3)
    nheads = size(k, 2) * size(q, 4)                # key/value heads, over the batch
    work = Lq * Lk * size(q, 2) * size(q, 4) * (2 * D + 3 * Dv)
    ntasks = should_use_spawn() ? min(Threads.nthreads(:default), cld(work, 2^17)) : 1
    dq, dk, dv = map(x -> attention_zeros(A, x, ntasks), (q, k, v))
    nk = attention_keys_per_tile(max(D, Dv))
    # Ranges on each side: one if the heads alone balance the tasks, since every round
    # spawns them anew.
    nr = ntasks == 1 ? 1 : clamp(cld(4 * ntasks, nheads), 1, min(cld(Lq, 32), cld(Lk, nk)))
    rq = cld(Lq, nr)
    rk = nk * cld(cld(Lk, nr), nk)
    sched = (; rq, rk, nrq = cld(Lq, rq), nrk = cld(Lk, rk), nq = min(rq, 512))
    ntasks = min(ntasks, min(sched.nrq, sched.nrk) * nheads)
    wss = [attention_grad_workspace(A, D, Dv, sched.nq) for _ in 1:ntasks]
    for round in 0:(max(sched.nrq, sched.nrk) - 1)
        next = Threads.Atomic{Int}(1)
        run(ws) = attention_grad_round!(next, round, ws, dq, dk, dv, dout, o, lse, q, k, v, bias, mask,
                                        q_lengths, kv_lengths, scale, qscale, is_causal, sched)
        ntasks == 1 ? run(wss[1]) : foreach(wait, [Threads.@spawn(run(ws)) for ws in wss])
    end
    return attention_narrow(T, dq), attention_narrow(T, dk), attention_narrow(T, dv)
end

attention_narrow(::Type{T}, x::AbstractArray{T}) where {T} = x
attention_narrow(::Type{T}, x::AbstractArray) where {T} = T.(x)

# Zeros of the size of `x`, filled by the tasks: a new array is paged in as it is written.
function attention_zeros(::Type{A}, x::Array, ntasks::Int) where {A}
    y = similar(x, A)
    n = cld(length(y), ntasks)
    zero!(t) = fill!(view(y, ((t - 1) * n + 1):min(t * n, length(y))), zero(A))
    if ntasks == 1
        zero!(1)
    else
        foreach(wait, [Threads.@spawn(zero!(t)) for t in 1:ntasks])
    end
    return y
end


#######################
# GPU
#######################

# Three launches of one kernel, which is the forward kernel with a tile of positions
# owned by a block, walked against the tiles of the other side:
#
#   dq   a block owns queries and walks the keys     dQ = scale * K dS
#   dk   a block owns keys and walks the queries     dK = scale * Q dSᵀ
#   dv   a block owns keys and walks the queries     dV = dO Pᵀ
#
# For each pair of tiles the logits S and, but for `dv`, G = dOᵀV are accumulated in the
# registers as the logits of the forward pass are, P = exp(S - lse) or
# dS = P ∘ (G - δ) goes to the shared tile, and its product with the walked side is
# added to the registers that hold the gradient of the owned positions for the whole
# kernel. Nothing is accumulated in global memory, so that each block reads what the
# blocks of its head share, as in the forward pass, and writes its gradient once.
#
# In the kernel `o` stands for the owned side and `w` for the walked one; the shared
# tile has the owned positions as its rows.

# the walked head `g` of an owned one, and the positions to walk
@inline attention_grad_head(::Val{:q}, ho, g, Ho, Hw) = (ho - 1) ÷ (Ho ÷ Hw) + 1
@inline attention_grad_head(::Val{:k}, ho, g, Ho, Hw) = (ho - 1) * (Hw ÷ Ho) + g + 1
@inline attention_grad_range(::Val{:q}, ::Val{causal}, o0, wlen, c) where {causal} =
    (0, causal ? min(wlen, o0 + c.BQ) : wlen)
@inline attention_grad_range(::Val{:k}, ::Val{causal}, o0, wlen, c) where {causal} =
    (causal ? (o0 ÷ c.BK) * c.BK : 0, wlen)

# the logits of a thread, as `attention_masked`
@inline function attention_grad_masked(acc::NTuple{MN,A}, mask, bias, o0, w0, tr, tc, hq, b, omax, wmax,
                                       ::Val{causal}, ::Val{own}, ::Val{c}) where {MN,A,causal,own,c}
    ntuple(Val(MN)) do x
        @inline
        a = o0 + tr + c.NR * ((x - 1) % c.TM)
        w = w0 + tc + c.NC * ((x - 1) ÷ c.TM)
        i, j = own === :q ? (a, w) : (w, a)
        ok = (a < omax) & (w < wmax) & (!causal | (j <= i))
        ok && attention_allowed(attention_at(mask, j + 1, i + 1, hq, b)) ?
            attention_biased(acc[x], attention_at(bias, j + 1, i + 1, hq, b)) : A(-Inf)
    end
end

# The `lse` or the `δ` of the queries `i0 + s*m` of a thread. A query that takes no
# part, being padding or attending to no key, gets an infinite `lse`, hence zero weights.
@inline function attention_grad_lse(lse, ::Type{A}, i0, s, hq, b, qmax, ::Val{N}) where {A,N}
    ntuple(Val(N)) do m
        @inline
        i = i0 + s * (m - 1)
        x = A(@inbounds lse[hq, min(i, size(lse, 2) - 1) + 1, b])
        ifelse((i < qmax) & (x != A(-Inf)), x, A(Inf))
    end
end
@inline function attention_grad_delta(delta, ls::NTuple{N,A}, i0, s, hq, b) where {N,A}
    ntuple(Val(N)) do m
        @inline
        x = A(@inbounds delta[hq, min(i0 + s * (m - 1), size(delta, 2) - 1) + 1, b])
        ifelse(ls[m] != A(Inf), x, zero(A))
    end
end

# the queries are the rows of a thread's tile when they are owned, else its columns
@inline attention_grad_queries(::Val{:q}, o0, w0, tr, tc, ::Val{c}) where {c} = (o0 + tr, c.NR, Val(c.TM))
@inline attention_grad_queries(::Val{:k}, o0, w0, tr, tc, ::Val{c}) where {c} = (w0 + tc, c.NC, Val(c.TN))
@inline attention_grad_query(::Val{:q}, x, TM) = (x - 1) % TM + 1
@inline attention_grad_query(::Val{:k}, x, TM) = (x - 1) ÷ TM + 1

@inline attention_grad_p(sc::NTuple{MN,A}, ls::NTuple{N,A}, own::Val, ::Val{c}) where {MN,A,N,c} =
    ntuple(x -> exp(sc[x] - ls[attention_grad_query(own, x, c.TM)]), Val(MN))

@inline attention_grad_ds(pr::NTuple{MN,A}, gacc::NTuple{MN,A}, de::NTuple{N,A}, own::Val, ::Val{c}) where {MN,A,N,c} =
    ntuple(x -> pr[x] * (gacc[x] - de[attention_grad_query(own, x, c.TM)]), Val(MN))

@inline attention_get_tile(w, ::Type{A}, tr, tc, ::Val{c}) where {A,c} =
    ntuple(x -> @inbounds(w[tr + c.NR * ((x - 1) % c.TM) + c.PP * (tc + c.NC * ((x - 1) ÷ c.TM)) + 1]),
           Val(c.TM * c.TN))

@inline function attention_grad_write!(dx, oacc::NTuple{MN,A}, scale::A, ud, uq, o0, ho, b, ::Val{c}) where {MN,A,c}
    Dx, Lo = size(dx, 1), size(dx, 3)
    ntuple(Val(MN)) do x
        @inline
        d = ud + c.ND * ((x - 1) % c.TD)
        n = o0 + uq + c.NQ * ((x - 1) ÷ c.TD)
        if (d < Dx) & (n < Lo)
            @inbounds dx[d + 1, ho, n + 1, b] = oacc[x] * scale
        end
        nothing
    end
    return nothing
end

# dx = scale * Σ z * Wᵀ over the walked side, with W the weights P of the logits xoᵀxw
# if `yo` is nothing, else dS with G = yoᵀyw.
@kernel inbounds=true cpu=false unsafe_indices=true function attention_grad_kernel!(
        dx, @Const(xo), @Const(xw), yo, yw, @Const(z), @Const(lse), delta,
        bias, mask, o_lengths, w_lengths, oscale::A, wscale::A, dxscale::A, causal::Val,
        own::Val, til::Val{c}) where {A,c}
    w = @localmem A (c.PP * c.BK,)
    slab = @localmem A (c.SLAB,)

    t = @index(Local, Linear) - 1
    go, ho, b = @index(Group, NTuple)
    D = size(xo, 1)
    o0 = (go - 1) * c.BQ
    omax = attention_len(o_lengths, b, size(xo, 3))
    wfirst, wmax = attention_grad_range(own, causal, o0, attention_len(w_lengths, b, size(xw, 3)), c)
    ng = own === Val(:q) ? 1 : size(xw, 2) ÷ size(xo, 2)

    tr, tc = t ÷ c.NC, t % c.NC
    ud, uq = t ÷ c.NQ, t % c.NQ
    dd, r0 = t % c.KD, t ÷ c.KD

    oacc = ntuple(_ -> zero(A), Val(c.TD * c.TQ))
    zv = ntuple(_ -> zero(A), Val(c.KV * cld(c.DVP, c.T)))
    g = 0
    while g < ng
        hw = attention_grad_head(own, ho, g, size(xo, 2), size(xw, 2))
        hq = own === Val(:q) ? ho : hw
        w0 = wfirst
        ov = attention_fetch_rows(xo, oscale, dd, ho, o0, r0, b, D, omax, Val(c.BQ), Val(c.RS))
        wv = attention_fetch_rows(xw, wscale, dd, hw, w0, r0, b, D, wmax, Val(c.BK), Val(c.RS))
        while w0 < wmax
            # ---- logits
            acc = ntuple(_ -> zero(A), Val(c.TM * c.TN))
            d0 = 0
            while d0 < D
                attention_put_rows!(slab, 0, c.PQ, ov, dd, r0, Val(c.BQ), Val(c.RS))
                attention_put_rows!(slab, c.PQ * c.KD, c.PK, wv, dd, r0, Val(c.BK), Val(c.RS))
                @synchronize()
                d0 += c.KD
                if d0 < D
                    ov = attention_fetch_rows(xo, oscale, d0 + dd, ho, o0, r0, b, D, omax, Val(c.BQ), Val(c.RS))
                    wv = attention_fetch_rows(xw, wscale, d0 + dd, hw, w0, r0, b, D, wmax, Val(c.BK), Val(c.RS))
                elseif yo === nothing
                    zv = attention_fetch_dims(z, A, t, hw, w0, b, wmax, til)
                else
                    ov = attention_fetch_rows(yo, one(A), dd, ho, o0, r0, b, size(yo, 1), omax, Val(c.BQ), Val(c.RS))
                    wv = attention_fetch_rows(yw, one(A), dd, hw, w0, r0, b, size(yo, 1), wmax, Val(c.BK), Val(c.RS))
                end
                acc = attention_logits_slab(acc, slab, tr, tc, til)
                @synchronize()
            end

            # ---- weights
            i0, s, nq = attention_grad_queries(own, o0, w0, tr, tc, til)
            ls = attention_grad_lse(lse, A, i0, s, hq, b, own === Val(:q) ? omax : wmax, nq)
            pr = attention_grad_p(attention_grad_masked(acc, mask, bias, o0, w0, tr, tc, hq, b, omax, wmax,
                                                        causal, own, til), ls, own, til)
            attention_put_tile!(w, pr, tr, tc, til)
            if yo !== nothing
                # ---- G, as the logits, and dS in place of P
                acc = ntuple(_ -> zero(A), Val(c.TM * c.TN))
                d0 = 0
                while d0 < size(yo, 1)
                    attention_put_rows!(slab, 0, c.PQ, ov, dd, r0, Val(c.BQ), Val(c.RS))
                    attention_put_rows!(slab, c.PQ * c.KD, c.PK, wv, dd, r0, Val(c.BK), Val(c.RS))
                    @synchronize()
                    d0 += c.KD
                    if d0 < size(yo, 1)
                        ov = attention_fetch_rows(yo, one(A), d0 + dd, ho, o0, r0, b, size(yo, 1), omax, Val(c.BQ), Val(c.RS))
                        wv = attention_fetch_rows(yw, one(A), d0 + dd, hw, w0, r0, b, size(yo, 1), wmax, Val(c.BK), Val(c.RS))
                    else
                        zv = attention_fetch_dims(z, A, t, hw, w0, b, wmax, til)
                    end
                    acc = attention_logits_slab(acc, slab, tr, tc, til)
                    @synchronize()
                end
                de = attention_grad_delta(delta, ls, i0, s, hq, b)
                attention_put_tile!(w, attention_grad_ds(attention_get_tile(w, A, tr, tc, til), acc, de, own, til),
                                    tr, tc, til)
            end
            @synchronize()

            # ---- its product with the walked side
            jj0 = 0
            while jj0 < c.BK
                attention_put_dims!(slab, zv, t, til)
                @synchronize()
                jj0 += c.KV
                if jj0 < c.BK
                    zv = attention_fetch_dims(z, A, t, hw, w0 + jj0, b, wmax, til)
                elseif w0 + c.BK < wmax
                    ov = attention_fetch_rows(xo, oscale, dd, ho, o0, r0, b, D, omax, Val(c.BQ), Val(c.RS))
                    wv = attention_fetch_rows(xw, wscale, dd, hw, w0 + c.BK, r0, b, D, wmax, Val(c.BK), Val(c.RS))
                end
                oacc = attention_output_slab(oacc, slab, w, ud, uq, jj0 - c.KV, til)
                @synchronize()
            end
            w0 += c.BK
        end
        g += 1
    end
    attention_grad_write!(dx, oacc, dxscale, ud, uq, o0, ho, b, til)
end

function attention_grad_launch!(backend, dx, xo, xw, yo, yw, z, lse, delta, bias, mask, o_lengths,
                                w_lengths, oscale::A, wscale::A, dxscale::A, is_causal::Bool, own) where {A}
    c = attention_tiling(size(dx, 1), length(dx) ÷ size(dx, 1), attention_shmem_budget(backend) ÷ sizeof(A))
    attention_grad_kernel!(backend, (c.T, 1, 1))(dx, xo, xw, yo, yw, z, lse, delta, bias, mask, o_lengths,
        w_lengths, oscale, wscale, dxscale, Val(is_causal), own, Val(c);
        ndrange=(c.T * cld(size(dx, 3), c.BQ), size(dx, 2), size(dx, 4)))
    return dx
end

function fused_attention_grad(backend::GPU, dout, o, lse, q, k, v, bias, mask, q_lengths, kv_lengths,
                              scale::A, qscale::A, is_causal::Bool) where {A}
    delta = dropdims(mapreduce(attention_wide_prod, +, dout, o; dims=1); dims=1)
    dq = attention_grad_launch!(backend, similar(q), q, k, dout, v, k, lse, delta, bias, mask, q_lengths,
                                kv_lengths, scale, one(A), qscale, is_causal, Val(:q))
    dk = attention_grad_launch!(backend, similar(k), k, q, v, dout, q, lse, delta, bias, mask, kv_lengths,
                                q_lengths, one(A), scale, scale, is_causal, Val(:k))
    dv = attention_grad_launch!(backend, similar(v), k, q, nothing, nothing, dout, lse, nothing, bias, mask,
                                kv_lengths, q_lengths, one(A), scale, one(A), is_causal, Val(:k))
    return dq, dk, dv
end
