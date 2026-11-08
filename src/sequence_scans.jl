"""
    NNlib.ssd_scan(x, dt, A, B, C; D=nothing, initial_state=nothing, lengths=nothing,
                   chunk_size=128) -> (y, final_state)

The selective scan of Mamba-2, in the state space duality (SSD) form of
[Dao and Gu](https://arxiv.org/abs/2405.21060). Each head carries a state `S` of size
`(head_dim, state_size)` along the sequence. At the token `t`, with the step
`Δ = dt[h, t]` of the head `h`,

    S = exp(Δ * A[h]) * S + Δ * x[:, h, t] * B[:, t]ᵀ
    y[:, h, t] = S * C[:, t] + D[h] * x[:, h, t]

The arrays have the layout of [`scaled_dot_product_attention`](@ref), and the heads are in
groups that share `B` and `C`, as the queries share keys and values there:

- `x`: The inputs, of size `(head_dim, nheads, seq_len, batch_size)`.
- `dt`: The positive steps, of size `(nheads, seq_len, batch_size)`. In Mamba-2, the
        `softplus` of a projection of the input plus `dt_bias`, clamped.
- `A`: The negative rates of decay of the heads, a vector of length `nheads`. In Mamba-2,
        `-exp.(A_log)`.
- `B`, `C`: The projections that write the input into the state and read the state into
        the output, of size `(state_size, ngroups, seq_len, batch_size)`, with `nheads` a
        multiple of `ngroups`. The heads `(i - 1) * r + 1` to `i * r`, with
        `r = nheads ÷ ngroups`, use the group `i`.

Returns `y`, of the size and element type of `x`, and the state after the last token,
`final_state`, of size `(head_dim, state_size, nheads, batch_size)`. It is computed in at
least `Float32`, as the outputs, and kept in that type.

# Keywords

- `D`: Either `nothing` or the vector of the skip connections of the heads. Default
       `nothing`.
- `initial_state`: Either `nothing`, meaning zeros, or the states before the first token,
       of size `(head_dim, state_size, nheads, batch_size)`, as the `final_state` of the
       previous part of the sequences. Default `nothing`.
- `lengths`: Either `nothing` or the number of valid positions of each sequence of a batch
       padded on the right: an integer, or an integer array with one entry for each batch
       element. `final_state` is then the state at the end of each sequence, and the
       outputs past it are not meaningful. Default `nothing`, meaning all the positions.
- `chunk_size`: The number of tokens of the chunks of the chunked algorithm. Default `128`.

A sequence of a single token takes one step of the recurrence, as when decoding. Longer
sequences are cut in chunks of `chunk_size` tokens, computed in parallel by matrix
products, as `mamba_chunk_scan_combined` of [mamba_ssm](https://github.com/state-spaces/mamba)
and its PyTorch version in Hugging Face Transformers: the states at the ends of the
chunks are combined by a product with a matrix of their decays. The
chunk size trades memory, `(chunk_size, chunk_size)` per head and chunk, against the
`(head_dim, state_size)` state per head and chunk.

On the GPU backends, the chunked algorithm runs as kernels, as those of mamba_ssm: they
form `Cᵀ B` once for each group of heads and apply the decays as they read it, carry the
states from chunk to chunk with one pass over them, and have a gradient that keeps only the
states at the starts of the chunks. The kernels take chunks of at most 256 tokens. The
gradient is that of the kernels under ChainRules (Zygote); other differentiation packages
differentiate the generic implementation. Outside of differentiation, a single token is one
kernel, which reads and writes the states once, for a `state_size` of at most 256.

The state of Transformers' `ssm_state`, of size `(batch_size, nheads, head_dim,
state_size)`, is `permutedims(final_state, (4, 3, 1, 2))`.

# Examples

The scan of a prompt followed by the decoding of one token, with continuing states:

```julia
P, H, N, G = 64, 8, 128, 1       # head_dim, nheads, state_size, ngroups
x = randn(Float32, P, H, 10, 2)  # (head_dim, nheads, seq_len, batch_size)
dt = softplus.(randn(Float32, H, 10, 2))
A = -exp.(randn(Float32, H))
B, C = randn(Float32, N, G, 10, 2), randn(Float32, N, G, 10, 2)
y, state = NNlib.ssd_scan(x, dt, A, B, C)

x1, dt1 = randn(Float32, P, H, 1, 2), softplus.(randn(Float32, H, 1, 2))
B1, C1 = randn(Float32, N, G, 1, 2), randn(Float32, N, G, 1, 2)
y1, state = NNlib.ssd_scan(x1, dt1, A, B1, C1; initial_state=state)
```
"""
function ssd_scan(x::AbstractArray{<:Any,4}, dt::AbstractArray{<:Any,3}, A::AbstractVector,
                  B::AbstractArray{<:Any,4}, C::AbstractArray{<:Any,4};
                  D=nothing, initial_state=nothing, lengths=nothing, chunk_size::Integer=128)
    check_ssd_scan(x, dt, A, B, C, D, initial_state, chunk_size)
    P, h, n, b = size(x)
    T = promote_type(Float32, eltype(x), eltype(dt), eltype(A), eltype(B), eltype(C))
    if n == 1 && !within_gradient(x) && step_kernels_apply(size(B, 1), x, dt, A, B, C, D, initial_state)
        return ssd_step_kernel(T, scan_dense.((x, dt, A, B, C, D, initial_state))..., lengths)
    end
    if n > 1 && scan_kernels_apply(x, dt, A, B, C, D, initial_state) && scan_kernels_differentiable(x)
        dt = reshape(mask_padding(reshape(T.(dt), 1, h, n, b), lengths), h, n, b)
        return ssd_kernels(scan_dense(x), dt, T.(A), scan_dense(B), scan_dense(C),
                           D === nothing ? nothing : T.(D), scan_dense(initial_state),
                           min(chunk_size, n, L_MAX_SCAN))
    end
    r = h ÷ size(B, 2)
    xT = T.(x)
    dt = mask_padding(reshape(T.(dt), 1, h, n, b), lengths)
    A = reshape(T.(A), 1, h)
    B, C = T.(B), T.(C)
    state = initial_state === nothing ? nothing : T.(initial_state)
    y, state = n == 1 ? ssd_step(xT, dt, A, repeat_heads(B, r), repeat_heads(C, r), state) :
                        ssd_chunked(xT, dt, A, B, C, state, chunk_size)
    y = D === nothing ? y : y .+ reshape(T.(D), 1, h) .* xT
    return eltype(y) == eltype(x) ? y : eltype(x).(y), state
end

function check_ssd_scan(x, dt, A, B, C, D, initial_state, chunk_size)
    P, h, n, b = size(x)
    N, g = size(B, 1), size(B, 2)
    size(dt) == (h, n, b) || throw(DimensionMismatch(
        "`dt` has to have size (nheads, seq_len, batch_size) = $((h, n, b)), got $(size(dt))."))
    length(A) == h || throw(DimensionMismatch(
        "`A` has to have one entry per head ($h), got $(length(A))."))
    size(C) == size(B) || throw(DimensionMismatch(
        "`B` and `C` have to have the same size, got $(size(B)) and $(size(C))."))
    size(B)[3:4] == (n, b) || throw(DimensionMismatch(
        "`B` and `C` have to have size (state_size, ngroups, seq_len, batch_size) with \
         seq_len = $n and batch_size = $b, got $(size(B))."))
    h % g == 0 || throw(DimensionMismatch(
        "The number of heads ($h) has to be a multiple of the number of groups ($g)."))
    D === nothing || length(D) == h || throw(DimensionMismatch(
        "`D` has to have one entry per head ($h), got $(length(D))."))
    initial_state === nothing || size(initial_state) == (P, N, h, b) ||
        throw(DimensionMismatch("The initial state has to have size \
                                 (head_dim, state_size, nheads, batch_size) = \
                                 $((P, N, h, b)), got $(size(initial_state))."))
    chunk_size >= 1 || throw(ArgumentError("The chunk size has to be positive, got $chunk_size."))
    return nothing
end

@non_differentiable check_ssd_scan(::Any...)

# One token: `x` (P, h, 1, b), `dt` (1, h, 1, b), `A` (1, h), `B` and `C` (N, h, 1, b).
function ssd_step(x, dt, A, B, C, state)
    P, h, _, b = size(x)
    N = size(B, 1)
    write = reshape(x .* dt, P, 1, h, b) .* reshape(B, 1, N, h, b)
    state = state === nothing ? write :
            state .* reshape(exp.(dt .* A), 1, 1, h, b) .+ write
    y = batched_mul(reshape(state, P, N, :), reshape(C, N, 1, :))
    return reshape(y, P, h, 1, b), state
end

# The chunks are processed in parallel: the outputs within a chunk are a masked product
# of its matrices, as in attention, and the state at the start of each chunk is a sum of
# the states written by the previous chunks, weighted by their decays. `B` and `C` are those
# of the groups of heads, of which `Bᵀ C` is formed once.
function ssd_chunked(x, dt, A, B, C, state, chunk_size)
    P, h, n, b = size(x)
    N, g = size(B, 1), size(B, 2)
    r = h ÷ g
    L = min(chunk_size, n)
    pad = mod(-n, L)
    nc = (n + pad) ÷ L
    chunked(z) = to_chunks(z, L, pad)
    flat(z) = reshape(z, size(z, 1), size(z, 2), :)
    # (features, L, nc, nheads * batch_size)
    xdt = chunked(x .* dt)
    cum = scan_cumsum(chunked(dt .* A); dims=2)
    BC = batched_mul(batched_transpose(flat(chunked(B))), flat(chunked(C)))
    weights = reshape(BC, L, L, nc, 1, g, b) .* reshape(chunk_decays(cum), L, L, nc, r, g, b)
    y = batched_mul(flat(xdt), reshape(weights, L, L, :))
    Bc, Cc = chunked(repeat_heads(B, r)), chunked(repeat_heads(C, r))
    Bt = batched_transpose(flat(Bc))
    last = cum[:, L:L, :, :]
    written = batched_mul(flat(xdt .* exp.(last .- cum)), Bt)
    s0 = state === nothing ? zeros_like(x, (P * N, 1, h * b)) : reshape(state, P * N, 1, :)
    # the states at the start of each chunk, and after the last
    states = batched_mul(cat(s0, reshape(written, P * N, nc, :); dims=2),
                         chunk_decays(scan_cumsum(pad_zeros(reshape(last, 1, nc, :), (1, 0); dims=2); dims=2)))
    y = y .+ batched_mul(reshape(states[:, 1:nc, :], P, N, :), flat(Cc)) .* flat(exp.(cum))
    return from_chunks(reshape(y, P, L, nc, :), h, n), reshape(states[:, nc + 1, :], P, N, h, b)
end

"""
    NNlib.gated_delta_rule(q, k, v, g, beta; scale=nothing, initial_state=nothing,
                           lengths=nothing, chunk_size=64) -> (o, final_state)

The gated delta rule of Gated DeltaNet ([Yang et al.](https://arxiv.org/abs/2412.06464)),
the linear attention of Qwen3-Next and Qwen3.5. Each head carries a state `S` of size
`(head_v_dim, head_k_dim)` along the sequence. At the token `t`, it decays by `exp(g)`, then
its value at the key `k_t` moves toward `v_t` at the rate `beta`, and is read at `q_t`:

    S = exp(g[h, t]) * S
    S = S + beta[h, t] * (v_t - S * k_t) * k_tᵀ
    o_t = S * (scale * q_t)

The arrays have the layout of [`scaled_dot_product_attention`](@ref), and as there the
queries and keys can have fewer heads than the values:

- `q`, `k`: The queries and the keys, of size `(head_k_dim, nkheads, seq_len, batch_size)`.
        The delta rule expects keys of unit norm: Qwen3-Next and Qwen3.5 normalise the
        queries and the keys (Transformers' `use_qk_l2norm_in_kernel`).
- `v`: The values, of size `(head_v_dim, nheads, seq_len, batch_size)`, with `nheads` a
        multiple of `nkheads`. The heads `(i - 1) * r + 1` to `i * r`, with
        `r = nheads ÷ nkheads`, use the query and key head `i`.
- `g`: The logarithms of the decays, nonpositive, of size `(nheads, seq_len, batch_size)`.
        In Gated DeltaNet, `-exp(A_log) * softplus(a + dt_bias)`.
- `beta`: The rates of the writes, in `[0, 1]`, of size `(nheads, seq_len, batch_size)`.

Returns the outputs `o`, of size `(head_v_dim, nheads, seq_len, batch_size)` and the
element type of `v`, and the state after the last token, `final_state`, of size
`(head_v_dim, head_k_dim, nheads, batch_size)`. It is computed in at least `Float32`, as the
outputs, and kept in that type.

# Keywords

- `scale`: The multiplier of the queries. Default `nothing`, meaning `1 / √head_k_dim`.
- `initial_state`: Either `nothing`, meaning zeros, or the states before the first token,
       of size `(head_v_dim, head_k_dim, nheads, batch_size)`, as the `final_state` of the
       previous part of the sequences. Default `nothing`.
- `lengths`: Either `nothing` or the number of valid positions of each sequence of a batch
       padded on the right: an integer, or an integer array with one entry for each batch
       element. `final_state` is then the state at the end of each sequence, and the
       outputs past it are not meaningful. Default `nothing`, meaning all the positions.
- `chunk_size`: The number of tokens of the chunks of the chunked algorithm, a power of 2.
       Default `64`.

A sequence of a single token takes one step of the recurrence, as when decoding, as
`torch_recurrent_gated_delta_rule` of Transformers. Longer sequences are cut in chunks of
up to `chunk_size` tokens, as in `torch_chunk_gated_delta_rule`: within a chunk, the
writes are found by solving a unit triangular system, here by inverting its matrix by
blocks, and the outputs are matrix products. The state is carried from one chunk to the
next by a loop over the chunks.

On the GPU backends, the chunked algorithm runs as kernels, as those of
flash-linear-attention: the products of the queries and the keys are formed once for each
head of keys, the triangular inverse of each chunk in shared memory, and the state is
carried through the chunks by blocks of its rows. The gradient keeps the states at the
starts of the chunks. The kernels take chunks of at most 64 tokens. The gradient is that
of the kernels under ChainRules (Zygote); other differentiation packages differentiate the
generic implementation. Outside of differentiation, a single token is one kernel, which
reads and writes the states once, for a `head_k_dim` of at most 256.

The state of Transformers' `recurrent_state`, of size `(batch_size, nheads, head_k_dim,
head_v_dim)`, is `permutedims(final_state, (4, 3, 2, 1))`.

# Examples

The gated delta rule of a prompt followed by the decoding of one token, with continuing
states, here with 4 heads of values sharing 2 heads of queries and keys:

```julia
l2norm(x) = x ./ sqrt.(sum(abs2, x; dims=1) .+ 1f-6)
q, k = l2norm(randn(Float32, 128, 2, 10, 3)), l2norm(randn(Float32, 128, 2, 10, 3))
v = randn(Float32, 128, 4, 10, 3)           # (head_v_dim, nheads, seq_len, batch_size)
g, beta = -rand(Float32, 4, 10, 3), rand(Float32, 4, 10, 3)
o, state = NNlib.gated_delta_rule(q, k, v, g, beta)

q1, k1 = l2norm(randn(Float32, 128, 2, 1, 3)), l2norm(randn(Float32, 128, 2, 1, 3))
v1, g1, beta1 = randn(Float32, 128, 4, 1, 3), -rand(Float32, 4, 1, 3), rand(Float32, 4, 1, 3)
o1, state = NNlib.gated_delta_rule(q1, k1, v1, g1, beta1; initial_state=state)
```
"""
function gated_delta_rule(q::AbstractArray{<:Any,4}, k::AbstractArray{<:Any,4},
                          v::AbstractArray{<:Any,4}, g::AbstractArray{<:Any,3},
                          beta::AbstractArray{<:Any,3}; scale=nothing, initial_state=nothing,
                          lengths=nothing, chunk_size::Integer=64)
    check_gated_delta_rule(q, k, v, g, beta, initial_state, chunk_size)
    dk, hk, n, b = size(q)
    h = size(v, 2)
    T = promote_type(Float32, eltype(q), eltype(k), eltype(v), eltype(g), eltype(beta))
    scale = T(something(scale, 1 / sqrt(T(dk))))
    if n == 1 && !within_gradient(q) && step_kernels_apply(dk, q, k, v, g, beta, initial_state)
        return delta_step_kernel(scan_dense.((q, k, v, g, beta))..., scale, scan_dense(initial_state), lengths)
    end
    if n > 1 && delta_kernels_apply(q, k, v, g, beta, initial_state) && scan_kernels_differentiable(q)
        g = reshape(mask_padding(reshape(T.(g), 1, h, n, b), lengths), h, n, b)
        beta = reshape(mask_padding(reshape(T.(beta), 1, h, n, b), lengths), h, n, b)
        return delta_kernels(scan_dense(q), scan_dense(k), scan_dense(v), g, beta, scale,
                             scan_dense(initial_state), min(chunk_size, n, L_MAX_DELTA))
    end
    q, k = repeat_heads(T.(q) .* scale, h ÷ hk), repeat_heads(T.(k), h ÷ hk)
    g = mask_padding(reshape(T.(g), 1, h, n, b), lengths)
    beta = mask_padding(reshape(T.(beta), 1, h, n, b), lengths)
    state = initial_state === nothing ? nothing : T.(initial_state)
    o, state = n == 1 ? gated_delta_step(q, k, T.(v), g, beta, state) :
                        gated_delta_chunked(q, k, T.(v), g, beta, state, chunk_size)
    return eltype(o) == eltype(v) ? o : eltype(v).(o), state
end

function check_gated_delta_rule(q, k, v, g, beta, initial_state, chunk_size)
    dk, hk, n, b = size(q)
    dv, h = size(v, 1), size(v, 2)
    size(k) == size(q) || throw(DimensionMismatch(
        "The queries and the keys have to have the same size, got $(size(q)) and $(size(k))."))
    size(v)[3:4] == (n, b) || throw(DimensionMismatch(
        "The values of size $(size(v)) have to have the seq_len and batch_size of the \
         queries, $((n, b))."))
    h % hk == 0 || throw(DimensionMismatch(
        "The number of heads of the values ($h) has to be a multiple of that of the \
         queries and keys ($hk)."))
    size(g) == size(beta) == (h, n, b) || throw(DimensionMismatch(
        "`g` and `beta` have to have size (nheads, seq_len, batch_size) = $((h, n, b)), \
         got $(size(g)) and $(size(beta))."))
    initial_state === nothing || size(initial_state) == (dv, dk, h, b) ||
        throw(DimensionMismatch("The initial state has to have size \
                                 (head_v_dim, head_k_dim, nheads, batch_size) = \
                                 $((dv, dk, h, b)), got $(size(initial_state))."))
    ispow2(chunk_size) || throw(ArgumentError(
        "The chunk size has to be a power of 2, got $chunk_size."))
    return nothing
end

@non_differentiable check_gated_delta_rule(::Any...)

# One token: `q`, `k` (dk, h, 1, b), `v` (dv, h, 1, b), `g`, `beta` (1, h, 1, b).
function gated_delta_step(q, k, v, g, beta, state)
    dk, h, _, b = size(q)
    dv = size(v, 1)
    k = reshape(k, dk, 1, :)
    v, beta = reshape(v, dv, 1, :), reshape(beta, 1, 1, :)
    if state === nothing
        s = batched_mul(v .* beta, batched_transpose(k))
    else
        s = reshape(state, dv, dk, :) .* reshape(exp.(g), 1, 1, :)
        s = s .+ batched_mul((v .- batched_mul(s, k)) .* beta, batched_transpose(k))
    end
    o = batched_mul(s, reshape(q, dk, 1, :))
    return reshape(o, dv, h, 1, b), reshape(s, dv, dk, h, b)
end

# Within a chunk of tokens `1:L` with the cumulative log decays `cum` and the state `S0`
# at its start, the state after the token `t` is
# `exp(cum[t]) * S0 + sum(exp(cum[t] - cum[j]) * u[j] * k[j]ᵀ for j in 1:t)`, where the
# writes `u` solve a unit triangular system. The outputs then are matrix products, and
# only the states at the starts of the chunks need a loop.
function gated_delta_chunked(q, k, v, g, beta, state, chunk_size)
    dk, h, n, b = size(q)
    dv = size(v, 1)
    L = min(chunk_size, nextpow(2, n))
    pad = mod(-n, L)
    nc = (n + pad) ÷ L
    chunked(z) = to_chunks(z, L, pad)
    flat(z) = reshape(z, size(z, 1), size(z, 2), :)
    # (features, L, nc, nheads * batch_size)
    q, k, v, g, beta = chunked(q), chunked(k), chunked(v), chunked(g), chunked(beta)
    k_beta = k .* beta
    cum = scan_cumsum(g; dims=2)
    decay = flat(chunk_decays(cum))
    kt = batched_transpose(flat(k))
    attn = batched_mul(kt, flat(q)) .* decay
    solve = unit_triangular_inverse(batched_mul(kt, flat(k_beta)) .* decay)
    u = batched_mul(flat(v .* beta), solve)
    w = batched_mul(flat(k_beta .* exp.(cum)), solve)
    last = cum[:, L:L, :, :]
    s0 = state === nothing ? zeros_like(q, (dv, dk, h * b)) : reshape(state, dv, dk, :)
    states = chunk_states(s0, reshape(u, dv, L, nc, :), reshape(w, dk, L, nc, :),
                          k .* exp.(last .- cum), exp.(last))
    s = reshape(states[:, :, 1:nc, :], dv, dk, :)
    o = batched_mul(s, flat(q .* exp.(cum))) .+ batched_mul(u .- batched_mul(s, w), attn)
    return from_chunks(reshape(o, dv, L, nc, :), h, n), reshape(states[:, :, nc + 1, :], dv, dk, h, b)
end

# The inverse of `I + U` for the strictly upper triangular part `U` of each matrix of `a`
# (L, L, n), L a power of 2. It inverts diagonal blocks of doubling size: a block
# [A B; 0 D] whose diagonal blocks are inverted has the inverse [A⁻¹ -A⁻¹BD⁻¹; 0 D⁻¹], so
# that with `x` the block diagonal matrix of the inverses and `B` the off-diagonal blocks
# of `U`, the next `x` is `x - x B x`. Its intermediate results are blocks of the inverse,
# unlike those of the series `I - U + U² - ...`, which can be much larger and lose
# precision to cancellation.
function unit_triangular_inverse(a)
    L = size(a, 1)
    x = identity_mask(a, L) .- a .* block_mask(a, L, 1)
    s = 2
    while s < L
        x = x .- batched_mul(batched_mul(x, a .* block_mask(a, L, s)), x)
        s *= 2
    end
    return x
end

function identity_mask(x, L)
    mask = similar(x, L, L)
    return mask .= (1:L) .== (1:L)'
end

# 1 at the rows of the first half and the columns of the second half of each diagonal
# block of size 2s, else 0.
function block_mask(x, L, s)
    i, j = 0:(L - 1), (0:(L - 1))'
    mask = similar(x, L, L)
    return mask .= (i .÷ 2s .== j .÷ 2s) .& iseven.(i .÷ s) .& isodd.(j .÷ s)
end

@non_differentiable identity_mask(::Any...)
@non_differentiable block_mask(::Any...)

# The states (dv, dk, nc + 1, n) at the start of each of the `nc` chunks and after the
# last, from the state `s` (dv, dk, n) before the first, the writes `u` (dv, L, nc, n),
# the keys that read the state `w` and that write it `kd` (dk, L, nc, n), and the decays
# `decay` (1, 1, nc, n) of the chunks. After a chunk, the state is
# `decay * s + (u - s * w) * kdᵀ`. Its gradient runs the loop backwards.
function chunk_states(s, u, w, kd, decay)
    nc = size(u, 3)
    states = similar(s, size(s, 1), size(s, 2), nc + 1, size(s, 3))
    for c in 1:nc
        view(states, :, :, c, :) .= s
        s = s .* decay[:, :, c, :] .+
            batched_mul(u[:, :, c, :] .- batched_mul(s, w[:, :, c, :]),
                        batched_transpose(kd[:, :, c, :]))
    end
    view(states, :, :, nc + 1, :) .= s
    return states
end

function rrule(::typeof(chunk_states), s0, u, w, kd, decay)
    states = chunk_states(s0, u, w, kd, decay)
    function chunk_states_pullback(Δ)
        Δ = unthunk(Δ)
        Δ isa AbstractZero && return (NoTangent(), ntuple(_ -> Δ, 5)...)
        nc = size(u, 3)
        du, dw, dkd, ddecay = similar(u), similar(w), similar(kd), similar(decay)
        # the gradient of the state after the chunk `c`
        ds = Δ[:, :, nc + 1, :]
        for c in nc:-1:1
            s, wc = states[:, :, c, :], w[:, :, c, :]
            du_c = batched_mul(ds, kd[:, :, c, :])
            view(du, :, :, c, :) .= du_c
            view(dw, :, :, c, :) .= .-batched_mul(batched_transpose(s), du_c)
            view(dkd, :, :, c, :) .= batched_mul(batched_transpose(ds),
                                                 u[:, :, c, :] .- batched_mul(s, wc))
            view(ddecay, :, :, c, :) .= sum(ds .* s; dims=(1, 2))
            ds = Δ[:, :, c, :] .+ ds .* decay[:, :, c, :] .- batched_mul(du_c, batched_transpose(wc))
        end
        return NoTangent(), ds, du, dw, dkd, ddecay
    end
    return states, chunk_states_pullback
end

# `[j, i] = exp(cum[i] - cum[j])` for `j ≤ i`, else 0, from `cum` of size `(1, L, dims...)`:
# `(L, L, dims...)`. The decay from the position `j` to the position `i`, when `cum` are
# cumulative log decays.
function chunk_decays(cum)
    L = size(cum, 2)
    return exp.(cum .- reshape(cum, L, 1, size(cum)[3:end]...) .+ causal_log_mask(cum, L))
end

# 0 where `j ≤ i`, -Inf after: (L, L), on the device of `x`.
function causal_log_mask(x, L)
    mask = similar(x, L, L)
    return mask .= ifelse.((1:L) .<= (1:L)', zero(eltype(x)), typemin(eltype(x)))
end

@non_differentiable causal_log_mask(::Any...)

# `cumsum`, with a pullback without `reverse`, which GPUArrays 11 runs on the CPU (Metal)
scan_cumsum(x; dims) = cumsum(x; dims)

function rrule(::typeof(scan_cumsum), x; dims)
    scan_cumsum_pullback(Δ) = (NoTangent(), reverse_cumsum(unthunk(Δ); dims))
    return scan_cumsum(x; dims), scan_cumsum_pullback
end

# `Σ_{j ≥ i} d[j]` along `dims`
reverse_cumsum(d; dims) = sum(d; dims) .- cumsum(d; dims) .+ d

# `x` (features, nheads, seq_len, batch_size) padded with `pad` zero tokens and cut in
# chunks of `L` tokens: (features, L, chunks, nheads * batch_size).
function to_chunks(x, L, pad)
    d, h, _, b = size(x)
    x = pad == 0 ? x : pad_zeros(x, (0, pad); dims=3)
    return reshape(permutedims(reshape(x, d, h, L, :, b), (1, 3, 4, 2, 5)), d, L, :, h * b)
end

# The inverse of `to_chunks`, back to `n` tokens.
function from_chunks(y, h, n)
    d, L, nc = size(y, 1), size(y, 2), size(y, 3)
    y = reshape(permutedims(reshape(y, d, L, nc, h, :), (1, 4, 2, 3, 5)), d, h, L * nc, :)
    return L * nc == n ? y : y[:, :, 1:n, :]
end

# `x` (1, nheads, seq_len, batch_size), zero past the `lengths` of the sequences.
mask_padding(x, ::Nothing) = x

function mask_padding(x, lengths)
    n, b = size(x, 3), size(x, 4)
    return x .* reshape(sequence_mask(x, lengths, n, b), 1, 1, n, b)
end

sequence_mask(x, lengths, n, b) = (1:n) .<= sequence_lengths(x, lengths, n, b)

@non_differentiable sequence_mask(::Any...)

# torch's `x.repeat_interleave(r, dim=2)` for the heads of `x` (features, nheads, seq_len,
# batch_size).
repeat_heads(x, r) = r == 1 ? x : repeat(x; inner=(1, r, 1, 1))

zeros_like(x, sz) = fill!(similar(x, sz), 0)

@non_differentiable zeros_like(::Any...)
