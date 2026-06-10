const AA3{T} = AbstractArray{T,3}
const AA4{T} = AbstractArray{T,4}
const AA{N,T} = AbstractArray{T,N}

"""
    scaled_dot_product_attention(query, key, value, [bias]; [fdrop, mask, scale, is_causal])

Multihead dot product attention used in transformer architectures, with an explicit
head axis as in PyTorch.

The input arrays must have the shape `(head_dim, nheads, seq_len, batch_size...)`, that
is a feature (head) dimension, a heads dimension, a sequence-length dimension, then an
arbitrary number of batch dimensions or none. The number of heads is taken from the
second dimension of each input, so no `nheads` keyword is needed.

Grouped-query attention (GQA) is supported: if `key` and `value` have fewer heads than
`query`, they are shared across query heads. The number of query heads must then be
divisible by the number of key/value heads.

Returns the attention output array of size `(v_head_dim, nheads, q_len, batch_size...)`.

See also [`scaled_dot_product_attention_scores`](@ref) if you need the attention scores
of size `(kv_len, q_len, nheads, batch_size...)`.

# Arguments

- `query`: Query array of size `(qk_head_dim, nheads, q_len, batch_size...)`.
- `key`: Key array of size `(qk_head_dim, nkvheads, kv_len, batch_size...)`.
- `value`: Value array of size `(v_head_dim, nkvheads, kv_len, batch_size...)`.
- `bias`: Either `nothing` or an array broadcastable to size `(kv_len, q_len, nheads, batch_size)`.
          It will be added to the attention scores before applying the softmax. Default `nothing`.
- `fdrop`: A dropout function or layer to be applied on the attention scores right after the softmax.
           Default `identity` (no dropout).
- `mask`: Either `nothing` or a boolean array broadcastable to size `(kv_len, q_len, nheads, batch_size)`.
          The mask is applied to the attention scores just before the softmax.
          See [`make_causal_mask`](@ref) for creating causal masks. Default `nothing`.
          Cannot be used together with `is_causal=true`.
- `scale`: The denominator used to scale the `queryᵀ * key` dot products before
          the softmax. Default `nothing`, meaning `√(qk_head_dim)`.
- `is_causal`: If `true`, a causal mask is applied so that each query position can
          only attend to key positions up to and including its own. Convenience for
          `mask = make_causal_mask(...)`; cannot be combined with `mask`. Default `false`.

# Examples

```julia
# (head_dim, nheads, seq_len, batch)
q = rand(Float32, 16, 8, 20, 4)
k = v = rand(Float32, 16, 8, 30, 4)
y = scaled_dot_product_attention(q, k, v)
# size(y) == (16, 8, 20, 4)

# grouped-query attention: 8 query heads sharing 2 key/value heads
q = rand(Float32, 16, 8, 20, 4)
k = v = rand(Float32, 16, 2, 30, 4)
y = scaled_dot_product_attention(q, k, v)
```
"""
function scaled_dot_product_attention(q::AA{N}, k::AA{N}, v::AA{N}, bias=nothing;
            kws...) where N
    x, _ = _scaled_dot_product_attention(q, k, v, bias; kws...)
    return x
end

# Shared implementation returning both the attention output and the scores `α`,
# so the (deprecated) `dot_product_attention` can return a consistent `(x, α)` pair
# without applying a stochastic `fdrop` twice.
function _scaled_dot_product_attention(q::AA{N}, k::AA{N}, v::AA{N}, bias=nothing;
            fdrop=identity, mask=nothing, scale=nothing, is_causal::Bool=false) where N

    N >= 3 || throw(ArgumentError(
        "Inputs must have at least 3 dimensions (head_dim, nheads, seq_len). Got $(N)."))

    size(q, 1) == size(k, 1) || throw(ArgumentError("""
    Head dimension (first dimension) of query and key has to be the same. Instead:
    - size(q): $(size(q))
    - size(k): $(size(k))
    """))
    size(k, 2) == size(v, 2) || throw(ArgumentError("""
    Number of heads (second dimension) of key and value has to be the same. Instead:
    - size(k): $(size(k))
    - size(v): $(size(v))
    """))
    size(k, 3) == size(v, 3) || throw(ArgumentError("""
    Sequence length (third dimension) of key and value has to be the same. Instead:
    - size(k): $(size(k))
    - size(v): $(size(v))
    """))
    size(q)[4:end] == size(k)[4:end] == size(v)[4:end] || throw(ArgumentError("""
    Batch dimensions have to be the same. Instead:
    - size(q): $(size(q))
    - size(k): $(size(k))
    - size(v): $(size(v))
    """))
    nheads, nkvheads = size(q, 2), size(k, 2)
    (nheads % nkvheads == 0) || throw(ArgumentError("""
    Number of query heads must be divisible by the number of key/value heads
    (grouped-query attention). Instead:
    - nheads (size(q, 2)): $nheads
    - nkvheads (size(k, 2)): $nkvheads
    """))
    (is_causal && mask !== nothing) && throw(ArgumentError(
        "`mask` and `is_causal=true` cannot be specified at the same time."))

    if is_causal
        mask = make_causal_mask(q, size(k, 3), size(q, 3))
    end

    batch_size = size(q)[4:end]
    # collapse the batch dimensions into a single one
    q, k, v = map(x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), :), (q, k, v))
    if nkvheads != nheads
        # Grouped-query attention: expand the kv heads so each group of
        # `nheads ÷ nkvheads` query heads shares one key/value head.
        r = nheads ÷ nkvheads
        k = repeat_kv_heads(k, r)
        v = repeat_kv_heads(v, r)
    end

    x, α = _attention(q, k, v, bias, fdrop, mask, scale)

    x = reshape(x, size(x, 1), size(x, 2), size(x, 3), batch_size...)
    α = reshape(α, size(α)[1:3]..., batch_size...)
    return x, α
end

function _attention(q::AA4, k::AA4, v::AA4, bias, fdrop, mask, scale)
    # [q] = [qk_head_dim, nheads, q_len, batch_size]
    # [k] = [qk_head_dim, nheads, kv_len, batch_size]
    # [v] = [v_head_dim,  nheads, kv_len, batch_size]

    α = _attention_scores(q, k, bias, fdrop, mask, scale)
    # [α] = [kv_len, q_len, nheads, batch_size]

    # The following permutedims and batched_mul are equivalent to
    # @tullio x[d, h, i, b] := α[j, i, h, b] * v[d, h, j, b]
    vt = permutedims(v, (1, 3, 2, 4))
    x = batched_mul(vt, α)
    x = permutedims(x, (1, 3, 2, 4))
    # [x] = [v_head_dim, nheads, q_len, batch_size]
    return x, α
end

"""
    scaled_dot_product_attention_scores(query, key, [bias]; [fdrop, mask, scale, is_causal])

Return the attention scores for [`scaled_dot_product_attention`](@ref).
Input arrays must have dimensions `(head_dim, nheads, seq_len, batch_size...)`.

The `scale` keyword sets the denominator scaling the `queryᵀ * key` dot products;
it defaults to `√(head_dim)`.

See [`scaled_dot_product_attention`](@ref) for more details.
"""
function scaled_dot_product_attention_scores(q::AA{N}, k::AA{N}, bias=nothing;
            fdrop=identity, mask=nothing, scale=nothing, is_causal::Bool=false) where N

    N >= 3 || throw(ArgumentError(
        "Inputs must have at least 3 dimensions (head_dim, nheads, seq_len). Got $(N)."))
    size(q, 1) == size(k, 1) || throw(ArgumentError("""
    Head dimension (first dimension) of query and key has to be the same. Instead:
    - size(q): $(size(q))
    - size(k): $(size(k))
    """))
    nheads, nkvheads = size(q, 2), size(k, 2)
    (nheads % nkvheads == 0) || throw(ArgumentError(
        "Number of query heads ($nheads) must be divisible by the number of \
         key/value heads ($nkvheads)."))
    (is_causal && mask !== nothing) && throw(ArgumentError(
        "`mask` and `is_causal=true` cannot be specified at the same time."))

    if is_causal
        mask = make_causal_mask(q, size(k, 3), size(q, 3))
    end

    batch_size = size(q)[4:end]
    q, k = map(x -> reshape(x, size(x, 1), size(x, 2), size(x, 3), :), (q, k))
    if nkvheads != nheads
        k = repeat_kv_heads(k, nheads ÷ nkvheads)
    end

    α = _attention_scores(q, k, bias, fdrop, mask, scale)
    return reshape(α, size(α)[1:3]..., batch_size...)
end

function _attention_scores(q::AA4{T}, k::AA4, bias, fdrop, mask, scale) where T
    s = scale === nothing ? √T(size(q, 1)) : T(scale)
    # The following permutedims and batched_mul are equivalent to
    # @tullio logits[j, i, h, b] := q[d, h, i, b] * k[d, h, j, b] / s
    kt = permutedims(k, (3, 1, 2, 4))
    qt = permutedims(q, (1, 3, 2, 4)) ./ s
    logits = batched_mul(kt, qt)
    # [logits] = [kv_len, q_len, nheads, batch_size]

    logits = apply_attn_bias(logits, bias)
    logits = apply_attn_mask(logits, mask)

    α = softmax(logits, dims=1)
    return fdrop(α)
end

apply_attn_bias(logits, bias::Nothing) = logits

apply_attn_bias(logits, bias) = logits .+ bias

apply_attn_mask(logits, mask::Nothing) = logits

function apply_attn_mask(logits, mask)
    neginf = typemin(eltype(logits))
    ifelse.(mask, logits, neginf)
end


"""
    make_causal_mask(x, dims=3)

Return a boolean square matrix `m` of the same type as `x` and of side `size(x, dims)`.
Its elements are set such that `m[i, j] == i ≤ j`.

Can be used to mask the attention scores in [`scaled_dot_product_attention`](@ref),
whose inputs have the sequence-length along `dims=3`.
"""
function make_causal_mask(x::AbstractArray; dims::Int=3)
  len = size(x, dims)
  mask = triu(trues_like(x, (len, len)))
  return mask
end

# Rectangular causal mask of size `(kvlen, qlen)` with `m[j, i] == j ≤ i`,
# used internally by `scaled_dot_product_attention(...; is_causal=true)`.
make_causal_mask(x::AbstractArray, kvlen::Int, qlen::Int) = triu(trues_like(x, (kvlen, qlen)))

trues_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), true)
falses_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), false)

split_heads(x, nheads) = reshape(x, size(x, 1) ÷ nheads, nheads, size(x)[2:end]...)
join_heads(x) = reshape(x, :, size(x)[3:end]...)

# Grouped-query attention: repeat each kv head `r` times contiguously along the
# heads dimension, so it lines up with the query heads.
# [x] = [head_dim, nkvheads, len, batch] -> [head_dim, nkvheads * r, len, batch]
repeat_kv_heads(x::AA4, r::Int) = repeat(x; inner=(1, r, 1, 1))

@non_differentiable make_causal_mask(::Any...)
@non_differentiable trues_like(::Any...)
@non_differentiable falses_like(::Any...)
