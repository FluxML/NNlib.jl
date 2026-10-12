### Deprecated in v0.9.37

# The softmax gradient helpers lost their `_data` suffix: the `x`-free
# `∇softmax(dy, y)` / `∇logsoftmax(dy, y)` are now the public spelling.
function ∇softmax_data(dy::AbstractArray, y::AbstractArray; dims = 1)
    Base.depwarn("`∇softmax_data(dy, y)` is deprecated, use `∇softmax(dy, y)` instead.", :∇softmax_data)
    return ∇softmax(dy, y; dims)
end

function ∇logsoftmax_data(dy::AbstractArray, y::AbstractArray; dims = 1)
    Base.depwarn("`∇logsoftmax_data(dy, y)` is deprecated, use `∇logsoftmax(dy, y)` instead.", :∇logsoftmax_data)
    return ∇logsoftmax(dy, y; dims)
end


### Deprecated in v0.9.46

"""
    dot_product_attention(query, key, value, [bias]; [nheads, fdrop, mask])

Deprecated: use [`scaled_dot_product_attention`](@ref), which takes inputs with an
explicit head axis `(head_dim, nheads, seq_len, batch_size...)` and returns only the
attention output, and [`scaled_dot_product_attention_scores`](@ref) for the scores.

Multihead dot product attention on inputs of size `(features, seq_len, batch_size...)`,
whose heads are packed into the feature dimension and split according to `nheads`.
Returns the attention output of size `(v_dim, q_len, batch_size...)` and the attention
scores of size `(kv_len, q_len, nheads, batch_size...)`.
"""
function dot_product_attention(q, k, v, bias=nothing;
            nheads=1, nkvheads=nheads, fdrop=identity, mask=nothing, scale=nothing,
            is_causal::Bool=false)
    Base.depwarn("`dot_product_attention(q, k, v; nheads)` is deprecated, use \
        `scaled_dot_product_attention(q, k, v)` with an explicit head axis \
        `(head_dim, nheads, seq_len, batch...)` instead.", :dot_product_attention)
    q = split_heads(q, nheads)
    k = split_heads(k, nkvheads)
    v = split_heads(v, nkvheads)
    check_attention_inputs(q, k, v)
    batch_size = size(q)[4:end]
    q, k, v, bias, mask, _, _ = collapse_attention_inputs(q, k, v, bias, mask, nothing, nothing)
    # the generic implementation, as this interface also returns the scores
    x, α = _attention(q, k, v, bias; fdrop, mask, scale, is_causal)
    x = reshape(x, size(x, 1), size(x, 2), size(x, 3), batch_size...)
    α = reshape(α, size(α)[1:3]..., batch_size...)
    return join_heads(x), α
end

"""
    dot_product_attention_scores(query, key, [bias]; [fdrop, mask])

Deprecated: use [`scaled_dot_product_attention_scores`](@ref).
"""
function dot_product_attention_scores(q, k, bias=nothing; kws...)
    Base.depwarn("`dot_product_attention_scores` is deprecated, use \
        `scaled_dot_product_attention_scores` instead.", :dot_product_attention_scores)
    scaled_dot_product_attention_scores(q, k, bias; kws...)
end

export dot_product_attention, dot_product_attention_scores
