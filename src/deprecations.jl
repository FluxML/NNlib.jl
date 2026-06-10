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


# Old interface: heads are packed into the feature dimension and split via `nheads`,
# inputs are `(features, seq_len, batch...)`, and the output heads are joined back.
# The new `scaled_dot_product_attention` takes an explicit head axis instead.
function dot_product_attention(q, k, v, bias=nothing;
            nheads=1, nkvheads=nheads, kws...)
    Base.depwarn("`dot_product_attention(q, k, v; nheads)` is deprecated, use \
        `scaled_dot_product_attention(q, k, v)` with an explicit head axis \
        `(head_dim, nheads, seq_len, batch...)` instead.", :dot_product_attention)
    qh = split_heads(q, nheads)
    kh = split_heads(k, nkvheads)
    vh = split_heads(v, nkvheads)
    x, α = _scaled_dot_product_attention(qh, kh, vh, bias; kws...)
    return join_heads(x), α
end

# Old `dot_product_attention_scores` already used the explicit-head 4D convention.
function dot_product_attention_scores(q, k, bias=nothing; kws...)
    Base.depwarn("`dot_product_attention_scores` is deprecated, use \
        `scaled_dot_product_attention_scores` instead.", :dot_product_attention_scores)
    scaled_dot_product_attention_scores(q, k, bias; kws...)
end
