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
- `scale`: The multiplier used to scale the `queryᵀ * key` dot products before
          the softmax. Default `nothing`, meaning `1 / √(qk_head_dim)`.
- `is_causal`: If `true`, a causal mask is applied so that each query position can
          only attend to key positions up to and including its own. Equivalent to
          `mask = make_causal_mask(query; dims=3)` for equal query and key lengths, but
          a fused implementation does not have to materialize the mask. Cannot be
          combined with `mask`. Default `false`.

# Fused GPU implementation

With CUDA.jl and cuDNN.jl loaded, `Float16` and `BFloat16` `CuArray`s are handled by
cuDNN's fused kernel, which computes the output without materializing the attention
scores, so that memory grows linearly rather than quadratically with the sequence
length. It supports `scale`, `is_causal` and grouped-query attention, and requires:

- no `bias`, `fdrop` or `mask` (that is `nothing`, `identity` and `nothing`);
- a head dimension that is a multiple of 8 and at most 256;
- a `value` with the same head dimension as `query`.

Any other call takes the generic implementation, which materializes the scores.

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
            fdrop=identity, mask=nothing, scale=nothing, is_causal::Bool=false) where N
    check_attention_inputs(q, k, v, mask, is_causal)
    batch_size = size(q)[4:end]
    q, k, v = map(collapse_batch_dims, (q, k, v))
    x = _scaled_dot_product_attention(q, k, v, bias; fdrop, mask, scale, is_causal)
    return reshape(x, size(x, 1), size(x, 2), size(x, 3), batch_size...)
end

# Attention on `(head_dim, nheads, seq_len, batch_size)` arrays. This is the method for
# backends to overload: it receives the arguments as the user gave them, before the
# generic implementation materializes the causal mask, the repeated key/value heads of
# grouped-query attention and the attention scores, none of which a fused kernel needs.
function _scaled_dot_product_attention(q::AA4, k::AA4, v::AA4, bias;
            fdrop, mask, scale, is_causal)
    x, _ = _attention(q, k, v, bias; fdrop, mask, scale, is_causal)
    return x
end

function check_attention_inputs(q::AA{N}, k::AA{N}, v::AA{N}, mask, is_causal) where N
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
    return nothing
end

@non_differentiable check_attention_inputs(::Any...)

# collapse the batch dimensions into a single one
collapse_batch_dims(x) = reshape(x, size(x, 1), size(x, 2), size(x, 3), :)

# Generic implementation, returning both the attention output and the scores `α`,
# so the (deprecated) `dot_product_attention` can return a consistent `(x, α)` pair
# without applying a stochastic `fdrop` twice.
function _attention(q::AA4, k::AA4, v::AA4, bias; fdrop, mask, scale, is_causal)
    # [q] = [qk_head_dim, nheads, q_len, batch_size]
    # [k] = [qk_head_dim, nkvheads, kv_len, batch_size]
    # [v] = [v_head_dim,  nkvheads, kv_len, batch_size]
    if is_causal
        mask = make_causal_mask(q, size(k, 3), size(q, 3))
    end
    nheads, nkvheads = size(q, 2), size(k, 2)
    if nkvheads != nheads
        # Grouped-query attention: expand the kv heads so each group of
        # `nheads ÷ nkvheads` query heads shares one key/value head.
        r = nheads ÷ nkvheads
        k = repeat_kv_heads(k, r)
        v = repeat_kv_heads(v, r)
    end

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

The `scale` keyword sets the multiplier scaling the `queryᵀ * key` dot products;
it defaults to `1 / √(head_dim)`.

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
    q, k = map(collapse_batch_dims, (q, k))
    if nkvheads != nheads
        k = repeat_kv_heads(k, nheads ÷ nkvheads)
    end

    α = _attention_scores(q, k, bias, fdrop, mask, scale)
    return reshape(α, size(α)[1:3]..., batch_size...)
end

function _attention_scores(q::AA4{T}, k::AA4, bias, fdrop, mask, scale) where T
    s = scale === nothing ? inv(√T(size(q, 1))) : T(scale)
    # The following permutedims and batched_mul are equivalent to
    # @tullio logits[j, i, h, b] := s * q[d, h, i, b] * k[d, h, j, b]
    kt = permutedims(k, (3, 1, 2, 4))
    qt = permutedims(q, (1, 3, 2, 4)) .* s
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
    make_causal_mask(x; dims)

Return a boolean square matrix `m` of the same type as `x` and of side `size(x, dims)`.
Its elements are set such that `m[i, j] == i ≤ j`.

Can be used to mask the attention scores in [`scaled_dot_product_attention`](@ref),
whose inputs have the sequence length along `dims=3`. For a plain causal mask, prefer
the `is_causal=true` keyword there, which lets backends with a fused implementation
avoid materializing the mask.

Calling `make_causal_mask(x)` without `dims` is deprecated. It keeps returning the mask
for `dims=2`, the sequence dimension of the deprecated `dot_product_attention`.
"""
function make_causal_mask(x::AbstractArray; dims::Union{Int,Nothing}=nothing)
  if dims === nothing
    Base.depwarn("`make_causal_mask(x)` without the `dims` keyword is deprecated. \
        Pass `dims=3` for the inputs of `scaled_dot_product_attention` (or use its \
        `is_causal=true` keyword), `dims=2` for the deprecated `dot_product_attention`.",
        :make_causal_mask)
    dims = 2
  end
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

function rrule(::typeof(repeat_kv_heads), x::AA4, r::Int)
    y = repeat_kv_heads(x, r)
    function repeat_kv_heads_pullback(Δ)
        Δ = unthunk(Δ)
        Δ = reshape(Δ, size(x, 1), r, size(x, 2), size(x, 3), size(x, 4))
        dx = reshape(sum(Δ; dims=2), size(x))
        return NoTangent(), dx, NoTangent()
    end
    return y, repeat_kv_heads_pullback
end

@non_differentiable make_causal_mask(::Any...)
@non_differentiable trues_like(::Any...)
@non_differentiable falses_like(::Any...)
