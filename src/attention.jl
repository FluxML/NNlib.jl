const AA3{T} = AbstractArray{T,3}
const AA4{T} = AbstractArray{T,4}
const AA{N,T} = AbstractArray{T,N}

"""
    scaled_dot_product_attention(query, key, value, [bias]; [fdrop, mask, scale, is_causal, q_lengths, kv_lengths])
    sdpa(query, key, value, [bias]; kws...)

Multihead dot product attention used in transformer architectures, with an explicit
head axis.

The input arrays must have the shape `(head_dim, nheads, seq_len, batch_size...)`, that
is a feature (head) dimension, a heads dimension, a sequence-length dimension, then an
arbitrary number of batch dimensions or none. The number of heads is taken from the
second dimension of each input, so no `nheads` keyword is needed.

This is the reverse of the `(batch_size, seq_len, nheads, head_dim)` layout of JAX and
Flax. It is not the reverse of PyTorch's `(batch_size, nheads, seq_len, head_dim)`: an
array ported from PyTorch by reversing its dimensions has size
`(head_dim, seq_len, nheads, batch_size)` and still needs `permutedims(x, (1, 3, 2, 4))`.
A missing permutation cannot be detected, since the array is a valid input whose heads
are taken as the sequence.

Grouped-query attention (GQA) is supported: if `key` and `value` have fewer heads than
`query`, they are shared across query heads. The number of query heads must then be
divisible by the number of key/value heads.

Returns the attention output array of size `(v_head_dim, nheads, q_len, batch_size...)`.

See also [`scaled_dot_product_attention_with_weights`](@ref) if you need the attention
weights too.

# Arguments

- `query`: Query array of size `(qk_head_dim, nheads, q_len, batch_size...)`.
- `key`: Key array of size `(qk_head_dim, nkvheads, kv_len, batch_size...)`.
- `value`: Value array of size `(v_head_dim, nkvheads, kv_len, batch_size...)`.
- `bias`: Either `nothing` or an array broadcastable to size `(kv_len, q_len, nheads, batch_size...)`.
          It will be added to the attention logits before applying the softmax. Default `nothing`.
- `fdrop`: A dropout function or layer to be applied on the attention weights right after the softmax.
           Default `identity` (no dropout).
- `mask`: Either `nothing` or a boolean array broadcastable to size `(kv_len, q_len, nheads, batch_size...)`.
          The mask is applied to the attention logits just before the softmax.
          See [`make_causal_mask`](@ref) for creating causal masks. Default `nothing`.
- `scale`: The multiplier used to scale the `queryᵀ * key` dot products before
          the softmax. Default `nothing`, meaning `1 / √(qk_head_dim)`.
- `is_causal`: If `true`, a causal mask is applied so that each query position can
          only attend to key positions up to and including its own. Equivalent to
          `mask = make_causal_mask(query; dims=3)` for equal query and key lengths, but
          a fused implementation does not have to materialize the mask. Default `false`.
- `q_lengths`, `kv_lengths`: Either `nothing` or the number of valid positions of each
          query and of each key/value sequence in the batch, the following ones being
          padding: an integer array with one entry for each batch element, or an
          integer for all of them. The keys past `kv_lengths` are not attended to, and
          the output is zero at the queries past `q_lengths`. Each can be given without
          the other. Default `nothing`.

`mask`, `is_causal` and `kv_lengths` can be combined: a key is then attended to only
where all of them allow it.

# Fused implementation

Without an `fdrop`, the output is computed tile by tile, with the softmax taken online
over the tiles of keys: the attention weights are never materialized, so that memory
grows linearly rather than quadratically with the sequence length. This is what `Array`s
and GPU arrays of `Float16`, `Float32` and `Float64` take, and on the GPU those of
`BFloat16`, with any of the arguments above. The `bias` and the `mask`, if any, have to
be on the device of the inputs.

Any other call takes the materialized implementation, which forms the weights.
[`NNlib.is_fused_attention`](@ref) tells which of the two a call takes.

The fused implementation has a gradient with respect to `query`, `key`, `value` and
`scale`, which recomputes the weights tile by tile and does not materialize them either.
It is what automatic differentiation based on ChainRules (as that of Zygote) and the
reverse mode of Enzyme use. Under them a call with a `bias` takes the materialized
implementation, which gives the gradient with respect to the bias too, and so does every
call under the other automatic differentiation packages. A query that attends to no key,
whose output is `NaN`, takes no part in the fused gradient.

With CUDA.jl and cuDNN.jl loaded, the `Float16` and `BFloat16` `CuArray`s that cuDNN's
own fused kernel has an engine for are handed to it: it runs on tensor cores, and has a
gradient. It supports `scale`, `is_causal`, `q_lengths`, `kv_lengths` and grouped-query
attention, and requires:

- no `bias`, `fdrop` or `mask` (that is `nothing`, `identity` and `nothing`);
- a head dimension that is a multiple of 8 and at most 256;
- a `value` with the same head dimension as `query`.

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

# a padded batch, whose first key/value sequence has 12 valid positions out of 30
y = scaled_dot_product_attention(q, k, v; kv_lengths=[12, 30, 30, 30])
```
"""
function scaled_dot_product_attention(q::AbstractArray, k::AbstractArray, v::AbstractArray,
            bias=nothing; fdrop=identity, mask=nothing, scale=nothing,
            is_causal::Bool=false, q_lengths=nothing, kv_lengths=nothing)
    check_attention_inputs(q, k, v)
    batch_size = size(q)[4:end]
    q, k, v, bias, mask, q_lengths, kv_lengths =
        collapse_attention_inputs(q, k, v, bias, mask, q_lengths, kv_lengths)
    x = _scaled_dot_product_attention(q, k, v, bias; fdrop, mask, scale, is_causal,
                                      q_lengths, kv_lengths)
    return reshape(x, size(x, 1), size(x, 2), size(x, 3), batch_size...)
end

const sdpa = scaled_dot_product_attention

# Attention on `(head_dim, nheads, seq_len, batch_size)` arrays, with `q_lengths` and
# `kv_lengths` as `(1, 1, 1, batch_size)` arrays or `nothing`. This is the method for
# backends to overload: it receives the arguments as the user gave them, before the
# materialized implementation builds the causal and padding masks, the repeated
# key/value heads of grouped-query attention and the attention weights, none of which a
# fused kernel needs.
#
_scaled_dot_product_attention(q::AA4, k::AA4, v::AA4, bias; kws...) =
    generic_attention(q, k, v, bias; kws...)

# The implementation for every backend, which those with a kernel of their own fall back
# to for the calls it does not take: the fused one where it applies, else the
# materialized one. Under automatic differentiation the latter is also taken, and
# differentiated, where the fused one has no rule.
function generic_attention(q::AA4, k::AA4, v::AA4, bias;
            fdrop, mask, scale, is_causal, q_lengths, kv_lengths)
    if fused_attention_applies(q, k, v, bias, fdrop, mask) && fused_attention_differentiable(q, bias)
        return fused_attention(q, k, v, bias, mask, scale, is_causal, q_lengths, kv_lengths)
    end
    x, _ = _attention(q, k, v, bias; fdrop, mask, scale, is_causal, q_lengths, kv_lengths)
    return x
end

"""
    scaled_dot_product_attention_with_weights(query, key, value, [bias]; kws...)
    sdpa_with_weights(query, key, value, [bias]; kws...)

Return the output of [`scaled_dot_product_attention`](@ref) together with the attention
weights it is computed from, in a single pass.

The arguments are those of [`scaled_dot_product_attention`](@ref). The weights have size
`(kv_len, q_len, nheads, batch_size...)`, with `α[j, i, h, b]` the weight of the key `j`
for the query `i` in the head `h`: the softmax over the keys of the attention logits,
after `fdrop`. They are zero at the queries past `q_lengths`.

Returning the weights means materializing them, so this function never takes a fused
implementation: prefer [`scaled_dot_product_attention`](@ref) when they are not needed.

# Examples

```julia
q = rand(Float32, 16, 8, 20, 4)
k = v = rand(Float32, 16, 8, 30, 4)
y, α = scaled_dot_product_attention_with_weights(q, k, v)
# size(y) == (16, 8, 20, 4)
# size(α) == (30, 20, 8, 4)
```
"""
function scaled_dot_product_attention_with_weights(q::AbstractArray, k::AbstractArray,
            v::AbstractArray, bias=nothing; fdrop=identity, mask=nothing, scale=nothing,
            is_causal::Bool=false, q_lengths=nothing, kv_lengths=nothing)
    check_attention_inputs(q, k, v)
    batch_size = size(q)[4:end]
    q, k, v, bias, mask, q_lengths, kv_lengths =
        collapse_attention_inputs(q, k, v, bias, mask, q_lengths, kv_lengths)
    x, α = _attention(q, k, v, bias; fdrop, mask, scale, is_causal, q_lengths, kv_lengths)
    x = reshape(x, size(x, 1), size(x, 2), size(x, 3), batch_size...)
    α = reshape(α, size(α, 1), size(α, 2), size(α, 3), batch_size...)
    return x, α
end

const sdpa_with_weights = scaled_dot_product_attention_with_weights

"""
    NNlib.is_fused_attention(query, key, value, [bias]; kws...)

Return `true` if [`scaled_dot_product_attention`](@ref) called with the same arguments
is computed by a fused implementation, which does not materialize the attention weights,
and `false` if it takes the materialized one.

This is about a call outside of automatic differentiation: under it, the materialized
implementation is differentiated where the fused one has no gradient, as with a `bias`.
"""
function is_fused_attention(q::AbstractArray, k::AbstractArray, v::AbstractArray,
            bias=nothing; fdrop=identity, mask=nothing, scale=nothing,
            is_causal::Bool=false, q_lengths=nothing, kv_lengths=nothing)
    check_attention_inputs(q, k, v)
    q, k, v, bias, mask, q_lengths, kv_lengths =
        collapse_attention_inputs(q, k, v, bias, mask, q_lengths, kv_lengths)
    return _is_fused_attention(q, k, v, bias; fdrop, mask, scale, is_causal,
                               q_lengths, kv_lengths)
end

# To be overloaded together with `_scaled_dot_product_attention`.
_is_fused_attention(q::AA4, k::AA4, v::AA4, bias; fdrop, mask, kws...) =
    fused_attention_applies(q, k, v, bias, fdrop, mask)

function check_attention_inputs(q::AbstractArray, k::AbstractArray, v::AbstractArray)
    ndims(q) == ndims(k) == ndims(v) || throw(ArgumentError("""
    Query, key and value must have the same number of dimensions. Instead:
    - size(q): $(size(q))
    - size(k): $(size(k))
    - size(v): $(size(v))
    """))
    ndims(q) >= 3 || throw(ArgumentError(
        "Inputs must have at least 3 dimensions (head_dim, nheads, seq_len). Got $(ndims(q))."))

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
    return nothing
end

@non_differentiable check_attention_inputs(::Any...)

# Bring the arguments of the public functions to the layout of the implementations:
# `q`, `k` and `v` with a single batch dimension, `bias` and `mask` broadcastable against
# it, the lengths as `(1, 1, 1, batch_size)` arrays on the device of the inputs.
function collapse_attention_inputs(q, k, v, bias, mask, q_lengths, kv_lengths)
    batch_size = size(q)[4:end]
    q, k, v = map(collapse_batch_dims, (q, k, v))
    bias = collapse_batch_dims(bias, batch_size)
    mask = collapse_batch_dims(mask, batch_size)
    q_lengths = attention_lengths(q, q_lengths)
    kv_lengths = attention_lengths(k, kv_lengths)
    return q, k, v, bias, mask, q_lengths, kv_lengths
end

# collapse the batch dimensions into a single one
collapse_batch_dims(x) = reshape(x, size(x, 1), size(x, 2), size(x, 3), :)

# The same for a `bias` or a `mask`, which is only broadcastable to
# `(kv_len, q_len, nheads, batch_size...)`.
collapse_batch_dims(x::Nothing, batch_size::Tuple) = nothing

function collapse_batch_dims(x::AbstractArray, batch_size::Tuple)
    (length(batch_size) <= 1 || ndims(x) <= 3) && return x
    if needs_batch_expansion(size(x), batch_size)
        x = expand_batch_dims(x, batch_size)
    end
    return reshape(x, size(x, 1), size(x, 2), size(x, 3), :)
end

# Singleton batch dimensions have to be materialized before being collapsed, unless
# they all are.
function needs_batch_expansion(sz::Tuple, batch_size::Tuple)
    xbatch = sz[4:end]
    xbatch == (prod(batch_size),) && return false   # already collapsed
    length(xbatch) <= length(batch_size) &&
        all(i -> xbatch[i] in (1, batch_size[i]), eachindex(xbatch)) ||
        throw(DimensionMismatch("A bias or mask of size $sz cannot be broadcast to the \
            batch dimensions $batch_size of the attention inputs."))
    return xbatch != batch_size && any(!=(1), xbatch)
end

@non_differentiable needs_batch_expansion(::Any...)

expand_batch_dims(x::AbstractArray{Bool}, batch_size::Tuple) =
    x .| falses_like(x, (1, 1, 1, batch_size...))
expand_batch_dims(x::AbstractArray, batch_size::Tuple) =
    x .+ falses_like(x, (1, 1, 1, batch_size...))

# Sequence lengths as an `Int32` array of size `(1, 1, 1, batch_size)` on the device of
# `x`, a `(head_dim, nheads, seq_len, batch_size)` array. The lengths are not restricted
# to `Integer`s, which the traced integers of Reactant are not.
attention_lengths(x::AA4, lengths::Nothing) = nothing

function attention_lengths(x::AA4, lengths::Union{Number,AbstractArray})
    out = similar(x, Int32, (1, 1, 1, size(x, 4)))
    lengths isa Number && return fill!(out, clamp(lengths, 0, size(x, 3)))
    length(lengths) == length(out) || throw(ArgumentError(
        "Expected a sequence length for each of the $(length(out)) batch elements, \
         got $(length(lengths))."))
    return copyto!(out, Int32.(clamp.(vec(lengths), 0, size(x, 3))))
end

@non_differentiable attention_lengths(::Any...)

# Materialized implementation, returning both the attention output and the weights `α`.
function _attention(q::AA4, k::AA4, v::AA4, bias; fdrop, mask, scale, is_causal,
                    q_lengths=nothing, kv_lengths=nothing)
    # [q] = [qk_head_dim, nheads, q_len, batch_size]
    # [k] = [qk_head_dim, nkvheads, kv_len, batch_size]
    # [v] = [v_head_dim,  nkvheads, kv_len, batch_size]
    mask = attention_mask(q, k, mask, is_causal, kv_lengths)
    nheads, nkvheads = size(q, 2), size(k, 2)
    if nkvheads != nheads
        # Grouped-query attention: expand the kv heads so each group of
        # `nheads ÷ nkvheads` query heads shares one key/value head.
        r = nheads ÷ nkvheads
        k = repeat_kv_heads(k, r)
        v = repeat_kv_heads(v, r)
    end

    α = _attention_weights(q, k, bias, fdrop, mask, scale, q_lengths)
    # [α] = [kv_len, q_len, nheads, batch_size]

    # The following permutedims and batched_mul are equivalent to
    # @tullio x[d, h, i, b] := α[j, i, h, b] * v[d, h, j, b]
    vt = permutedims(v, (1, 3, 2, 4))
    x = batched_mul(vt, α)
    x = permutedims(x, (1, 3, 2, 4))
    # [x] = [v_head_dim, nheads, q_len, batch_size]
    return x, α
end

function _attention_weights(q::AA4{T}, k::AA4, bias, fdrop, mask, scale, q_lengths=nothing) where T
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
    α = apply_query_lengths(α, q_lengths)
    return fdrop(α)
end

apply_attn_bias(logits, bias::Nothing) = logits

apply_attn_bias(logits, bias) = logits .+ bias

apply_attn_mask(logits, mask::Nothing) = logits

function apply_attn_mask(logits, mask)
    neginf = typemin(eltype(logits))
    ifelse.(mask, logits, neginf)
end

# Zeroing the weights of the padded queries zeroes the attention output there too.
apply_query_lengths(α, q_lengths::Nothing) = α

apply_query_lengths(α, q_lengths) = α .* padding_mask(size(α, 2), q_lengths; dims=2)

# The mask of the materialized implementation: the `mask` of the user, the causal one
# and the one of the padded keys, a key being attended to only where all of them allow it.
function attention_mask(q::AA4, k::AA4, mask, is_causal::Bool, kv_lengths)
    if is_causal
        mask = combine_masks(mask, make_causal_mask(q, size(k, 3), size(q, 3)))
    end
    if kv_lengths !== nothing
        mask = combine_masks(mask, padding_mask(size(k, 3), kv_lengths; dims=1))
    end
    return mask
end

combine_masks(mask::Nothing, m) = m
combine_masks(mask, m) = mask .& m

# Boolean mask of the positions `1:len` of the dimension `dims` (1 or 2) that are within
# the `lengths`, a `(1, 1, 1, batch_size)` array.
padding_mask(len::Int, lengths; dims::Int) = (dims == 1 ? (1:len) : (1:len)') .<= lengths

@non_differentiable attention_mask(::Any...)
@non_differentiable padding_mask(::Any...)


"""
    make_causal_mask(x; dims)

Return a boolean square matrix `m` of the same type as `x` and of side `size(x, dims)`.
Its elements are set such that `m[i, j] == i ≤ j`.

Can be used as the `mask` of [`scaled_dot_product_attention`](@ref),
whose inputs have the sequence length along `dims=3`. For a plain causal mask, prefer
the `is_causal=true` keyword there, which lets a fused implementation avoid materializing
the mask.

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
