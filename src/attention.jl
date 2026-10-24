const AA3{T} = AbstractArray{T,3}
const AA4{T} = AbstractArray{T,4}
const AA{N,T} = AbstractArray{T,N}

"""
    scaled_dot_product_attention(query, key, value, [bias]; [fdrop, mask, scale, is_causal, window, q_offset, q_lengths, kv_lengths, q_segments, kv_segments, maxlen])
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
          a fused implementation does not have to materialize the mask. If
          `:bottom_right`, the mask is aligned with the last valid key instead of the
          first: the queries are taken as the last positions of the keys, as when
          decoding with a KV cache, and `q_offset` is `kv_lengths .- q_lengths`, these
          being the sizes of the arrays where not given. This is the causal mask of
          FlashAttention and PyTorch's `causal_lower_right`. Default `false`.
- `window`: Either `nothing` or a positive integer, the size of a sliding window: the
          query `i` attends only to the keys `j` with `|q_offset + i - j| < window`. With
          `is_causal`, these are the last `window` keys up to its own position,
          `q_offset + i - window < j ≤ q_offset + i`, as in the sliding-window layers of
          Mistral, Gemma or gpt-oss. This is the `sliding_window` of transformers, and the
          `window_size` of FlashAttention is `(window - 1, 0)` with `is_causal`,
          `(window - 1, window - 1)` without. A fused implementation only goes through
          the keys of the windows. Default `nothing`.
- `q_offset`: With `is_causal=true`, the number of keys that precede the first query:
          the query `i` attends to the keys `j ≤ q_offset + i`. An integer, or an integer
          array with one entry for each batch element. It counts the keys given, which
          with a sliding-window cache is not the absolute position. It has no effect
          without `is_causal` or `window`, and is set by `is_causal=:bottom_right`.
          Default `0`.
- `q_lengths`, `kv_lengths`: Either `nothing` or the number of valid positions of each
          query and of each key/value sequence in the batch, the following ones being
          padding: an integer array with one entry for each batch element, or an
          integer for all of them. The keys past `kv_lengths` are not attended to, and
          the output is zero at the queries past `q_lengths`. Each can be given without
          the other. Default `nothing`.
- `q_segments`, `kv_segments`: Either `nothing` or the lengths of the sequences packed
          one after the other along the sequence dimension of `query` and of `key` and
          `value`, as an integer vector. See below. `kv_segments` defaults to `q_segments`.
- `maxlen`: With `q_segments`, either `nothing` or an integer at least as large as every
          segment. It saves reading the lengths of the segments on the host where they
          are on a GPU, which waits for the device, and stands in for them where they
          cannot be read, as the traced ones of Reactant. Default `nothing`.

`mask`, `is_causal`, `window` and `kv_lengths` can be combined: a key is then attended to
only where all of them allow it. A query that attends to no key, as a padding query past
the window of every valid key, has a `NaN` output, unless `q_lengths` makes it zero.

# Packed sequences

With `q_segments`, the inputs hold sequences of different lengths one after the other,
without padding: they have size `(head_dim, nheads, total_len)`, without batch
dimensions, and the output has the size of `query`. The `s`-th segment of `query`
attends to the `s`-th segment of `key` and `value` only, as a batch element does in a
padded batch: `is_causal` and `window` apply within each segment, and `q_offset`,
`q_lengths` and `kv_lengths` take one entry for each segment. The lengths of the segments
have to add up to the sequence lengths of the arrays. A `bias` or a `mask` is not
supported.

This is how documents packed in a sequence are kept from attending to each other, and
how a batch of sequences of different lengths avoids padding: layers other than the
attention act on the `(features, total_len)` array of all the positions.

# Fused implementation

Without an `fdrop`, the output is computed tile by tile, with the softmax taken online
over the tiles of keys: the attention weights are never materialized, so that memory
grows linearly rather than quadratically with the sequence length. This is what `Array`s
and GPU arrays of `Float16`, `Float32` and `Float64` take, and on the GPU those of
`BFloat16`, with any of the arguments above. The `bias` and the `mask`, if any, have to
be on the device of the inputs.

Any other call takes the materialized implementation, which forms the weights.
[`NNlib.is_fused_attention`](@ref) tells which of the two a call takes. With a `window`,
it forms them over the band of the window only, for blocks of queries with the keys of
their windows, so that memory grows linearly with the sequence length there too. `fdrop`
is then given these blocks of weights rather than the whole array.

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

- no `bias`, `fdrop`, `mask` or `window` (that is `nothing`, `identity`, `nothing` and
  `nothing`);
- with `is_causal`, a causal mask aligned with the first key: the default `q_offset` of
  zero, and for `:bottom_right` as many queries as keys, without `q_lengths` or
  `kv_lengths`;
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

# decoding with a KV cache of 64 positions: 3 new queries for each batch element,
# appended to the 10, 20, 30 and 40 keys already cached
q = rand(Float32, 16, 8, 3, 4)
k = v = rand(Float32, 16, 8, 64, 4)
y = scaled_dot_product_attention(q, k, v; is_causal=:bottom_right, kv_lengths=[13, 23, 33, 43])
# the same, with the offsets of the queries
y = scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=[10, 20, 30, 40],
                                 kv_lengths=[13, 23, 33, 43])

# sliding-window attention: each query attends to its last 8 keys, its own included
x = rand(Float32, 16, 8, 100, 4)
y = scaled_dot_product_attention(x, x, x; is_causal=true, window=8)

# three documents of 5, 12 and 3 positions packed in one sequence, each attending
# causally to itself only
x = rand(Float32, 16, 8, 20)
y = scaled_dot_product_attention(x, x, x; is_causal=true, q_segments=[5, 12, 3])
# size(y) == (16, 8, 20)
```
"""
function scaled_dot_product_attention(q::AbstractArray, k::AbstractArray, v::AbstractArray,
            bias=nothing; fdrop=identity, mask=nothing, scale=nothing,
            is_causal::Union{Bool,Symbol}=false, window=nothing, q_offset=0, q_lengths=nothing,
            kv_lengths=nothing, q_segments=nothing, kv_segments=q_segments, maxlen=nothing)
    check_attention_inputs(q, k, v)
    check_causal(is_causal, q_offset)
    check_window(window)
    batch_size = size(q)[4:end]
    q, k, v, bias, mask, q_offset, q_lengths, kv_lengths, segments =
        collapse_attention_inputs(q, k, v, bias, mask, q_offset, q_lengths, kv_lengths,
                                  q_segments, kv_segments, maxlen)
    x = _scaled_dot_product_attention(q, k, v, bias; fdrop, mask, scale, is_causal, window,
                                      q_offset, q_lengths, kv_lengths, segments)
    return reshape(x, size(x, 1), size(x, 2), size(x, 3), batch_size...)
end

const sdpa = scaled_dot_product_attention

# Attention on `(head_dim, nheads, seq_len, batch_size)` arrays, with `q_lengths` and
# `kv_lengths` as `(1, 1, 1, batch_size)` arrays or `nothing`, and `q_offset` as such an
# array or an `Int`. With `segments`, an `AttentionSegments`, the arrays hold packed
# sequences, and those arguments have an entry for each segment. This is the method for
# backends to overload: it receives the arguments as the user gave them,
# `is_causal=:bottom_right` and `window` included, before the materialized implementation
# builds the causal, window and padding masks, the repeated key/value heads of
# grouped-query attention and the attention weights, none of which a fused kernel needs.
#
_scaled_dot_product_attention(q::AA4, k::AA4, v::AA4, bias; kws...) =
    generic_attention(q, k, v, bias; kws...)

# The implementation for every backend, which those with a kernel of their own fall back
# to for the calls it does not take: the fused one where it applies, else the
# materialized one. Under automatic differentiation the latter is also taken, and
# differentiated, where the fused one has no rule.
function generic_attention(q::AA4, k::AA4, v::AA4, bias;
            fdrop, mask, scale, is_causal, q_offset, q_lengths, kv_lengths, segments=nothing,
            window=nothing)
    is_causal, q_offset = causal_offsets(q, k, is_causal, q_offset, q_lengths, kv_lengths)
    is_causal, q_offset, window = attention_band(is_causal, q_offset, window)
    if fused_attention_applies(q, k, v, bias, fdrop, mask) && fused_attention_differentiable(q, bias) &&
            fused_attention_segments(segments)
        return fused_attention(q, k, v, bias, mask, scale, is_causal, q_offset, q_lengths, kv_lengths,
                               segments, window)
    end
    segments === nothing ||
        return packed_attention(q, k, v; fdrop, scale, is_causal, window, q_offset, q_lengths, kv_lengths,
                                segments)
    return materialized_attention(q, k, v, bias; fdrop, mask, scale, is_causal, window, q_offset, q_lengths,
                                  kv_lengths)
end

# The output of the materialized implementation, over the band of the window only where
# that has fewer pairs of a query and a key than the whole.
function materialized_attention(q::AA4, k::AA4, v::AA4, bias; fdrop, mask, scale, is_causal, window=nothing,
                                q_offset=0, q_lengths=nothing, kv_lengths=nothing, segments=nothing)
    is_causal, q_offset = causal_offsets(q, k, is_causal, q_offset, q_lengths, kv_lengths)
    is_causal, q_offset, window = attention_band(is_causal, q_offset, window)
    segments === nothing && banded_attention_cheaper(size(q, 1) + size(v, 1), size(q, 3), size(k, 3), window) &&
        return banded_attention(q, k, v, bias; fdrop, mask, scale, window, q_offset, q_lengths, kv_lengths)
    x, _ = _attention(q, k, v, bias; fdrop, mask, scale, is_causal, window, q_offset, q_lengths, kv_lengths,
                      segments)
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
after `fdrop`. They are zero at the queries past `q_lengths`. With `q_segments` they
have size `(kv_len, q_len, nheads)`, and are zero but for the keys of the segment of
each query.

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
            is_causal::Union{Bool,Symbol}=false, window=nothing, q_offset=0, q_lengths=nothing,
            kv_lengths=nothing, q_segments=nothing, kv_segments=q_segments, maxlen=nothing)
    check_attention_inputs(q, k, v)
    check_causal(is_causal, q_offset)
    check_window(window)
    batch_size = size(q)[4:end]
    q, k, v, bias, mask, q_offset, q_lengths, kv_lengths, segments =
        collapse_attention_inputs(q, k, v, bias, mask, q_offset, q_lengths, kv_lengths,
                                  q_segments, kv_segments, maxlen)
    x, α = _attention(q, k, v, bias; fdrop, mask, scale, is_causal, window, q_offset, q_lengths,
                      kv_lengths, segments)
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
            is_causal::Union{Bool,Symbol}=false, window=nothing, q_offset=0, q_lengths=nothing,
            kv_lengths=nothing, q_segments=nothing, kv_segments=q_segments, maxlen=nothing)
    check_attention_inputs(q, k, v)
    check_causal(is_causal, q_offset)
    check_window(window)
    q, k, v, bias, mask, q_offset, q_lengths, kv_lengths, segments =
        collapse_attention_inputs(q, k, v, bias, mask, q_offset, q_lengths, kv_lengths,
                                  q_segments, kv_segments, maxlen)
    return _is_fused_attention(q, k, v, bias; fdrop, mask, scale, is_causal, window,
                               q_offset, q_lengths, kv_lengths, segments)
end

# To be overloaded together with `_scaled_dot_product_attention`.
_is_fused_attention(q::AA4, k::AA4, v::AA4, bias; fdrop, mask, segments=nothing, kws...) =
    fused_attention_applies(q, k, v, bias, fdrop, mask) && fused_attention_segments(segments)

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

check_causal(is_causal::Bool, q_offset) = nothing

function check_causal(is_causal::Symbol, q_offset)
    is_causal === :bottom_right || throw(ArgumentError(
        "`is_causal` must be `true`, `false` or `:bottom_right`, got :$is_causal."))
    q_offset isa Integer && iszero(q_offset) || throw(ArgumentError(
        "`is_causal=:bottom_right` sets the offsets of the queries: leave `q_offset` at its default."))
    return nothing
end

@non_differentiable check_causal(::Any...)

check_window(window::Nothing) = nothing
check_window(window::Integer) = window >= 1 ? nothing : throw(ArgumentError(
    "`window` must be a positive integer, got $window."))
check_window(window) = throw(ArgumentError(
    "`window` must be `nothing` or a positive integer, got $(typeof(window))."))

@non_differentiable check_window(::Any...)

# Bring the arguments of the public functions to the layout of the implementations:
# `q`, `k` and `v` with a single batch dimension, `bias` and `mask` broadcastable against
# it, the lengths and the offsets but an `Integer` one as `(1, 1, 1, batch_size)` arrays
# on the device of the inputs. Packed sequences have a batch dimension of one, an
# `AttentionSegments`, and the lengths of their segments as `q_lengths` and `kv_lengths`
# where these are not given. Inlined: Enzyme takes the integer arrays of a returned tuple
# that also holds active arrays for active memory (EnzymeAD/Enzyme.jl#3873).
@inline function collapse_attention_inputs(q, k, v, bias, mask, q_offset, q_lengths, kv_lengths,
                                           q_segments=nothing, kv_segments=nothing, maxlen=nothing)
    batch_size = size(q)[4:end]
    check_segments(q, bias, mask, q_segments, kv_segments)
    q, k, v = map(collapse_batch_dims, (q, k, v))
    segments = attention_segments(q, k, q_segments, kv_segments, maxlen)
    if segments !== nothing
        n = length(segments.q_ends)
        q_offset = attention_offsets(q, q_offset, n)
        q_lengths = segment_lengths(q, q_lengths, segments.q_sizes)
        kv_lengths = segment_lengths(k, kv_lengths, segments.kv_sizes)
        return q, k, v, nothing, nothing, q_offset, q_lengths, kv_lengths, segments
    end
    bias = collapse_batch_dims(bias, batch_size)
    mask = collapse_batch_dims(mask, batch_size)
    q_offset = attention_offsets(q, q_offset)
    q_lengths = attention_lengths(q, q_lengths)
    kv_lengths = attention_lengths(k, kv_lengths)
    return q, k, v, bias, mask, q_offset, q_lengths, kv_lengths, nothing
end

check_segments(q, bias, mask, q_segments::Nothing, kv_segments::Nothing) = nothing
check_segments(q, bias, mask, q_segments::Nothing, kv_segments) =
    throw(ArgumentError("`kv_segments` is given without `q_segments`."))

function check_segments(q, bias, mask, q_segments, kv_segments)
    ndims(q) == 3 || throw(ArgumentError(
        "Packed sequences have size `(head_dim, nheads, total_len)`, without batch \
         dimensions: got arrays of size $(size(q))."))
    bias === nothing && mask === nothing || throw(ArgumentError(
        "A `bias` or a `mask` is not supported with `q_segments`."))
    length(q_segments) == length(kv_segments) || throw(ArgumentError(
        "Expected as many segments of keys as of queries, got $(length(kv_segments)) and \
         $(length(q_segments))."))
    return nothing
end

@non_differentiable check_segments(::Any...)

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
attention_lengths(x::AA4, lengths::Nothing, n::Int=size(x, 4)) = nothing

function attention_lengths(x::AA4, lengths::Union{Number,AbstractArray}, n::Int=size(x, 4))
    out = similar(x, Int32, (1, 1, 1, n))
    lengths isa Number && return fill!(out, clamp(lengths, 0, size(x, 3)))
    length(lengths) == length(out) || throw(ArgumentError(
        "Expected a sequence length for each of the $(length(out)) sequences, \
         got $(length(lengths))."))
    return copyto!(out, Int32.(clamp.(vec(lengths), 0, size(x, 3))))
end

# The offsets of the queries as the lengths, but an `Integer`, which is kept as it is:
# a backend then recognizes the default of zero without reading the device.
attention_offsets(x::AA4, offset::Integer, n::Int=size(x, 4)) = Int(offset)

function attention_offsets(x::AA4, offsets::Union{Number,AbstractArray}, n::Int=size(x, 4))
    out = similar(x, Int32, (1, 1, 1, n))
    offsets isa Number && return fill!(out, offsets)
    length(offsets) == length(out) || throw(ArgumentError(
        "Expected a query offset for each of the $(length(out)) sequences, \
         got $(length(offsets))."))
    return copyto!(out, Int32.(vec(offsets)))
end

# The valid positions of each segment: all of them, or the `lengths` given.
segment_lengths(x::AA4, lengths::Nothing, sizes) = reshape(sizes, 1, 1, 1, :)
segment_lengths(x::AA4, lengths, sizes) =
    min.(attention_lengths(x, lengths, length(sizes)), reshape(sizes, 1, 1, 1, :))

@non_differentiable attention_lengths(::Any...)
@non_differentiable attention_offsets(::Any...)
@non_differentiable segment_lengths(::Any...)

# Packed sequences, in `(head_dim, nheads, total_len, 1)` arrays: the segment `s` of the
# queries, their positions `q_ends[s - 1] + 1:q_ends[s]`, attends to the segment `s` of
# the keys only. The sizes and the cumulative ends of the segments are `Int32` vectors on
# the device of the inputs. The longest segment is `nothing` where the sizes are not read
# on the host and no `maxlen` is given: the fused kernels, whose launches it sizes, are
# then not taken.
struct AttentionSegments{V<:AbstractVector}
    q_sizes::V
    q_ends::V
    kv_sizes::V
    kv_ends::V
    maxlen_q::Union{Int,Nothing}
    maxlen_kv::Union{Int,Nothing}
end

attention_segments(q::AA4, k::AA4, q_segments::Nothing, kv_segments, maxlen) = nothing

function attention_segments(q::AA4, k::AA4, q_segments, kv_segments, maxlen)
    q_sizes, q_ends, maxlen_q = segment_sizes(q, q_segments, maxlen, "queries")
    kv_sizes, kv_ends, maxlen_kv = segment_sizes(k, kv_segments, maxlen, "keys")
    return AttentionSegments(q_sizes, q_ends, kv_sizes, kv_ends, maxlen_q, maxlen_kv)
end

function segment_sizes(x::AA4, segments::AbstractVector, maxlen, name)
    n = length(segments)
    host = segments_on_host(segments, maxlen)
    if host === nothing
        sizes = copyto!(similar(x, Int32, n), Int32.(segments))
        # `accumulate` keeps `Int32`, which Reactant's scan needs where `cumsum` widens it
        # (EnzymeAD/Reactant.jl#3525)
        return sizes, accumulate(+, sizes), maxlen === nothing ? nothing : Int(maxlen)
    end
    sizes = Int32.(host)
    all(>=(0), sizes) && sum(sizes; init=0) == size(x, 3) || throw(ArgumentError(
        "The lengths of the $n segments of the $name have to be nonnegative and add up to \
         the $(size(x, 3)) positions of the sequence."))
    return copyto!(similar(x, Int32, n), sizes), copyto!(similar(x, Int32, n), accumulate(+, sizes)),
           Int(maximum(sizes; init=0))
end

segment_sizes(x, segments, maxlen, name) =
    throw(ArgumentError("The lengths of the segments have to be a vector, got $(typeof(segments))."))

# The lengths of the segments on the host, or `nothing` where they are not read there:
# on a device when `maxlen` is given, and where they cannot be, as the traced numbers of
# Reactant.
segments_on_host(segments::Array, maxlen) = segments
segments_on_host(segments::AbstractArray, maxlen) =
    maxlen === nothing && eltype(segments) <: Integer ? Array(segments) : nothing

@non_differentiable attention_segments(::Any...)

fused_attention_segments(segments::Nothing) = true
fused_attention_segments(segments::AttentionSegments) =
    segments.maxlen_q !== nothing && segments.maxlen_kv !== nothing

@non_differentiable fused_attention_segments(::Any...)

# Materialized implementation, returning both the attention output and the weights `α`.
function _attention(q::AA4, k::AA4, v::AA4, bias; fdrop, mask, scale, is_causal, window=nothing,
                    q_offset=0, q_lengths=nothing, kv_lengths=nothing, segments=nothing)
    # [q] = [qk_head_dim, nheads, q_len, batch_size]
    # [k] = [qk_head_dim, nkvheads, kv_len, batch_size]
    # [v] = [v_head_dim,  nkvheads, kv_len, batch_size]
    is_causal, q_offset = causal_offsets(q, k, is_causal, q_offset, q_lengths, kv_lengths)
    is_causal, q_offset, window = attention_band(is_causal, q_offset, window)
    if segments !== nothing
        mask, valid = packed_attention_mask(q, k, is_causal, window, q_offset, q_lengths, kv_lengths,
                                            segments)
        x, α = _attention(q, k, v, bias; fdrop, mask, scale, is_causal=false)
        return x .* reshape(valid, 1, 1, :, 1), α .* reshape(valid, 1, :, 1, 1)
    end
    mask = attention_mask(q, k, mask, is_causal, window, q_offset, q_lengths, kv_lengths)
    # grouped-query attention: each group of `r` query heads shares a key/value head
    r = size(q, 2) ÷ size(k, 2)

    α = _attention_weights(q, k, bias, fdrop, mask, scale, q_lengths, r)
    # [α] = [kv_len, q_len, nheads, batch_size]

    # The following permutedims and batched_mul are equivalent to
    # @tullio x[d, h, i, b] := α[j, i, h, b] * v[d, (h - 1) ÷ r + 1, j, b]
    vt = permute_kv_heads(v, (1, 3, 2, 4), r)
    x = batched_mul(vt, α)
    x = permutedims(x, (1, 3, 2, 4))
    # [x] = [v_head_dim, nheads, q_len, batch_size]
    return x, α
end

function _attention_weights(q::AA4{T}, k::AA4, bias, fdrop, mask, scale, q_lengths=nothing, r=1) where T
    s = scale === nothing ? inv(√T(size(q, 1))) : T(scale)
    # The following permutedims and batched_mul are equivalent to
    # @tullio logits[j, i, h, b] := s * q[d, h, i, b] * k[d, (h - 1) ÷ r + 1, j, b]
    kt = permute_kv_heads(k, (3, 1, 2, 4), r)
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

# The mask of the materialized implementation: the `mask` of the user, the causal one, that
# of the window and the one of the padded keys, a key being attended to only where all of
# them allow it. The padded queries, whose weights are zeroed, attend to every key: one
# attending to none, as past a window, would have NaN weights, which the softmax passes on
# to the gradients of every key.
function attention_mask(q::AA4, k::AA4, mask, is_causal::Bool, window, q_offset, q_lengths, kv_lengths)
    if is_causal
        mask = combine_masks(mask, make_causal_mask(q, size(k, 3), size(q, 3), q_offset))
    end
    if window !== nothing
        mask = combine_masks(mask, make_window_mask(q, size(k, 3), size(q, 3), q_offset, window))
    end
    if kv_lengths !== nothing
        mask = combine_masks(mask, padding_mask(size(k, 3), kv_lengths; dims=1))
    end
    if mask !== nothing && q_lengths !== nothing
        mask = mask .| .!padding_mask(size(q, 3), q_lengths; dims=2)
    end
    return mask
end

# `is_causal` as a `Bool` and the offsets it goes with: `:bottom_right` aligns the last
# valid query with the last valid key.
causal_offsets(q, k, is_causal::Bool, q_offset, q_lengths, kv_lengths) = is_causal, q_offset
causal_offsets(q, k, is_causal::Symbol, q_offset, q_lengths, kv_lengths) =
    true, bottom_right_offsets(q, k, q_lengths, kv_lengths)

bottom_right_offsets(q, k, q_lengths::Nothing, kv_lengths::Nothing) = size(k, 3) - size(q, 3)
bottom_right_offsets(q, k, q_lengths, kv_lengths) =
    something(kv_lengths, size(k, 3)) .- something(q_lengths, size(q, 3))

@non_differentiable causal_offsets(::Any...)

# The window as the implementations take it, the band of the `window` keys that end at
# the causal bound: the query `i` attends to the keys `q_offset + i - window < j ≤ q_offset + i`.
# Without `is_causal`, the band of `|q_offset + i - j| < window` has `2 * window - 1` keys
# and ends `window - 1` keys later.
attention_band(is_causal::Bool, q_offset, window::Nothing) = is_causal, q_offset, window
attention_band(is_causal::Bool, q_offset, window::Integer) =
    is_causal ? (true, q_offset, Int(window)) :
                (true, attention_shift(q_offset, window - 1), 2 * Int(window) - 1)

attention_shift(q_offset::Integer, n) = q_offset + Int(n)
attention_shift(q_offset::AbstractArray, n) = q_offset .+ Int32(n)

@non_differentiable attention_band(::Any...)

combine_masks(mask::Nothing, m) = m
combine_masks(mask, m) = mask .& m

# The materialized implementation over the band of a window, the queries in blocks of `C`.
# The queries of a block attend to at most `C + window - 1` keys, from the first of the
# window of its first query to its last query: the blocks with their keys make a batch of
# problems, each with the band as a causal mask, which the materialized implementation
# takes. Memory and compute then grow as `q_len * (C + window)` rather than as
# `q_len * kv_len`.
function banded_attention(q::AA4, k::AA4, v::AA4, bias; fdrop, mask, scale, window::Int, q_offset,
                          q_lengths, kv_lengths)
    C = attention_block_size(size(q, 1) + size(v, 1), window, size(q, 3))
    blocks = attention_blocks(q, k, C, window, q_offset, q_lengths, kv_lengths)
    mask = combine_masks(attention_mask_band(mask, blocks), blocks.kv_valid)
    x, _ = _attention(gather_positions(q, blocks.q), gather_positions(k, blocks.kv),
                      gather_positions(v, blocks.kv), attention_band_of(bias, blocks); fdrop, mask, scale,
                      is_causal=true, window, q_offset=window - 1, q_lengths=blocks.q_lengths)
    # [x] = [v_head_dim, nheads, C, nblocks]: the blocks of a batch element are consecutive
    x = reshape(x, size(x, 1), size(x, 2), :, size(q, 4))
    return x[:, :, 1:size(q, 3), :]
end

# A query goes through `C + window - 1` keys in a block of `C`, which copies the keys and
# the values `1 + (window - 1) / C` times: their sum is least near `C = √(head_dims * window)`.
attention_block_size(head_dims::Int, window::Int, q_len::Int) =
    min(max(isqrt(head_dims * window), 32), q_len)

# The pairs of a query and a key the materialized implementation goes through.
banded_attention_pairs(head_dims::Int, q_len::Int, kv_len::Int, window::Nothing) = q_len * kv_len

function banded_attention_pairs(head_dims::Int, q_len::Int, kv_len::Int, window::Int)
    C = attention_block_size(head_dims, window, q_len)
    return min(cld(q_len, C) * C * (C + window - 1), q_len * kv_len)
end

banded_attention_cheaper(head_dims::Int, q_len::Int, kv_len::Int, window) =
    banded_attention_pairs(head_dims, q_len, kv_len, window) < q_len * kv_len

# The blocks of `C` queries of each batch element, one after the other, with the keys they
# attend to: the block that starts after the query `i0` has the queries `i0 .+ (1:C)` and
# the keys `j0 .+ (1:C + window - 1)`, with `j0 = q_offset + i0 - window + 1`. Returns their
# positions in `q` and `k`, those past the ends of the sequence clamped to them, the mask
# of the keys in the sequence and in its `kv_lengths`, the number of valid queries of each
# block, and for `attention_band_of` the positions and the batch element of each block.
function attention_blocks(q::AA4, k::AA4, C::Int, window::Int, q_offset, q_lengths, kv_lengths)
    q_len, kv_len = size(q, 3), size(k, 3)
    nblocks = cld(q_len, C)
    n = nblocks * size(q, 4)
    blk = copyto!(similar(q, Int32, n), collect(Int32(0):Int32(n - 1)))
    b = blk .÷ Int32(nblocks) .+ Int32(1)
    i0 = Int32(C) .* (blk .% Int32(nblocks))
    j0 = i0 .+ per_segment(q_offset, b) .- Int32(window - 1)
    nvalid = clamp.(per_segment(something(q_lengths, q_len), b) .- i0, 0, C)
    jmax = per_segment(something(kv_lengths, kv_len), b) .- j0
    i = (1:C) .+ reshape(i0, 1, :)
    j = (1:(C + window - 1)) .+ reshape(j0, 1, :)
    kv_valid = (j .>= 1) .& ((1:(C + window - 1)) .<= reshape(jmax, 1, :))
    i, j, b = min.(i, q_len), clamp.(j, 1, kv_len), reshape(b, 1, :)
    return (; q=i .+ q_len .* (b .- 1), kv=j .+ kv_len .* (b .- 1),
            kv_valid=reshape(kv_valid, size(j, 1), 1, 1, :), q_lengths=reshape(nvalid, 1, 1, 1, :),
            i=reshape(i, 1, C, 1, :), j=reshape(j, size(j, 1), 1, 1, :), b=reshape(b, 1, 1, 1, :))
end

# The band of a `bias` or a `mask` broadcastable to `(kv_len, q_len, nheads, batch_size)`,
# for the blocks of `attention_blocks`, as their logits.
attention_band_of(x::Nothing, blocks) = nothing

function attention_band_of(x::AbstractArray, blocks)
    nk, nq, nh, nb = size(x, 1), size(x, 2), size(x, 3), size(x, 4)
    idx = min.(blocks.j, nk) .+
          nk .* (min.(blocks.i, nq) .- 1 .+ nq .* (reshape(0:(nh - 1), 1, 1, :) .+ nh .* (min.(blocks.b, nb) .- 1)))
    return gather(unlazy(flat_logits(x)), idx)
end

# An alias of `x`, as the reshapes of the inputs (see the Enzyme extension).
flat_logits(x::AbstractArray) = vec(x)

attention_mask_band(mask, blocks) = attention_band_of(mask, blocks)

@non_differentiable banded_attention_cheaper(::Any...)
@non_differentiable attention_blocks(::Any...)
@non_differentiable attention_mask_band(::Any...)

# Packed sequences outside of the fused kernels: in a padded batch with a segment in each
# batch element, or over the whole packed sequence with a mask that keeps each query to
# the keys of its segment, whichever attends to fewer pairs, those of the band of a window
# in the padded batch. Long segments among short ones pad to many times the positions they
# have.
function packed_attention(q::AA4, k::AA4, v::AA4; fdrop, scale, is_causal::Bool, window, q_offset,
                          q_lengths, kv_lengths, segments::AttentionSegments)
    padded_segments_cheaper(q, v, segments, window) &&
        return padded_segments_attention(q, k, v; fdrop, scale, is_causal, window, q_offset, q_lengths,
                                         kv_lengths, segments)
    mask, valid = packed_attention_mask(q, k, is_causal, window, q_offset, q_lengths, kv_lengths, segments)
    x = _scaled_dot_product_attention(q, k, v, nothing; fdrop, mask, scale, is_causal=false,
                                      window=nothing, q_offset=0, q_lengths=nothing,
                                      kv_lengths=nothing, segments=nothing)
    return x .* reshape(valid, 1, 1, :, 1)
end

function padded_segments_cheaper(q::AA4, v::AA4, s::AttentionSegments, window)
    (s.maxlen_q === nothing || s.maxlen_kv === nothing) && return false
    pairs = banded_attention_pairs(size(q, 1) + size(v, 1), s.maxlen_q, s.maxlen_kv, window)
    return length(s.q_ends) * pairs <= size(q, 3) * size(v, 3)
end

function padded_segments_attention(q::AA4, k::AA4, v::AA4; fdrop, scale, is_causal, window, q_offset,
                                   q_lengths, kv_lengths, segments::AttentionSegments)
    (; maxlen_q, maxlen_kv) = segments
    qi = padded_segment_indices(segments.q_sizes, segments.q_ends, maxlen_q)
    ki = padded_segment_indices(segments.kv_sizes, segments.kv_ends, maxlen_kv)
    x = _scaled_dot_product_attention(gather_positions(q, qi), gather_positions(k, ki),
                                      gather_positions(v, ki), nothing; fdrop, mask=nothing,
                                      scale, is_causal, window, q_offset, q_lengths, kv_lengths,
                                      segments=nothing)
    # [x] = [v_head_dim, nheads, maxlen_q, nsegments]
    id, pos = segment_positions(segments.q_sizes, segments.q_ends, size(q, 3))
    x = gather_positions(x, pos .+ maxlen_q .* (id .- 1) .+ 1)
    return reshape(x, size(x)..., 1)
end

# The positions `idx` of `x`, its batch elements taken one after the other. Reactant
# leaves the reshape lazy, which its `gather` does not take (EnzymeAD/Reactant.jl#3526):
# it is copied there.
gather_positions(x::AA4, idx) = gather(unlazy(reshape(x, size(x, 1), size(x, 2), :)), idx)

unlazy(x) = x
unlazy(x::Base.ReshapedArray) = copy(x)

# The positions of the packed sequence in a padded batch of `maxlen` positions for each
# segment, as a `(maxlen, nsegments)` array of indices: the padding repeats a position of
# the segment, or the first one of the sequence for an empty segment.
padded_segment_indices(sizes, ends, maxlen::Int) =
    max.(reshape(ends .- sizes, 1, :) .+ min.(0:(maxlen - 1), reshape(sizes, 1, :) .- 1) .+ 1, 1)

# The segment of each of the `len` positions of a packed sequence, and its position in
# the segment, counted from zero.
function segment_positions(sizes, ends, len::Int)
    id = vec(sum((0:(len - 1)) .>= reshape(ends, 1, :); dims=2)) .+ 1
    return id, (0:(len - 1)) .- (ends .- sizes)[id]
end

# The `(kv_len, q_len, 1, 1)` mask that keeps each query to the valid keys of its
# segment, and to its causal bound and window, and which of the queries are valid. The
# others attend to every key, as in `attention_mask`.
function packed_attention_mask(q::AA4, k::AA4, is_causal::Bool, window, q_offset, q_lengths, kv_lengths,
                               s::AttentionSegments)
    qid, qpos = segment_positions(s.q_sizes, s.q_ends, size(q, 3))
    kid, kpos = segment_positions(s.kv_sizes, s.kv_ends, size(k, 3))
    mask = (kid .== reshape(qid, 1, :)) .& (kpos .< vec(kv_lengths)[kid])
    if is_causal
        mask = mask .& (kpos .<= reshape(qpos .+ per_segment(q_offset, qid), 1, :))
    end
    if window !== nothing
        mask = mask .& (kpos .> reshape(qpos .+ per_segment(q_offset, qid) .- window, 1, :))
    end
    valid = qpos .< vec(q_lengths)[qid]
    return reshape(mask .| .!reshape(valid, 1, :), size(mask)..., 1, 1), valid
end

per_segment(x::Integer, id) = x
per_segment(x::AbstractArray, id) = vec(x)[id]

@non_differentiable padded_segments_cheaper(::Any...)
@non_differentiable padded_segment_indices(::Any...)
@non_differentiable segment_positions(::Any...)
@non_differentiable packed_attention_mask(::Any...)

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

# Rectangular causal mask of size `(kvlen, qlen)` with `m[j, i] == j ≤ offset + i`,
# used internally by `scaled_dot_product_attention(...; is_causal=true)`. An array of
# offsets, of size `(1, 1, 1, batch_size)`, gives a mask for each batch element.
make_causal_mask(x::AbstractArray, kvlen::Int, qlen::Int, offset::Integer=0) =
    triu(trues_like(x, (kvlen, qlen)), clamp(-offset, 1 - kvlen, qlen + 1))
make_causal_mask(x::AbstractArray, kvlen::Int, qlen::Int, offset::AbstractArray) =
    (1:kvlen) .<= (1:qlen)' .+ offset

# The mask of a band of `window` keys ending at the causal bound, `m[j, i] == j > offset + i - window`.
make_window_mask(x::AbstractArray, kvlen::Int, qlen::Int, offset::Integer, window::Int) =
    tril(trues_like(x, (kvlen, qlen)), clamp(window - offset - 1, -kvlen - 1, qlen - 1))
make_window_mask(x::AbstractArray, kvlen::Int, qlen::Int, offset::AbstractArray, window::Int) =
    (1:kvlen) .> (1:qlen)' .+ (offset .- window)

trues_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), true)
falses_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), false)

split_heads(x, nheads) = reshape(x, size(x, 1) ÷ nheads, nheads, size(x)[2:end]...)
join_heads(x) = reshape(x, :, size(x)[3:end]...)

# `permutedims(x, perm)` of a key or value array, for a `perm` that takes its heads to the
# third dimension, with each head repeated `r` times for grouped-query attention, so that
# it lines up with the query heads. The heads are repeated in the permuted copy rather
# than in `x`: a branch between `x` itself and a new array cannot be differentiated by
# Enzyme where `x` is constant.
function permute_kv_heads(x::AA4, perm, r::Int)
    xt = permutedims(x, perm)
    return r == 1 ? xt : repeat_kv_heads(xt, r)
end

# [x] = [_, _, nkvheads, batch] -> [_, _, nkvheads * r, batch]
repeat_kv_heads(x::AA4, r::Int) = repeat(x; inner=(1, 1, r, 1))

function rrule(::typeof(repeat_kv_heads), x::AA4, r::Int)
    y = repeat_kv_heads(x, r)
    function repeat_kv_heads_pullback(Δ)
        Δ = unthunk(Δ)
        Δ = reshape(Δ, size(x, 1), size(x, 2), r, size(x, 3), size(x, 4))
        dx = reshape(sum(Δ; dims=3), size(x))
        return NoTangent(), dx, NoTangent()
    end
    return y, repeat_kv_heads_pullback
end

@non_differentiable make_causal_mask(::Any...)
@non_differentiable make_window_mask(::Any...)
@non_differentiable trues_like(::Any...)
@non_differentiable falses_like(::Any...)
