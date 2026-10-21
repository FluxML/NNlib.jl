"""
    NNlib.rotary_embedding_table(head_dim, maxlen, [T=Float32]; base=10_000)

Return the tables `(cos, sin)` of the rotary position embedding (RoPE) of
[Su et al.](https://arxiv.org/abs/2104.09864) for the positions `1:maxlen`, to be passed
to [`NNlib.apply_rotary_embedding`](@ref).

Each is a `Matrix{T}` of size `(head_dim ÷ 2, maxlen)`: the cosine and the sine of the
angle `θ[i] * (p - 1)` by which the `i`-th pair of features is rotated at the position
`p`, with the frequencies `θ[i] = base^(-2(i - 1) / head_dim)`. The first position is not
rotated, as the position zero of PyTorch and JAX.

The tables are computed in `Float64` on the CPU: move them to the device of the inputs.
Other frequencies, as those rescaled by Llama 3 or YaRN, give their own tables
`cos.(θ .* (0:maxlen - 1)')` and `sin.(θ .* (0:maxlen - 1)')`.

# Examples

```jldoctest
julia> cos_table, sin_table = NNlib.rotary_embedding_table(64, 4096);

julia> size(cos_table), size(sin_table)
((32, 4096), (32, 4096))
```
"""
function rotary_embedding_table(head_dim::Integer, maxlen::Integer, ::Type{T}=Float32;
                                base::Real=10_000) where T
    iseven(head_dim) || throw(ArgumentError(
        "The head dimension has to be even, to be split in pairs: got $head_dim."))
    θ = Float64(base) .^ (-(0:2:(head_dim - 2)) ./ head_dim)
    angles = θ .* (0:(maxlen - 1))'
    return T.(cos.(angles)), T.(sin.(angles))
end

"""
    NNlib.apply_rotary_embedding(x, cos, sin; offset=0, positions=nothing, interleaved=false)

Apply the rotary position embedding (RoPE) of [Su et al.](https://arxiv.org/abs/2104.09864)
to `x`, a query or a key of [`scaled_dot_product_attention`](@ref) of size
`(head_dim, nheads, seq_len, batch_size...)`: the features of each head are rotated pair
by pair, by angles proportional to the position of the token. The dot product of a query
and a key so rotated depends on their positions through their difference only. The
values are not rotated.

The tables `cos` and `sin` have size `(head_dim ÷ 2, maxlen)`, their column `p` holding
the cosines and the sines of the angles of the position `p`, as those of
[`NNlib.rotary_embedding_table`](@ref). At the position `p`, the `k`-th pair of features
`(a, b)` becomes `(a * cos[k, p] - b * sin[k, p], b * cos[k, p] + a * sin[k, p])`. The
pairs are `(x[k], x[k + head_dim ÷ 2])`, from the two halves of the features (as in
Llama, Hugging Face Transformers and Flax), or with `interleaved=true` the consecutive
features `(x[2k - 1], x[2k])` (as in RoFormer, GPT-J and torchtune).

Returns an array of the size of `x`.

# Keywords

- `offset`: The token `i` is at the position `offset + i`. An integer, or an integer array
          with one entry for each batch element. With a KV cache, the number of tokens
          already in the cache, as the `q_offset` of [`scaled_dot_product_attention`](@ref).
          Default `0`.
- `positions`: Either `nothing` or the positions of the tokens, in place of `offset`, as an
          integer array of size `(seq_len, batch_size...)`, or `(seq_len,)` for all the
          batch elements. They are the columns of the tables. Default `nothing`.
- `interleaved`: If `true`, the pairs are the consecutive features, else the two halves.
          Default `false`.

The positions have to be within the columns of the tables. To rotate part of the
features only (as GPT-NeoX and Phi), rotate the slice `x[1:d, :, :, :]` with tables of
`d ÷ 2` rows and concatenate the rest.

# Examples

The causal self-attention of a decoder, in which the queries and the keys are rotated
but not the values, here with 8 heads of dimension 64 sharing 2 key/value heads:

```julia
using NNlib: apply_rotary_embedding, rotary_embedding_table

cos_table, sin_table = rotary_embedding_table(64, 4096)
q = rand(Float32, 64, 8, 100, 4)   # (head_dim, nheads, seq_len, batch_size)
k = rand(Float32, 64, 2, 100, 4)
v = rand(Float32, 64, 2, 100, 4)
q = apply_rotary_embedding(q, cos_table, sin_table)
k = apply_rotary_embedding(k, cos_table, sin_table)
y = scaled_dot_product_attention(q, k, v; is_causal=true)
```

Decoding with a KV cache, which holds the keys already rotated: the new tokens are
rotated with the number of cached tokens as offset, the `q_offset` of the attention.

```julia
kcache = apply_rotary_embedding(rand(Float32, 64, 2, 10, 4), cos_table, sin_table)
vcache = rand(Float32, 64, 2, 10, 4)
# 3 new tokens at the positions 11 to 13
q = apply_rotary_embedding(rand(Float32, 64, 8, 3, 4), cos_table, sin_table; offset=10)
knew = apply_rotary_embedding(rand(Float32, 64, 2, 3, 4), cos_table, sin_table; offset=10)
k = cat(kcache, knew; dims=3)
v = cat(vcache, rand(Float32, 64, 2, 3, 4); dims=3)
y = scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=10)
```

Documents packed in one sequence (see [`scaled_dot_product_attention`](@ref)), whose
positions start over at each document:

```julia
segments = [5, 12, 3]
positions = reduce(vcat, [1:n for n in segments])   # [1:5; 1:12; 1:3]
q = apply_rotary_embedding(rand(Float32, 64, 8, 20), cos_table, sin_table; positions)
k = apply_rotary_embedding(rand(Float32, 64, 2, 20), cos_table, sin_table; positions)
v = rand(Float32, 64, 2, 20)
y = scaled_dot_product_attention(q, k, v; is_causal=true, q_segments=segments)
```
"""
function apply_rotary_embedding(x::AbstractArray, cos::AbstractMatrix, sin::AbstractMatrix;
                                offset=0, positions=nothing, interleaved::Bool=false)
    check_rotary_inputs(x, cos, sin, offset, positions)
    positions = rotary_positions(x, offset, positions)
    return rotate_pairs(x, gather(cos, positions), gather(sin, positions), interleaved)
end

function check_rotary_inputs(x::AbstractArray, cos::AbstractMatrix, sin::AbstractMatrix,
                             offset, positions)
    ndims(x) >= 3 || throw(ArgumentError(
        "The input must have at least 3 dimensions (head_dim, nheads, seq_len): got $(ndims(x))."))
    size(cos) == size(sin) || throw(DimensionMismatch(
        "The tables of the cosines and of the sines have sizes $(size(cos)) and $(size(sin))."))
    size(x, 1) == 2 * size(cos, 1) || throw(DimensionMismatch(
        "The head dimension $(size(x, 1)) has to be twice the $(size(cos, 1)) rows of the tables."))
    nbatch = prod(size(x)[4:end])
    if positions === nothing
        offset isa Number || length(offset) == nbatch || throw(DimensionMismatch(
            "Expected an offset for each of the $nbatch batch elements, got $(length(offset))."))
    else
        offset isa Integer && iszero(offset) || throw(ArgumentError(
            "Give either the `positions` of the tokens or their `offset`, not both."))
        size(positions, 1) == size(x, 3) && length(positions) in (size(x, 3), size(x, 3) * nbatch) ||
            throw(DimensionMismatch("Expected positions of size $((size(x, 3), size(x)[4:end]...)) \
                or $((size(x, 3),)), got $(size(positions))."))
    end
    return nothing
end

@non_differentiable check_rotary_inputs(::Any...)

# The positions of the tokens, the columns of the tables, as an array of size
# `(seq_len, batch_size...)` or `(seq_len,)` for all the batch elements.
rotary_positions(x::AbstractArray, offset::Number, positions::Nothing) = (1:size(x, 3)) .+ offset
rotary_positions(x::AbstractArray, offset, positions::Nothing) =
    (1:size(x, 3)) .+ reshape(offset, 1, :)
rotary_positions(x::AbstractArray, offset, positions) = positions

@non_differentiable rotary_positions(::Any...)

# `x` rotated pair by pair by the angles of the cosines `c` and the sines `s`, of size
# `(head_dim ÷ 2, seq_len, ...)`, or by the opposite angles: one broadcast whose output
# has the two features of each pair along a dimension of their own.
function rotate_pairs(x::AbstractArray, c::AbstractArray, s::AbstractArray, interleaved::Bool,
                      inverse::Bool=false)
    half, len = size(x, 1) ÷ 2, size(x, 3)
    if interleaved
        xp = interleaved_pairs(x)
        y = rotate_pair.(view(xp, 1:1, :, :, :, :), view(xp, 2:2, :, :, :, :),
                         reshape(c, 1, half, 1, len, :), reshape(s, 1, half, 1, len, :),
                         1:2, inverse)
    else
        xp = halved_pairs(x)
        y = rotate_pair.(view(xp, :, 1:1, :, :, :), view(xp, :, 2:2, :, :, :),
                         reshape(c, half, 1, 1, len, :), reshape(s, half, 1, 1, len, :),
                         reshape(1:2, 1, 2), inverse)
    end
    return reshape(y, size(x))
end

# `x` with the two features of each pair along a dimension of their own. Reshapes, which
# have an Enzyme rule for a constant `x` next to active tables.
interleaved_pairs(x) = reshape(x, 2, size(x, 1) ÷ 2, size(x, 2), size(x, 3), :)
halved_pairs(x) = reshape(x, size(x, 1) ÷ 2, 2, size(x, 2), size(x, 3), :)

# The element `k` of the pair `(a, b)` rotated.
@inline function rotate_pair(a, b, c, s, k, inverse)
    s = ifelse(inverse, -s, s)
    return ifelse(k == 1, a * c - b * s, b * c + a * s)
end

function rrule(::typeof(rotate_pairs), x::AbstractArray, c::AbstractArray, s::AbstractArray,
               interleaved::Bool)
    function rotate_pairs_pullback(Δ)
        dx = rotate_pairs(unthunk(Δ), c, s, interleaved, true)
        return NoTangent(), dx, NO_ROTARY_TABLE_GRAD, NO_ROTARY_TABLE_GRAD, NoTangent()
    end
    return rotate_pairs(x, c, s, interleaved), rotate_pairs_pullback
end

const NO_ROTARY_TABLE_GRAD = @not_implemented(
    "the gradient of `apply_rotary_embedding` with respect to the tables")
