"""
    rotary_embedding_table(head_dim, maxlen, [T=Float32]; base=10_000)

Return the tables `(cos, sin)` of the rotary position embedding (RoPE) of
[Su et al.](https://arxiv.org/abs/2104.09864) for the positions `1:maxlen`, to be passed
to [`apply_rotary_embedding`](@ref).

Each is a `Matrix{T}` of size `(head_dim ÷ 2, maxlen)`: the cosine and the sine of the
angle `θ[i] * (p - 1)` by which the `i`-th pair of features is rotated at the position
`p`, with the frequencies `θ[i] = base^(-2(i - 1) / head_dim)`. The first position is not
rotated, as the position zero of PyTorch and JAX.

The tables are computed in `Float64` on the CPU: move them to the device of the inputs.
Other frequencies, as those rescaled by Llama 3 or YaRN, give their own tables
`cos.(θ .* (0:maxlen - 1)')` and `sin.(θ .* (0:maxlen - 1)')`.

# Examples

```jldoctest
julia> cos_table, sin_table = rotary_embedding_table(64, 4096);

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
    apply_rotary_embedding(x, cos, sin; offset=0, positions=nothing, interleaved=false)

Apply the rotary position embedding (RoPE) of [Su et al.](https://arxiv.org/abs/2104.09864)
to `x`, a query or a key of [`scaled_dot_product_attention`](@ref) of size
`(head_dim, nheads, seq_len, batch_size...)`: the features of each head are rotated pair
by pair, by angles proportional to the position of the token. The dot product of a query
and a key so rotated depends on their positions through their difference only. The
values are not rotated.

The tables `cos` and `sin` have size `(head_dim ÷ 2, maxlen)`, their column `p` holding
the cosines and the sines of the angles of the position `p`, as those of
[`rotary_embedding_table`](@ref). At the position `p`, the `k`-th pair of features
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

The gradient with respect to `x` is the rotation by the opposite angles. The rule of
ChainRules, which Zygote uses, takes the tables as constants.

# Examples

The causal self-attention of a decoder, in which the queries and the keys are rotated
but not the values, here with 8 heads of dimension 64 sharing 2 key/value heads:

```julia
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
    offset, positions = rotary_positions(x, cos, offset, positions)
    return rotate_pairs(x, cos, sin, offset, positions, interleaved)
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

# The positions of the tokens as an offset, or as an array of size `(seq_len, batch_size)`
# or `(seq_len, 1)` for all the batch elements.
function rotary_positions(x::AbstractArray, cos::AbstractMatrix, offset::Integer, positions::Nothing)
    checkbounds(cos, :, offset .+ (1:size(x, 3)))
    return Int(offset), nothing
end

rotary_positions(x::AbstractArray, cos::AbstractMatrix, offset::Number, positions::Nothing) =
    0, (1:size(x, 3)) .+ offset
rotary_positions(x::AbstractArray, cos::AbstractMatrix, offset, positions::Nothing) =
    0, (1:size(x, 3)) .+ reshape(offset, 1, :)
rotary_positions(x::AbstractArray, cos::AbstractMatrix, offset, positions) =
    0, reshape(positions, size(positions, 1), :)

@non_differentiable rotary_positions(::Any...)

# `x` rotated pair by pair by the angles of the tables of cosines `c` and of sines `s`, or
# by the opposite angles, the token `l` of the batch element `b` taking the column
# `offset + l` or `positions[l, b]`. A kernel computes it on the arrays of the CPU and of
# the GPU, and a broadcast, which automatic differentiation can go through, under it and
# on other arrays. The rule of ChainRules calls the kernel.
function rotate_pairs(x::AbstractArray, c::AbstractMatrix, s::AbstractMatrix, offset::Int,
                      positions, interleaved::Bool, inverse::Bool=false)
    (within_gradient(x) || !(x isa AnyGPUArray || on_cpu(x))) &&
        return rotate_pairs_broadcast(x, c, s, offset, positions, interleaved, inverse)
    y = similar(x, promote_type(eltype(x), eltype(c), eltype(s)))
    positions === nothing || checkbounds_idx(positions, (size(c, 2),))
    return rotate_pairs!(y, x, c, s, offset, positions, interleaved, inverse)
end

# Whether the memory of `x` is that of an `Array`, which the loops of the CPU can index:
# the traced arrays of Reactant are `DenseArray`s too.
on_cpu(x::Array) = true
on_cpu(x::AbstractArray) = parent(x) !== x && on_cpu(parent(x))

function rotate_pairs!(y::AbstractArray, x::AbstractArray, c, s, offset::Int, positions,
                       interleaved::Bool, inverse::Bool)
    x4, y4 = rotary_kernel_args(x), rotary_kernel_args(y)
    ntokens = size(x4, 3) * size(x4, 4)
    ntasks = should_use_spawn() ? min(Threads.nthreads(:default), cld(length(x), 2^16)) : 1
    if ntasks == 1
        rotate_tokens!(y4, x4, c, s, 1:ntokens, offset, positions, interleaved, inverse)
    else
        @sync for tokens in Iterators.partition(1:ntokens, cld(ntokens, ntasks))
            Threads.@spawn rotate_tokens!(y4, x4, c, s, tokens, offset, positions, interleaved, inverse)
        end
    end
    return y
end

function rotate_tokens!(y, x, c, s, tokens, offset, positions, interleaved, inverse)
    @inbounds for t in tokens
        l, b = mod1(t, size(x, 3)), cld(t, size(x, 3))
        p = rotary_column(offset, positions, l, b)
        for h in axes(x, 2)
            @simd for k in axes(c, 1)
                rotate_pair!(y, x, c, s, k, h, l, b, p, interleaved, inverse)
            end
        end
    end
end

function rotate_pairs!(y::AnyGPUArray, x::AnyGPUArray, c, s, offset::Int, positions,
                       interleaved::Bool, inverse::Bool)
    x4, y4 = rotary_kernel_args(x), rotary_kernel_args(y)
    positions = positions === nothing ? nothing : _to_backend(y, positions)
    rotate_pairs_kernel!(get_backend(y))(y4, x4, c, s, offset, positions, interleaved, inverse;
                                         ndrange=(size(c, 1), size(x4)[2:4]...))
    return y
end

@kernel inbounds=true cpu=false function rotate_pairs_kernel!(y, @Const(x), @Const(c), @Const(s),
                                                              offset, positions, interleaved, inverse)
    k, h, l, b = @index(Global, NTuple)
    p = rotary_column(offset, positions, l, b)
    rotate_pair!(y, x, c, s, k, h, l, b, p, interleaved, inverse)
end

# `x` as a `(head_dim, nheads, seq_len, batch_size)` array.
rotary_kernel_args(x) = reshape(x, size(x, 1), size(x, 2), size(x, 3), :)

# The column of the tables for the token `l` of the batch element `b`.
rotary_column(offset, positions::Nothing, l, b) = offset + l
Base.@propagate_inbounds rotary_column(offset, positions, l, b) =
    Int(positions[l, ifelse(size(positions, 2) == 1, 1, b)])

# The pair `k` of the token `l` of the batch element `b`, in the head `h`, rotated by the
# angle of the column `p`. Inbounds where called so, for the loops of the CPU to vectorize.
Base.@propagate_inbounds function rotate_pair!(y, x, c, s, k, h, l, b, p, interleaved, inverse)
    i, j = interleaved ? (2k - 1, 2k) : (k, k + size(c, 1))
    x1, x2 = x[i, h, l, b], x[j, h, l, b]
    cs, sn = c[k, p], ifelse(inverse, -s[k, p], s[k, p])
    y[i, h, l, b] = x1 * cs - x2 * sn
    y[j, h, l, b] = x2 * cs + x1 * sn
    return nothing
end

# The columns of the tables are taken out of them, a copy: Enzyme cannot tell a constant
# table apart from active memory where a view of it is broadcast with an active `x`,
# without runtime activity.
function rotate_pairs_broadcast(x, c, s, offset, positions, interleaved, inverse)
    half, len = size(x, 1) ÷ 2, size(x, 3)
    c, s = rotary_columns(c, offset, positions, len), rotary_columns(s, offset, positions, len)
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

rotary_columns(c, offset, positions::Nothing, len) = c[:, offset .+ (1:len)]
rotary_columns(c, offset, positions, len) = gather(c, positions)

# `x` with the two features of each pair along a dimension of their own. Reshapes, which
# have an Enzyme rule for a constant `x` next to active tables.
interleaved_pairs(x) = reshape(x, 2, size(x, 1) ÷ 2, size(x, 2), size(x, 3), :)
halved_pairs(x) = reshape(x, size(x, 1) ÷ 2, 2, size(x, 2), size(x, 3), :)

# The element `k` of the pair `(a, b)` rotated.
@inline function rotate_pair(a, b, c, s, k, inverse)
    s = ifelse(inverse, -s, s)
    return ifelse(k == 1, a * c - b * s, b * c + a * s)
end

function rrule(::typeof(rotate_pairs), x::AbstractArray, c::AbstractMatrix, s::AbstractMatrix,
               offset::Int, positions, interleaved::Bool)
    function rotate_pairs_pullback(Δ)
        dx = rotate_pairs(unthunk(Δ), c, s, offset, positions, interleaved, true)
        return NoTangent(), dx, NO_ROTARY_TABLE_GRAD, NO_ROTARY_TABLE_GRAD, NoTangent(),
               NoTangent(), NoTangent()
    end
    return rotate_pairs(x, c, s, offset, positions, interleaved), rotate_pairs_pullback
end

const NO_ROTARY_TABLE_GRAD = @not_implemented(
    "the gradient of `apply_rotary_embedding` with respect to the tables")
