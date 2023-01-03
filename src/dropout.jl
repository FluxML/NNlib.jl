using Random, ChainRulesCore

"""
    dropout([rng], A, p; dims=:)

Returns an array in which each element of `A` is either replaced with zero,
with probability `p`, or else multiplied by `1/(1-p)`.

By default every element is treated independently.
With `dims=1`, a choice is made for every value of the 1st index
i.e. each row of a matrix is either zero or not.

Optional first argument is the random number generator used.

# Examples
```
julia> dropout(ones(2, 10), 1/5)
2×10 Matrix{Float64}:
 1.25  1.25  0.0   1.25  1.25  1.25  1.25  1.25  1.25  1.25
 1.25  1.25  1.25  0.0   1.25  1.25  0.0   1.25  1.25  1.25

julia> mean(dropout(ones(10^4, 5), 0.3), dims=1)
1×5 Matrix{Float64}:
 0.996  1.00171  1.00629  0.998714  0.992429

julia> dropout(ones(5, 5), 0.7, dims=1)  # whole row the same
5×5 Matrix{Float64}:
 3.33333  3.33333  3.33333  3.33333  3.33333
 0.0      0.0      0.0      0.0      0.0
 0.0      0.0      0.0      0.0      0.0
 3.33333  3.33333  3.33333  3.33333  3.33333
 0.0      0.0      0.0      0.0      0.0

julia> mean(dropout(ones(10^4, 5), 0.3, dims=1), dims=1)
1×5 Matrix{Float64}:
 1.00571  1.00571  1.00571  1.00571  1.00571
```
"""
dropout(A::AbstractArray, p::Real; dims = :) = dropout(_rng_from_array(A), A, p; dims)

function dropout(rng::AbstractRNG, A::AbstractArray, p::Real; dims = :)
    T = float(eltype(A))
    0 <= p <= 1 || throw(ArgumentError("dropout expects a probability 0 <= p <= 1"))
    if p > 0
        dst = similar(A, T)
        pT = convert(real(T), p)
        _dropout!(rng, dst, A, pT, dims)
    else
        # Not so sure we want fast paths... this tries but doesn't guarantee type-stability,
        # and the rrule does not have such a fast paths.
        convert(AbstractArray{T}, A)
    end
end

"""
    dropout!(B, A, p; dims=:)

This does exactly `B .= dropout(A, p; dims)`,
or rather, it's the implementation of out-of-place [`dropout`](@ref).
"""
function dropout!(dst::AbstractArray, src::AbstractArray, p::Real; dims=:)
    size(dst) == size(src) || throw(DimensionMismatch("dropout! expects output array the same size as input"))
    0 <= p <= 1 || throw(ArgumentError("dropout expects a probability 0 <= p <= 1"))
    if p > 0
        rng = _rng_from_array(A)
        pT = convert(real(eltype(dst)), p)
        _dropout!(rng, dst, src, pT, dims)
    else
        copyto!(dst, src)
    end
end

# This is the easy case in that we can safely use the output array for random numbers.
function _dropout!(rng::AbstractRNG, dst::AbstractArray, src::AbstractArray, p::Real, dims::Colon)
    val = convert(eltype(dst), 1/(1-p))
    rand!(rng, dst)
    # dst .= (dst.>p) .* val .* src  # hits a SIMD bug
    _fast_broadcast!(dst, src) do q, x
        (q>p) * val * x
    end
    dst
end

# For other dims, we we do need to allocate something.
function _dropout!(rng::AbstractRNG, dst::AbstractArray, src::AbstractArray, p::Real, dims)
    tmp = similar(dst, ntuple(d -> d in dims ? size(src,d) : 1, ndims(src)))
    rand!(rng, tmp)
    val = convert(eltype(dst), 1/(1-p))
    # One-pass strategy:
    # dst .= (tmp.>p) .* val .* src
    # Two-pass strategy:
    _fast_broadcast!(tmp) do q
        (q>p) * val
    end
    dst .= tmp .* src
end

# The gradient needs to keep the random choices made, thus store at least a BitArray,
# but the following way turns out to be faster & simpler:
function ChainRulesCore.rrule(::typeof(dropout), rng::AbstractRNG, A::AbstractArray, p::Real; dims = :)
    T = float(eltype(A))
    val = convert(T, 1/(1-p))
    keep = if dims isa Colon
        similar(A, T) 
    else
        similar(A, T, ntuple(d -> d in dims ? size(A,d) : 1, ndims(A)))
    end
    rand!(rng, keep)
    Y = @. (keep>p) * A * val
    function dropout_back(Δ)
        dY = unthunk(Δ)
        dA = @. (keep>p) * dY * val
        (NoTangent(), NoTangent(), dA, NoTangent())
    end
    return Y, dropout_back
end

"""
    _fast_broadcast!(f, x, y, z...)

This does `x .= f.(x, y, z...)`, but works around
an issue with broadcasting that prevents SIMD in such cases.
Can be removed once https://github.com/JuliaLang/julia/issues/43153 is fixed.

Not intended for general use. Does not check sizes!
"""
function _fast_broadcast!(f::F, x::Array, yz...) where {F<:Function}
    bc = Broadcast.instantiate(Broadcast.broadcasted(f, x, yz...))
    @simd ivdep for I in eachindex(bc)
        @inbounds x[I] = bc[I]
    end
    return x
end
function _fast_broadcast!(f::F, x::AbstractArray, yz...) where {F<:Function}
    # CUDA does not suffer from this bug
    broadcast!(f, x, x, yz...)
end


"""
    _rng_from_array(x)

Return the random number generator most appropriate for `x`:
`CUDA.default_rng()` for `CuArray`,
else `Random.default_rng()`
"""
_rng_from_array(::AbstractArray) = Random.default_rng()
# _rng_from_array(::CuArray) = CUDA.default_rng()

@non_differentiable _rng_from_array(::Any)


