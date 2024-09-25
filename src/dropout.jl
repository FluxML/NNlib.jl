
"""
    dropout([rng], A, p; [dims])

Returns an array in which each element of `A` is either replaced with zero,
with probability `p`, or else multiplied by `1/(1-p)`.

By default every element is treated independently.
With keyword `dims=1`, a choice is made for every value of the 1st index
i.e. each row of a matrix is either zero or not.

Optional first argument is the random number generator used.

# Examples
```julia-repl
julia> dropout(ones(2, 10), 0.2)
2×10 Matrix{Float64}:
 1.25  1.25  0.0   1.25  1.25  1.25  1.25  1.25  1.25  1.25
 1.25  1.25  1.25  0.0   1.25  1.25  0.0   1.25  1.25  1.25

julia> mean(dropout(ones(10^4, 5), 0.2), dims=1)
1×5 Matrix{Float64}:
 0.998  1.00075  0.99125  0.99575  1.00075

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
    _rng_compat_array(rng, A)
    T = float(eltype(A))
    0 <= p <= 1 || throw(ArgumentError("dropout expects a probability 0 <= p <= 1"))
    if p > 0
        dst = similar(A, T, size(A))
        pT = convert(real(T), p)
        _dropout!(rng, dst, A, pT, dims)
    else
        # Not so sure we want fast paths... this tries but doesn't guarantee type-stability,
        # and the rrule does not have such a fast paths.
        convert(AbstractArray{T}, A)
    end
end

"""
    dropout!(B, A, p; [dims])

This does exactly `B .= dropout(A, p; dims)`,
or rather, it's the implementation of out-of-place [`dropout`](@ref).
"""
dropout!(B::AbstractArray, A::AbstractArray, p::Real; dims = :) = dropout!(_rng_from_array(B), B, A, p; dims)

function dropout!(rng::AbstractRNG, dst::AbstractArray, src::AbstractArray, p::Real; dims=:)
    size(dst) == size(src) || throw(DimensionMismatch("dropout! expects output array the same size as input"))
    0 <= p <= 1 || throw(ArgumentError("dropout expects a probability 0 <= p <= 1"))
    _rng_compat_array(rng, src)
    if p > 0
        pT = convert(real(eltype(dst)), p)
        _dropout!(rng, dst, src, pT, dims)
    else
        # This fast path isn't free, but no concerns about types changing:
        copyto!(dst, src)
    end
end

# This is the easy case in that we can safely use the output array for random numbers.
function _dropout!(rng::AbstractRNG, dst::AbstractArray, src::AbstractArray, p::Real, dims::Colon)
    T = real(eltype(dst))
    val = convert(T, 1/(1-p))
    rand!(rng, dst)
    ## This is what we want, but it hits a SIMD bug, solved by _fast_broadcast!
    # dst .= (dst.>p) .* val .* src
    _fast_broadcast!(dst, src) do q, x
        ((real(q)>p) * val) * x
    end
    dst
end

# For other dims, we we do need to allocate something.
function _dropout!(rng::AbstractRNG, dst::AbstractArray, src::AbstractArray, p::Real, dims)
    T = real(eltype(dst))
    tmp = similar(dst, T, ntuple(d -> d in dims ? size(src,d) : 1, ndims(src)))
    rand!(rng, tmp)
    val = convert(T, 1/(1-p))
    ## One-pass strategy -- faster on GPU
    dst .= ((tmp.>p) .* val) .* src
    ## Two-pass strategy -- slightly faster on some CPUs?
    # _fast_broadcast!(tmp) do q
    #     (q>p) * val
    # end
    # dst .= tmp .* src
end

# The gradient needs to keep the random choices made, thus store at least a BitArray,
# but the following way turns out to be faster & simpler:
function ChainRulesCore.rrule(::typeof(dropout), rng::AbstractRNG, A::AbstractArray, p::Real; dims = :)
    T = float(real(eltype(A)))
    val = convert(T, 1/(1-p))
    keep = if dims isa Colon
        similar(A, T, size(A))
    else
        similar(A, T, ntuple(d -> d in dims ? size(A,d) : 1, ndims(A)))
    end
    rand!(rng, keep)
    Y = @. ((keep>p) * val) * A
    function dropout_back(Δ)
        dY = unthunk(Δ)
        dA = @. ((keep>p) * val) * dY
        (NoTangent(), NoTangent(), dA, NoTangent())
    end
    return Y, dropout_back
end
# Possibly TODO: another approach to the gradient would be to copy the RNG
# and then re-generate the same mask, instead of storing it. This saves memory
# and seems about as fast, but needs a method `copy(::CUDA.RNG)` and careful checking.
# https://github.com/FluxML/NNlib.jl/pull/454#issuecomment-1369357402


"""
    _rng_from_array(x)

Return the random number generator most appropriate for `x`:
`CUDA.default_rng()` for `CuArray`, else `Random.default_rng()`
"""
_rng_from_array(::AbstractArray) = Random.default_rng()

@non_differentiable _rng_from_array(::Any)

# This exists because `rand!(default_rng(), CUDA.rand(3))` ignores the RNG,
# and Flux would prefer an error. NNlibCUDAExt will overload it to produce that.
_rng_compat_array(::AbstractRNG, ::AbstractArray) = nothing
