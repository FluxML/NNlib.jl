"""
    grouped_mul(W::AbstractArray{<:Any,3}, X::AbstractMatrix, offsets) -> Y
    grouped_mul(A::AbstractMatrix, B::AbstractMatrix, offsets) -> C

Matrix multiplication over groups of different sizes. `offsets` is an integer vector
of nondecreasing cumulative group ends, starting from an implicit zero. Repeated
offsets represent empty groups.

For `W` of size `(n, k, G)` and `X` of size `(k, total)`, return `Y` of size
`(n, total)`, with `Y[:, r] = W[:, :, g] * X[:, r]` for each group `g`, where
`r = (g == 1 ? 0 : offsets[g-1])+1:offsets[g]`. `length(offsets)` must equal `G`.
`W` may be a [`batched_transpose`](@ref) or [`batched_adjoint`](@ref).

For `A` of size `(n, total)` and `B` of size `(total, k)`, return `C` of size
`(n, k, length(offsets))`, with `C[:, :, g] = A[:, r] * B[r, :]`.

Entries past the last offset are ignored. The corresponding columns of `Y`, and
the matrices of `C` for empty groups, are zero. Both forms support differentiation.

On CUDA 6.5+, supported strided arrays use a single cuBLAS grouped GEMM, with
`Float32` accumulation for `Float16` and `BFloat16`. Device offsets are copied to
the host once per call, since cuBLAS requires group sizes on the host. Other arrays
use a loop of `mul!` calls, parallelized over groups on multithreaded CPUs.

```jldoctest
julia> W = cat(ones(2, 3), fill(2.0, 2, 3); dims=3);

julia> grouped_mul(W, ones(3, 4), [1, 3])
2×4 Matrix{Float64}:
 3.0  6.0  6.0  0.0
 3.0  6.0  6.0  0.0

julia> size(grouped_mul(ones(2, 4), ones(4, 3), [1, 3]))
(2, 3, 2)
```
"""
function grouped_mul(A::AbstractArray{<:Any,3}, B::AbstractMatrix,
                     offsets::AbstractVector{<:Integer})
    C = similar(A, promote_type(eltype(A), eltype(B)), _grouped_mul_size(A, B, length(offsets)))
    return grouped_mul!(C, A, B, offsets)
end

function grouped_mul(A::AbstractMatrix, B::AbstractMatrix, offsets::AbstractVector{<:Integer})
    C = similar(A, promote_type(eltype(A), eltype(B)), _grouped_mul_size(A, B, length(offsets)))
    return grouped_mul!(C, A, B, offsets)
end

function _grouped_mul_size(A::AbstractArray{<:Any,3}, B::AbstractMatrix, groups)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("grouped_mul inner dimensions must match"))
    size(A, 3) == groups || throw(DimensionMismatch("grouped_mul needs one offset per matrix"))
    return (size(A, 1), size(B, 2))
end

function _grouped_mul_size(A::AbstractMatrix, B::AbstractMatrix, groups)
    size(A, 2) == size(B, 1) || throw(DimensionMismatch("grouped_mul inner dimensions must match"))
    return (size(A, 1), size(B, 2), groups)
end

function _grouped_offsets(offsets, total)
    Base.require_one_based_indexing(offsets)
    ends = offsets isa Vector ? offsets : Array(offsets)
    previous = 0
    for stop in ends
        previous <= stop <= total || throw(ArgumentError(
            "grouped_mul offsets must be nondecreasing and lie between 0 and $total"))
        previous = stop
    end
    return ends
end

"""
    grouped_mul!(C, A, B, offsets, α=1, β=0) -> C

Compute `C = α * grouped_mul(A, B, offsets) + β * C` in place.
The destination must have the shape returned by [`grouped_mul`](@ref) and must
not alias either input. Entries outside the groups are scaled by `β`.
"""
function grouped_mul!(C::AbstractArray{T}, A::AbstractArray, B::AbstractMatrix,
                      offsets::AbstractVector{<:Integer}, α::Number=one(T), β::Number=zero(T)) where {T}
    Base.require_one_based_indexing(C, A, B)
    size(C) == _grouped_mul_size(A, B, length(offsets)) ||
        throw(DimensionMismatch("grouped_mul destination has the wrong size"))
    total = ndims(A) == 3 ? size(B, 2) : size(A, 2)
    ends = _grouped_offsets(offsets, total)
    _grouped_mul!(storage_typejoin(C, A, B), C, A, B, ends, α, β)
    if ndims(A) == 3
        stop = isempty(ends) ? 0 : Int(last(ends))
        scale!(view(C, :, stop+1:total), β)
    end
    return C
end

_grouped_matrix(A, g) = view(A, :, :, g)
_grouped_matrix(A::BatchedTranspose, g) = transpose(view(parent(A), :, :, g))
_grouped_matrix(A::BatchedAdjoint, g) = adjoint(view(parent(A), :, :, g))
_grouped_matrix(A::PermutedDimsArray{<:Any,3,(2,1,3)}, g) = transpose(view(parent(A), :, :, g))

_grouped_columns(A, r) = view(A, :, r)
_grouped_columns(A::Transpose, r) = transpose(view(parent(A), r, :))
_grouped_columns(A::Adjoint, r) = adjoint(view(parent(A), r, :))
_grouped_rows(A, r) = view(A, r, :)
_grouped_rows(A::Transpose, r) = transpose(view(parent(A), :, r))
_grouped_rows(A::Adjoint, r) = adjoint(view(parent(A), :, r))

_grouped_views(C, A::AbstractArray{<:Any,3}, B, g, r) =
    (view(C, :, r), _grouped_matrix(A, g), _grouped_columns(B, r))
_grouped_views(C, A::AbstractMatrix, B, g, r) =
    (view(C, :, :, g), _grouped_columns(A, r), _grouped_rows(B, r))

_grouped_range(ends, g) = (g == 1 ? 0 : Int(ends[g-1]))+1:Int(ends[g])

function _grouped_product!(C, A, B, α, β)
    if size(A, 2) == 0
        scale!(C, β)
    else
        mul!(C, A, B, α, β)
    end
    return C
end

_grouped_mul!(::Type, C, A, B, ends, α, β) = _grouped_mul_generic!(C, A, B, ends, α, β)

function _grouped_mul_generic!(C, A, B, ends, α, β)
    function multiply_groups(groups)
        for g in groups
            c, a, b = _grouped_views(C, A, B, g, _grouped_range(ends, g))
            _grouped_product!(c, a, b, α, β)
        end
    end
    if storage_type(C) <: Array && should_use_spawn() && length(ends) > 1
        chunks = Iterators.partition(eachindex(ends), cld(length(ends), Threads.nthreads(:default)))
        with_single_threaded_blas() do
            @sync for groups in chunks
                Threads.@spawn multiply_groups(groups)
            end
        end
    else
        multiply_groups(eachindex(ends))
    end
    return C
end

function rrule(::typeof(grouped_mul), A::AbstractArray{<:Any,3}, B::AbstractMatrix,
               offsets::AbstractVector{<:Integer})
    ends = _grouped_offsets(offsets, size(B, 2))
    project_A, project_B = ProjectTo(A), ProjectTo(B)
    function grouped_mul_pullback(Δ)
        dC = unthunk(Δ)
        dC isa AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent())
        dA = @thunk project_A(grouped_mul(dC, adjoint(B), ends))
        dB = @thunk project_B(grouped_mul(batched_adjoint(A), dC, ends))
        return (NoTangent(), dA, dB, NoTangent())
    end
    return grouped_mul(A, B, ends), grouped_mul_pullback
end

function rrule(::typeof(grouped_mul), A::AbstractMatrix, B::AbstractMatrix,
               offsets::AbstractVector{<:Integer})
    ends = _grouped_offsets(offsets, size(A, 2))
    project_A, project_B = ProjectTo(A), ProjectTo(B)
    function grouped_mul_pullback(Δ)
        dC = unthunk(Δ)
        dC isa AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent())
        dA = @thunk project_A(grouped_mul(dC, adjoint(B), ends))
        dB = @thunk project_B(copy(adjoint(grouped_mul(batched_adjoint(dC), A, ends))))
        return (NoTangent(), dA, dB, NoTangent())
    end
    return grouped_mul(A, B, ends), grouped_mul_pullback
end
