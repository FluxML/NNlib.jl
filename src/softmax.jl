export softmax,
    softmax!,
    ∇softmax,
    ∇softmax!,
    logsoftmax,
    logsoftmax!,
    ∇logsoftmax,
    ∇logsoftmax!,
    logsumexp

"""
    softmax(x; dims=1)

[Softmax](https://en.wikipedia.org/wiki/Softmax_function) turns input array `x`
into probability distributions that sum to 1 along the dimensions specified by `dims`.
It is semantically equivalent to the following:

    softmax(x; dims=1) = exp.(x) ./ sum(exp.(x), dims=dims)

with additional manipulations enhancing numerical stability.

For a matrix input `x` it will by default (`dims=1`) treat it as a batch of vectors,
with each column independent. Keyword `dims=2` will instead treat rows independently,
etc...

See also [`logsoftmax`](@ref).

# Examples

```jldoctest
julia> softmax([1, 2, 3])
3-element Array{Float64,1}:
  0.0900306
  0.244728
  0.665241

julia> softmax([1 2 3; 2 2 2])  # dims=1
2×3 Array{Float64,2}:
 0.268941  0.5  0.731059
 0.731059  0.5  0.268941

julia> softmax([1 2 3; 2 2 2]; dims=2)
2×3 Array{Float64,2}:
 0.0900306  0.244728  0.665241
 0.333333   0.333333  0.333333
```
"""
softmax(x; dims = 1) = softmax!(similar(x, (float ∘ eltype)(x)), x; dims = dims)

softmax!(x; dims = 1) = softmax!(x, x; dims = dims)

function softmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
    max_ = maximum(x; dims = dims)
    if all(isfinite, max_)
        out .= exp.(x .- max_)
    else
        @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))
    end
    out ./= sum(out; dims = dims)  # could re-use max_ when dims != (:) and eltype(x) == T.
end

∇softmax(Δ::AbstractArray{T}, x::AbstractArray, y::AbstractArray{S}; dims = 1) where {T,S} = 
    ∇softmax!(similar(y, promote_type(T, S)), Δ, x, y; dims = dims)

## Can introduce at the end of deprecation cycle of ∇softmax!(out, Δ, x; dims = 1)  
#∇softmax!(Δ, x, y; dims = 1) = ∇softmax!(Δ, Δ, x, y; dims = dims)

function ∇softmax!(out::AbstractArray, Δ::AbstractArray, 
                    x::AbstractArray, y::AbstractArray; dims = 1)
    out .= Δ .* y
    out .= out .- y .* sum(out; dims = dims)
end

# Old 2-arg version recomputing forward
∇softmax(Δ, x; dims=1) = ∇softmax(Δ, x, softmax(x, dims=dims); dims=dims)
∇softmax!(Δ, x; dims=1) = ∇softmax!(Δ, Δ, x, softmax(x, dims=dims); dims=dims)
∇softmax!(out, Δ, x; dims=1) = ∇softmax!(out, Δ, x, softmax(x, dims=dims); dims=dims)

function rrule(::typeof(softmax), xs; dims=1)
    y = softmax(xs; dims=dims)
    softmax_pullback(Δ) = (NO_FIELDS, ∇softmax(Δ, xs, y, dims=dims))
    return y, softmax_pullback
end

"""
    logsoftmax(x; dims=1)

Computes the log of softmax in a more numerically stable
way than directly taking `log.(softmax(xs))`. Commonly used in
computing cross entropy loss.

It is semantically equivalent to the following:

    logsoftmax(x; dims=1) = x .- log.(sum(exp.(x), dims=dims))

See also [`softmax`](@ref).
"""
logsoftmax(x; dims = 1) = logsoftmax!(similar(x, (float ∘ eltype)(x)), x; dims = dims)

logsoftmax!(x; dims = 1) = logsoftmax!(x, x; dims = dims)

function logsoftmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
    max_ = maximum(x; dims = dims)
    if all(isfinite, max_)
        out .= x .- max_
    else
        @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 0, -Inf), x - max_)
    end
    log_ = log.(sum(exp, out; dims = dims))
    out .-= log_
end

∇logsoftmax(Δ::AbstractArray{T}, x::AbstractArray, y::AbstractArray{S}; dims = 1) where {T,S} =
    ∇logsoftmax!(similar(y, promote_type(T, S)), Δ, x, y; dims = dims)

# Old 2-arg version recomputing forward
∇logsoftmax(Δ, x; dims=1) =  ∇logsoftmax(Δ, x, logsoftmax(x, dims=dims); dims=dims)
∇logsoftmax!(Δ, x; dims=1) =  ∇logsoftmax!(Δ, Δ, x, logsoftmax(x, dims=dims); dims=dims)
∇logsoftmax!(out, Δ, x; dims=1) =  ∇logsoftmax!(out, Δ, x, logsoftmax(x, dims=dims); dims=dims)
    
function ∇logsoftmax!(out::AbstractArray, Δ::AbstractArray,
                    x::AbstractArray, y::AbstractArray; dims = 1) 
    out .= Δ .- sum(Δ, dims = dims) .* exp.(y)
end

function rrule(::typeof(logsoftmax), xs; dims=1)
    y = logsoftmax(xs; dims=dims)
    logsoftmax_pullback(Δ) = (NO_FIELDS, ∇logsoftmax(Δ, xs, y, dims=dims))
    return y, logsoftmax_pullback
end

"""
    logsumexp(x; dims=:)

Computes `log.(sum(exp.(x); dims=dims))` in a numerically stable
way.

See also [`logsoftmax`](@ref).
"""
function logsumexp(x::AbstractArray; dims = :)
    max_ = maximum(x; dims = dims)
    max_ .+ log.(sum(exp.(x .- max_); dims = dims))
end
