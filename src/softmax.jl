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

** Usage Examples**

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

function softmax!(out::O, x::T; dims = 1) where {O<:AbstractArray,T<:AbstractArray}
    out .= exp.(x .- maximum(x; dims = dims))
    out ./= sum(out; dims = dims)
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

function logsoftmax!(out::O, x::T; dims = 1) where {O<:AbstractArray,T<:AbstractArray}
    out .= x .- maximum(x; dims = dims)
    # out .-= log.(sum(exp.(out); dims = dims))  # WARN: this will decrease performance.
    log_ = log.(sum(exp, out; dims = dims))
    out .-= log_
end

∇logsoftmax(Δ::AbstractArray{T}, x::AbstractArray, y::AbstractArray{S}; dims = 1) where {T,S} = 
    ∇logsoftmax!(similar(y, promote_type(T, S)), Δ, x, y; dims = dims)

## Can introduce at the end of deprecation cycle of ∇logsoftmax!(out, Δ, x; dims = 1)  
# ∇logsoftmax!(Δ, x, y; dims = 1) = ∇logsoftmax!(Δ, Δ, x, y; dims = dims)

function ∇logsoftmax!(out::AbstractArray, Δ::AbstractArray,
                    x::AbstractArray, y::AbstractArray; dims = 1) 
    out .= Δ .- sum(Δ, dims = dims) .* exp.(y)
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
