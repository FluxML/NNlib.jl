export softmax, softmax!, ∇softmax, ∇softmax!,
       logsoftmax, logsoftmax!, ∇logsoftmax, ∇logsoftmax!

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
```julia-repl
julia> softmax([1, 2, 3])
3-element Array{Float64,1}:
  0.0900306
  0.244728
  0.665241
```

See also [`logsoftmax`](@ref).
"""
function softmax(xs::AbstractArray; dims=1)
    max_ = maximum(xs, dims=dims)
    exp_ = exp.(xs .- max_)
    exp_ ./ sum(exp_, dims=dims)
end

function softmax!(out::AbstractVecOrMat{T}, xs::AbstractVecOrMat{T}) where {T}
    @inbounds for j = 1:size(xs, 2)
        # First, store column-wise maximum in the last element of `out`
        out[end, j] = xs[end, j]
        @inbounds for i = 1:(size(xs, 1) - 1)
            out[end, j] = max(out[end, j], xs[i, j])
        end

        # Subtract the column-wise maximums to normalize, take exp()
        # out .= exp(xs .- out[end, :])
        @inbounds for i = 1:size(out, 1)
            out[i, j] = exp(xs[i, j] - out[end, j])
        end

        # Normalize by sum of the entire thing
        # out ./= sum(out, 1)
        s = T(0)
        @inbounds for i = 1:size(out, 1)
            s += out[i, j]
        end
        @inbounds for i = 1:size(out, 1)
            out[i, j] /= s
        end
    end
    return out
end

function ∇softmax!(out::AbstractVecOrMat, Δ::AbstractVecOrMat, xs::AbstractVecOrMat)
    sf = softmax(xs)
    out .= sf .* (Δ .- sum(Δ .* sf, dims = 1))
end
function ∇softmax(Δ, xs; dims=1)
    sf = softmax(xs, dims=dims)
    sf .* (Δ .- sum(Δ .* sf, dims=dims))
end
∇softmax!(Δ, xs) = ∇softmax!(Δ, Δ, xs)


"""
    logsoftmax(x; dims=1)

Computes the log of softmax in a more numerically stable
way than directly taking `log.(softmax(xs))`. Commonly used in
computing cross entropy loss.

It is semantically equivalent to the following:

    logsoftmax(x; dims=1) = x .- log.(sum(exp.(x), dims=dims))

See also [`softmax`](@ref).
"""
function logsoftmax(xs::AbstractArray; dims=1)
    max_ = maximum(xs, dims=dims)
    exp_ = exp.(xs .- max_)
    log_ = log.(sum(exp_, dims=dims))
    (xs .- max_) .- log_
end

function logsoftmax!(out::AbstractVecOrMat, xs::AbstractVecOrMat)
    for j = 1:size(xs, 2)
        @inbounds begin
            xi_max = xs[1, j]
            for i = 1:size(out, 1)
                xi_max = max(xi_max, xs[i, j])
            end
            s = zero(eltype(out))
            for i = 1:size(out, 1)
                s += exp(xs[i, j] - xi_max)
            end
            for i = 1:size(out, 1)
                out[i, j] = xs[i, j] - log(s) - xi_max
            end
        end
    end
    return out
end

∇logsoftmax(Δ, xs; dims=1) = Δ .- sum(Δ, dims=dims) .* softmax(xs, dims=dims)
∇logsoftmax!(Δ, xs) = ∇softmax!(Δ, Δ, xs)
