export softmax, softmax!, ∇softmax, ∇softmax!,
       logsoftmax, logsoftmax!, ∇logsoftmax, ∇logsoftmax!

"""
    softmax(xs) = exp.(xs) ./ sum(exp.(xs))

[Softmax](https://en.wikipedia.org/wiki/Softmax_function) takes
log-probabilities (any real vector) and returns a probability distribution that
sums to 1.

If given a matrix it will treat it as a batch of vectors, with each column
independent.

    julia> softmax([1,2,3.])
    3-element Array{Float64,1}:
      0.0900306
      0.244728
      0.665241
"""
softmax(xs) = softmax!(similar(xs), xs)

function softmax!(out::AbstractVecOrMat{T}, xs::AbstractVecOrMat{T}) where {T}
    # First, store column-wise maximum in the last element of `out`
    maxdims = ntuple(d -> (d in dims) ? (1:1) : (:), ndims(xs))
    out[maxdims...] = maximum!(out[maxdims...], xs)

    # Subtract the column-wise maximums to normalize, take exp()
    out .= exp.(xs.-out[maxdims...])

    # Normalize by sum of the entire thing
    out ./= sum(out, dims=dims)
    return out
end

function ∇softmax!(out::AbstractVecOrMat, Δ::AbstractVecOrMat, xs::AbstractVecOrMat)
    sf = softmax(xs)
    out .= sf .* (Δ .- sum(Δ .*sf, dims = 1))
end

∇softmax(Δ, xs) = ∇softmax!(similar(Δ), Δ, xs)
∇softmax!(Δ, xs) = ∇softmax!(Δ, Δ, xs)


"""
    logsoftmax(xs) = log.(exp.(xs) ./ sum(exp.(xs)))

`logsoftmax(xs)` computes the log of `softmax(xs)`, but in a more numerically stable
way than directly taking the log of the softmax function, which is commonly used in
computing cross entropy loss.
"""
logsoftmax(xs) = logsoftmax!(similar(xs), xs)
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
∇logsoftmax(Δ, xs) = Δ - sum(Δ, dims=1) .* softmax(xs)
∇logsoftmax!(Δ, xs) = ∇softmax!(Δ, Δ, xs)
