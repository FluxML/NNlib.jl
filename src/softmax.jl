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
    out .= sf .* (Δ .- sum(Δ .*sf, dims = 1))
end

∇softmax(Δ, xs) = ∇softmax!(similar(Δ), Δ, xs)
∇softmax!(Δ, xs) = ∇softmax!(Δ, Δ, xs)


"""
    logsoftmax(xs) = log.(exp.(xs) ./ sum(exp.(xs)))

`logsoftmax(xs)` computes the log of `softmax(xs)`, but in a more numerically stable
way than directly taking the log of the the softmax function, which is commonly used in
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
∇logsoftmax(Δ, xs) = ∇softmax(Δ ./ max.(eps(eltype(xs)),softmax(xs)), xs)
∇logsoftmax!(Δ, xs) = ∇softmax!(Δ, Δ, xs)