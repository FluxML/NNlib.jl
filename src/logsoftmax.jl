using Base.Threads

function logsoftmax!(out::AbstractVecOrMat, xs::AbstractVecOrMat)
  out .= xs .- maximum(xs, 1)
  out .= out .- log.(sum(exp.(out), 1))
  return out
end

logsoftmax!(xs) = logsoftmax!(xs, xs)
logsoftmax(xs) = logsoftmax!(similar(xs), xs)

∇logsoftmax!(Δ, xs) = ∇softmax!(Δ, Δ ./ softmax(xs), xs)
∇logsoftmax(Δ, xs) = ∇softmax!(similar(Δ), Δ ./ softmax(xs), xs)

"""
    logsoftmax(xs) = xs .- log.(sum(exp.(xs)))

[Softmax](https://en.wikipedia.org/wiki/Softmax_function) takes
log-probabilities (any real vector) and returns a probability distribution that
sums to 1.
logsoftmax computes the log of this vector. It is numerically more stable than
softmax
"""
softmax
