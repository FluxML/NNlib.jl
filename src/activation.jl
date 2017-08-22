σ(x) = 1 / (1 + exp(-x))

relu(x) = max(0, x)

softmax!(out::AbstractVecOrMat, xs::AbstractVecOrMat) =
  out .= exp.(xs) ./ sum(exp, xs, 1)

softmax!(xs) = softmax!(xs, xs)

softmax(xs) = softmax!(similar(xs), xs)
