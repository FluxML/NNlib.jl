σ(x) = 1 / (1 + exp(-x))

relu(x) = max(0, x)

softmax(xs::AbstractVector) = exp.(xs) ./ sum(exp.(xs))

softmax(xs::AbstractMatrix) = exp.(xs) ./ sum(exp.(xs), 1)
