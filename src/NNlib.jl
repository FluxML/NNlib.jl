module NNlib

export σ, relu, softmax

σ(x) = 1 / (1 + exp(-x))

relu(x) = max(0, x)

softmax(xs) = exp.(xs) ./ sum(exp.(xs), ndims(xs))

end # module
