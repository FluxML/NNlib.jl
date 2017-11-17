"""
    σ(x) = 1 / (1 + exp(-x))

Classic [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation
function.
"""
σ(x) = 1 / (1 + exp(-x))

# ForwardDiff numerical stability hack
σ(x::Float32) = ifelse(x < -80, zero(x), 1 / (1 + exp(-x)))

"""
    relu(x) = max(0, x)

[Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function.
"""
relu(x) = max(0, x)

"""
    leakyrelu(x) = max(0.01x, x)

Leaky [Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function.

You can also specify the coefficient explicitly, e.g. `leakyrelu(x, 0.01)`.
"""
leakyrelu(x, a = oftype(x, 0.01)) = max(a*x, x)

"""
    elu(x; α = 1) = x > 0 ? x : α * (exp(x) - one(x))

Exponential Linear Unit activation function. See [Fast and Accurate Deep Network
Learning by Exponential Linear Units](https://arxiv.org/abs/1511.07289)
"""
elu(x, α = one(x)) = ifelse(x ≥ 0, x, α * (exp(x) - one(x)))

"""
    swish(x) = x * σ(x)

Self-gated actvation function.

See [Swish: a Self-Gated Activation Function](https://arxiv.org/pdf/1710.05941.pdf).
"""
swish(x) = x * σ(x)

"""

    tanh(x) = 2 * σ(2*x) - 1

Hyperbolic activation function.

See [Hyperbolic function: (https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent)]
"""
tanh(x) = 2 * σ(2*x) - 1