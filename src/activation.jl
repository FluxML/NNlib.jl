"""
    σ(x) = 1 / (1 + exp(-x))

Classic [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation
function.
"""
σ(x) = one(x) / (one(x) + exp(-x))

const sigmoid = σ

# ForwardDiff numerical stability hack
σ_stable(x) = ifelse(x < -80, zero(x), one(x) / (one(x) + exp(-x)))

σ(x::Float32) = σ_stable(x)

@init @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin
  σ(x::ForwardDiff.Dual{T,Float32}) where T = σ_stable(x)
end

"""
    logσ(x)

Return `log(σ(x))` which is computed in a numerically stable way.

    julia> logσ(0.)
    -0.6931471805599453
    julia> logσ.([-100, -10, 100.])
    3-element Array{Float64,1}:
     -100.0
      -10.0
       -0.0
"""
function logσ(x)
  max_v = max(zero(x), -x)
  z = exp(-max_v) + exp(-x-max_v)
  -(max_v + log(z))
end

const logsigmoid = logσ

"""
    relu(x) = max(0, x)

[Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function.
"""
relu(x) = max(zero(x), x)


"""
    leakyrelu(x) = max(0.01x, x)

Leaky [Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function.
You can also specify the coefficient explicitly, e.g. `leakyrelu(x, 0.01)`.
"""
leakyrelu(x, a = oftype(x/1, 0.01)) = max(a*x, x/1)

"""
    elu(x, α = 1) =
      x > 0 ? x : α * (exp(x) - 1)

Exponential Linear Unit activation function.
See [Fast and Accurate Deep Network Learning by Exponential Linear Units](https://arxiv.org/abs/1511.07289).
You can also specify the coefficient explicitly, e.g. `elu(x, 1)`.
"""
elu(x, α = one(x)) = ifelse(x ≥ 0, x/1, α * (exp(x) - one(x)))

"""
    swish(x) = x * σ(x)

Self-gated actvation function.
See [Swish: a Self-Gated Activation Function](https://arxiv.org/pdf/1710.05941.pdf).
"""
swish(x) = x * σ(x)

"""
    selu(x) = λ * (x ≥ 0 ? x : α * (exp(x) - 1))

    λ ≈ 1.0507
    α ≈ 1.6733

Scaled exponential linear units.
See [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515.pdf).
"""
function selu(x)
  λ = oftype(x/1, 1.0507009873554804934193349852946)
  α = oftype(x/1, 1.6732632423543772848170429916717)
  λ * ifelse(x > 0, x/1, α * (exp(x) - 1))
end

"""
    softsign(x) = x / (1 + |x|)

See [Quadratic Polynomials Learn Better Image Features](http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/205).
"""
softsign(x) = x / (one(x) + abs(x))


"""
    softplus(x) = log(exp(x) + 1)

See [Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf).
"""
softplus(x) = log1p(exp(x))
