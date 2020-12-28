export σ, sigmoid, hardσ, hardsigmoid, hardtanh, relu, leakyrelu, relu6, rrelu, elu, gelu, swish, selu, celu, softplus, softsign, logσ,
       logsigmoid, logcosh, mish, tanhshrink, softshrink, thresholdrelu, trelu, lisht

## Activation functions
#
# Some of activation functions have its wrapper function for GPU in CuArrays.jl.
# https://github.com/JuliaGPU/CuArrays.jl/issues/614

"""
    σ(x) = 1 / (1 + exp(-x))

Classic [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation
function.
"""
function σ(x)
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(one(t) + t), t / (one(t) + t))
end
const sigmoid = σ

"""
    hardσ(x, a=0.2) = max(0, min(1.0, a * x + 0.5))

Segment-wise linear approximation of sigmoid.
See [BinaryConnect: Training Deep Neural Networks withbinary weights during propagations](https://arxiv.org/abs/1511.00363).
"""
hardσ(x, a=0.2) = oftype(x/1, max(zero(x/1), min(one(x/1), oftype(x/1,a) * x + oftype(x/1,0.5))))
const hardsigmoid = hardσ


"""
    logσ(x)

Return `log(σ(x))` which is computed in a numerically stable way.

    julia> logσ(0)
    -0.6931471805599453
    julia> logσ.([-100, -10, 100])
    3-element Array{Float64,1}:
     -100.0
      -10.000045398899218
       -3.720075976020836e-44
"""
logσ(x) = -softplus(-x)
const logsigmoid = logσ


"""
    hardtanh(x) = max(-1, min(1, x))

Segment-wise linear approximation of tanh. Cheaper  and  more  computational  efficient version of tanh.
See [Large Scale Machine Learning](https://ronan.collobert.com/pub/matos/2004_phdthesis_lip6.pdf).
"""
hardtanh(x) = max(-one(x), min( one(x), x))


"""
    relu(x) = max(0, x)

[Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function.
"""
relu(x) = max(zero(x), x)


"""
    leakyrelu(x, a=0.01) = max(a*x, x)

Leaky [Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function.
You can also specify the coefficient explicitly, e.g. `leakyrelu(x, 0.01)`.
"""
leakyrelu(x, a = oftype(x / 1, 0.01)) = max(a * x, x / one(x))

"""
    relu6(x) = min(max(0, x), 6)

[Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function capped at 6.
See [Convolutional Deep Belief Networks on CIFAR-10](https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf)
"""
relu6(x) = min(relu(x), oftype(x, 6))

"""
    rrelu(x, l=1/8, u=1/3) = max(a*x, x)

    a = randomly sampled from uniform distribution U(l, u)

Randomized Leaky [Rectified Linear Unit](https://arxiv.org/abs/1505.00853)
activation function.
You can also specify the bound explicitly, e.g. `rrelu(x, 0.0, 1.0)`.
"""
function rrelu(x, l = 1 / 8.0, u = 1 / 3.0)
    a = oftype(x / 1, (u - l) * rand() + l)
    return leakyrelu(x, a)
end

"""
    elu(x, α=1) =
      x > 0 ? x : α * (exp(x) - 1)

Exponential Linear Unit activation function.
See [Fast and Accurate Deep Network Learning by Exponential Linear Units](https://arxiv.org/abs/1511.07289).
You can also specify the coefficient explicitly, e.g. `elu(x, 1)`.
"""
elu(x, α = one(x)) = ifelse(x ≥ 0, x / one(x), α * (exp(x) - one(x)))


"""
    gelu(x) = 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))

[Gaussian Error Linear Unit](https://arxiv.org/abs/1606.08415)
activation function.
"""
function gelu(x)
    p = oftype(x / 1, π)
    λ = oftype(x / 1, √(2 / p))
    α = oftype(x / 1, 0.044715)
    h = oftype(x / 1, 0.5)
    h * x * (one(x) + tanh(λ * (x + α * x^3)))
end


"""
    swish(x) = x * σ(x)

Self-gated activation function.
See [Swish: a Self-Gated Activation Function](https://arxiv.org/abs/1710.05941).
"""
swish(x) = x * σ(x)


"""
    lisht(x) = x * tanh(x)

Non-Parametric Linearly Scaled Hyperbolic Tangent Activation Function.
See [LiSHT](https://arxiv.org/abs/1901.05894)
"""
lisht(x) = x * tanh(x)


"""
    selu(x) = λ * (x ≥ 0 ? x : α * (exp(x) - 1))

    λ ≈ 1.0507
    α ≈ 1.6733

Scaled exponential linear units.
See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515).
"""
function selu(x)
  λ = oftype(x / 1, 1.0507009873554804934193349852946)
  α = oftype(x / 1, 1.6732632423543772848170429916717)
  λ * ifelse(x > 0, x / one(x), α * (exp(x) - one(x)))
end

"""
    celu(x, α=1) =
        (x ≥ 0 ? x : α * (exp(x/α) - 1))

Continuously Differentiable Exponential Linear Units
See [Continuously Differentiable Exponential Linear Units](https://arxiv.org/abs/1704.07483).
"""
celu(x, α = one(x)) = ifelse(x ≥ 0, x / one(x), α * (exp(x/α) - one(x)))


"""
    trelu(x, theta = 1.0) = x > theta ? x : 0

Threshold Gated Rectified Linear.
See [ThresholdRelu](https://arxiv.org/abs/1402.3337)
"""
trelu(x, theta = one(x)) = ifelse(x> theta, x, zero(x))
const thresholdrelu = trelu


"""
    softsign(x) = x / (1 + |x|)

See [Quadratic Polynomials Learn Better Image Features](http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/205).
"""
softsign(x) = x / (one(x) + abs(x))


"""
    softplus(x) = log(exp(x) + 1)

See [Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf).
"""
softplus(x) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))


"""
    logcosh(x)

Return `log(cosh(x))` which is computed in a numerically stable way.
"""
logcosh(x) = x + softplus(-2x) - log(oftype(x, 2))


"""
    mish(x) = x * tanh(softplus(x))

Self Regularized Non-Monotonic Neural Activation Function.
See [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681).
"""
mish(x) = x * tanh(softplus(x))

"""
    tanhshrink(x) = x - tanh(x)

See [Tanhshrink Activation Function](https://www.gabormelli.com/RKB/Tanhshrink_Activation_Function).
"""
tanhshrink(x) = x - tanh(x)

"""
    softshrink(x, λ=0.5) =
        (x ≥ λ ? x - λ : (-λ ≥ x ? x + λ : 0))

See [Softshrink Activation Function](https://www.gabormelli.com/RKB/Softshrink_Activation_Function).
"""
softshrink(x, λ = oftype(x/1, 0.5)) = min(max(zero(x), x - λ), x + λ)

# Provide an informative error message if activation functions are called with an array
for f in (:σ, :hardσ, :logσ, :hardtanh, :relu, :leakyrelu, 
        :relu6, :rrelu, :elu, :gelu, :swish, :lisht, :selu, 
        :celu, :trelu, :softsign, :softplus, :logcosh, :mish, :tanhshrink, :softshrink)
    @eval $(f)(x::AbstractArray, args...) =
      error("Use broadcasting (`", $(string(f)), ".(x)`) to apply activation functions to arrays.")
end

# Define rrules for broadcasted activation functions.
# This is a performance hack is specifically for Zygote, because it doesn't handle fused
# broadcasts well; but it generally should be good (or at least harmless) for any AD, as 
# it saves ADing the broadcasting machinery.
for (f, df) in [
    (:relu, :(x .> 0)),
    (:selu, :(dselu.(x))),
    (:elu, :(delu.(x))),
    (:σ, :(conj.(Ω .* (1 .- Ω)))),
    ]

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(broadcasted),
                                                ::typeof($f), x::Numeric)
        Ω = $f.(x)
        $pullback(Δ) = (NO_FIELDS, NO_FIELDS, Δ .* $df)
        return Ω, $pullback
    end
end
