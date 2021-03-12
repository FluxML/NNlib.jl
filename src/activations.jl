## Activation functions
#
# Some of activation functions have its wrapper function for GPU in NNlibCUDA.jl.
# https://github.com/JuliaGPU/CuArrays.jl/issues/614

const ACTIVATIONS = 
    [:σ, :hardσ, :hardtanh, :relu, 
    :leakyrelu, :relu6, :rrelu, :elu, :gelu, :swish, :selu, 
    :celu, :softplus, :softsign, :logσ, :logcosh, 
    :mish, :tanhshrink, :softshrink, :trelu, 
    :lisht]

for f in ACTIVATIONS
    @eval export $(f)
end

# Aliases
export sigmoid, hardsigmoid, logsigmoid, thresholdrelu

# of type float
oftf(x, y) = oftype(float(x), y)

"""
    σ(x) = 1 / (1 + exp(-x))

Classic [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation
function.
"""
function σ(x)
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

const sigmoid = σ

"""
    hardσ(x) = max(0, min(1, (x + 3) / 6)

Piecewise linear approximation of sigmoid.
"""
hardσ(x) = max(0, min(1, (x + 3) / 6))

# https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html

const hardsigmoid = hardσ

"""
    logσ(x)

Return `log(σ(x))` which is computed in a numerically stable way.
"""
logσ(x) = -softplus(-x)
const logsigmoid = logσ

"""
    hardtanh(x) = max(-1, min(1, x))

Segment-wise linear approximation of tanh. Cheaper  and  more  computational  efficient version of tanh.
See [Large Scale Machine Learning](https://ronan.collobert.com/pub/matos/2004_phdthesis_lip6.pdf).
"""
hardtanh(x) = max(-one(x), min(one(x), x))

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
leakyrelu(x, a=oftf(x, 0.01)) = max(a * x, x)

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
function rrelu(x::T, l=1//8, u=1//3) where T<:Number
    a = (u - l) * rand(float(T)) + l
    return leakyrelu(x, a)
end

"""
    elu(x, α=1) = x > 0 ? x : α * (exp(x) - 1)

Exponential Linear Unit activation function.
See [Fast and Accurate Deep Network Learning by Exponential Linear Units](https://arxiv.org/abs/1511.07289).
You can also specify the coefficient explicitly, e.g. `elu(x, 1)`.
"""
elu(x, α=1) = ifelse(x ≥ 0, float(x), α * (exp(x) - 1))

deriv_elu(Ω, α=1) = ifelse(Ω ≥ 0, 1, Ω + α)

"""
    gelu(x) = 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))

[Gaussian Error Linear Unit](https://arxiv.org/abs/1606.08415)
activation function.
"""
function gelu(x)
    α = oftf(x, 0.044715)
    λ = oftf(x, gelu_λ)
    x/2 * (1 + tanh(λ * (x + α * x^3)))
end

const gelu_λ = √(2 / π)

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

    λ ≈ 1.05070...
    α ≈ 1.67326...

Scaled exponential linear units.
See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515).
"""
function selu(x)
    λ = oftf(x, selu_λ)
    α = oftf(x, selu_α)
    λ * ifelse(x > 0, x, α * (exp(x) - 1))
end

const selu_λ = 1.0507009873554804934193349852946
const selu_α = 1.6732632423543772848170429916717

function deriv_selu(Ω)
    λ = oftf(Ω, selu_λ)
    α = oftf(Ω, selu_α)
    ifelse(Ω > 0, λ, Ω + α * λ)
end

"""
    celu(x, α=1) = x ≥ 0 ? x : α * (exp(x/α) - 1)

Continuously Differentiable Exponential Linear Units
See [Continuously Differentiable Exponential Linear Units](https://arxiv.org/abs/1704.07483).
"""
celu(x, α=1) = ifelse(x ≥ 0, float(x), α * (exp(x/α) - 1))

"""
    trelu(x, theta=1) = x > theta ? x : 0

Threshold Gated Rectified Linear.
See [ThresholdRelu](https://arxiv.org/abs/1402.3337)
"""
trelu(x, theta=1) = ifelse(x > theta, x, zero(x))

const thresholdrelu = trelu

"""
    softsign(x) = x / (1 + |x|)

See [Quadratic Polynomials Learn Better Image Features](http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/205).
"""
softsign(x) = x / (1 + abs(x))

"""
    softplus(x) = log(exp(x) + 1)

See [Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf).
"""
softplus(x) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))

"""
    logcosh(x)

Return `log(cosh(x))` which is computed in a numerically stable way.
"""
logcosh(x) = x + softplus(-2x) - oftf(x, log2)

const log2 = log(2)

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
softshrink(x, λ=oftf(x, 0.5)) = min(max(0, x - λ), x + λ)

# Provide an informative error message if activation functions are called with an array
for f in ACTIVATIONS
    @eval $(f)(x::AbstractArray, args...) =
      error("Use broadcasting (`", $(string(f)), ".(x)`) to apply activation functions to arrays.")
end

## Define rrules for some activation functions, along with the 
## broadcasted rrule activation functions.
## TODO: add to the lists below all activations. 

## This is a performance hack specifically for Zygote, because it doesn't handle fused
## broadcasts well; but it generally should be good (or at least harmless) for any AD, as 
## it saves ADing the broadcasting machinery.
## Related Issue https://github.com/JuliaDiff/ChainRulesCore.jl/issues/271
  
UNARY_ACTS = [ # f, df
    (:relu,         :(x > 0)),
    (:hardtanh,     :(-1 < x < 1)),
    (:selu,         :(deriv_selu(Ω))),
    (:σ,            :(conj(Ω * (1 - Ω)))),
    (:elu,          :(deriv_elu(Ω))),
    ]

for (f, df) in UNARY_ACTS
    @eval @scalar_rule($f(x), $df)

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function rrule(::typeof(broadcasted),
                         ::typeof($f), x::Numeric)
        Ω = $f.(x)
        function $pullback(Δ) 
            NO_FIELDS, NO_FIELDS, @.(Δ * $df)
        end
        return Ω, $pullback
    end
end


BINARY_ACTS = [ # f, df1, df2
    (:elu, :(deriv_elu(Ω, x2)), :(DoesNotExist())), # TODO use real deriv instead of DNE
    ]

for (f, df1, df2) in BINARY_ACTS
    @eval @scalar_rule($f(x1, x2), ($df1, $df2))

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function rrule(::typeof(broadcasted),
                         ::typeof($f), 
                         x1::Numeric, x2::Numeric)
        Ω = $f.(x1, x2)
        function $pullback(Δ) 
            NO_FIELDS, NO_FIELDS, @.(Δ * $df1), @.(Δ * $df2)
        end
        return Ω, $pullback
    end
end
