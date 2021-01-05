## Activation functions
#
# Some of activation functions have its wrapper function for GPU in CUDA.jl.
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
    hardσ(x, a=0.2) = max(0, min(1, a * x + 0.5))

Segment-wise linear approximation of sigmoid.
See [BinaryConnect: Training Deep Neural Networks withbinary weights during propagations](https://arxiv.org/abs/1511.00363).
"""
hardσ(x, a=0.2) = oftype(x/1, max(zero(x/1), min(one(x/1), oftype(x/1,a) * x + oftype(x/1,0.5))))
    
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
leakyrelu(x, a = oftype(x/1, 0.01)) = max(a * x, x/1)

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
   sinerelu(x, epsilon = 0.0025) = x > 0 ? x : epsilon * (sin(x) - cos(x))
   
Sine Rectified Linear Unit activation function
See [SineReLU - An alternative to the ReLU](https://medium.com/@wilder.rodrigues/sinerelu-an-alternative-to-the-relu-activation-function-e46a6199997d)
"""
function sinerelu(x, epsilon = 0.0025)
    return x > 0 ? x : epsilon * (sin(x) - cos(x))
end
"""
    elu(x, α=1) = x > 0 ? x : α * (exp(x) - 1)

Exponential Linear Unit activation function.
See [Fast and Accurate Deep Network Learning by Exponential Linear Units](https://arxiv.org/abs/1511.07289).
You can also specify the coefficient explicitly, e.g. `elu(x, 1)`.
"""
elu(x, α=1) = ifelse(x ≥ 0, x/1, α * (exp(x) - 1))

deriv_elu(x, Ω, α=1) = ifelse(x ≥ 0, one(x), Ω + α)


"""
    gelu(x) = 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))

[Gaussian Error Linear Unit](https://arxiv.org/abs/1606.08415)
activation function.
"""
function gelu(x)
    λ = oftype(x / 1, √(2 / π))
    α = oftype(x / 1, 0.044715)
    x/2 * (1 + tanh(λ * (x + α * x^3)))
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

    λ ≈ 1.05070...
    α ≈ 1.67326...

Scaled exponential linear units.
See [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515).
"""
function selu(x)
    λ = oftype(x/1, 1.0507009873554804934193349852946)
    α = oftype(x/1, 1.6732632423543772848170429916717)
    λ * ifelse(x > 0, x/1, α * (exp(x) - 1))
end

function deriv_selu(Ω)
    λ = oftype(Ω/1, 1.0507009873554804934193349852946)
    α = oftype(Ω/1, 1.6732632423543772848170429916717)
    return ifelse(Ω > 0, λ, Ω + α*λ)
end

"""
    celu(x, α=1) = x ≥ 0 ? x : α * (exp(x/α) - 1)

Continuously Differentiable Exponential Linear Units
See [Continuously Differentiable Exponential Linear Units](https://arxiv.org/abs/1704.07483).
"""
celu(x, α=1) = ifelse(x ≥ 0, x/1, α * (exp(x/α) - 1))

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
    (:elu,          :(deriv_elu(x, Ω))),
    ]

for (f, df) in UNARY_ACTS
    @eval @scalar_rule($f(x), $df)

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(broadcasted),
                                        ::typeof($f), x::Numeric)
        Ω = $f.(x)
        function $pullback(Δ) 
            NO_FIELDS, NO_FIELDS, @.(Δ * $df)
        end
        return Ω, $pullback
    end
end


BINARY_ACTS = [ # f, df1, df2
    (:elu, :(deriv_elu(x1, Ω, x2)), :(DoesNotExist())), # TODO use real deriv instead of DNE
    ]

for (f, df1, df2) in BINARY_ACTS
    @eval @scalar_rule($f(x1, x2), ($df1, $df2))

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(broadcasted),
                                        ::typeof($f), 
                                        x1::Numeric, x2::Numeric)
        Ω = $f.(x1, x2)
        function $pullback(Δ) 
            NO_FIELDS, NO_FIELDS, @.(Δ * $df1), @.(Δ * $df2)
        end
        return Ω, $pullback
    end
end
