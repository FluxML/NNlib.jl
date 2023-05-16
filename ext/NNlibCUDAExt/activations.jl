# Activation functions

# Some of activation functions need a wrapper for GPU support
# https://github.com/JuliaGPU/CuArrays.jl/issues/614

# @cufunc softplus(x::Real) = ifelse(x > 0, x + log1p(exp(-x)), log1p(exp(x)))

# @cufunc logσ(x::Real) = -softplus(-x)

# @cufunc function gelu(x::Real)
#     p = oftype(x / 1, π)
#     λ = oftype(x / 1, √(2 / p))
#     α = oftype(x / 1, 0.044715)
#     h = oftype(x / 1, 0.5)
#     h * x * (one(x) + tanh(λ * (x + α * x^3)))
# end

# @cufunc lisht(x::Real) = x * tanh(x)

# @cufunc logcosh(x::Real) = x + softplus(-2x) - log(oftype(x, 2))

# @cufunc mish(x::Real) = x * tanh(softplus(x))

# @cufunc tanhshrink(x::Real) = x - tanh(x)
