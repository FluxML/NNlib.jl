# Activation functions

# `_exp`, `_expm1` and `_log1p_unit` are written for the CPU to vectorize. In a kernel that
# buys nothing, and libdevice's functions are faster.
CUDA.@device_override NNlib._exp(x::Float32) = exp(x)
CUDA.@device_override NNlib._exp(x::Float64) = exp(x)
CUDA.@device_override NNlib._expm1(x::Float32) = expm1(x)
CUDA.@device_override NNlib._expm1(x::Float64) = expm1(x)
CUDA.@device_override NNlib._log1p_unit(x::Float32) = log1p(x)
CUDA.@device_override NNlib._log1p_unit(x::Float64) = log1p(x)

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
