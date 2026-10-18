module NNlibFFTWExt

using FFTW
using NNlib
using KernelAbstractions
using ChainRulesCore: @non_differentiable

include("stft.jl")

end
