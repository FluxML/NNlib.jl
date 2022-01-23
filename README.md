# NNlib.jl

[![CI](https://github.com/FluxML/NNlib.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/FluxML/NNlib.jl/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/FluxML/NNlib.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/NNlib.jl) 
[![Documentation](https://img.shields.io/badge/docs-fluxml.ai-blue)](https://fluxml.ai/Flux.jl/stable/models/nnlib/)


This package provides a library of functions useful for neural networks, such as softmax, sigmoid, batched multiplication, convolutions and pooling. Many of these are used by [Flux.jl](https://github.com/FluxML/Flux.jl), which loads this package, but they may be used independently.

For use with automatic differentiation, this package defines gradients using [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl). These will be seen by various packages including [Zygote.jl](https://github.com/FluxML/Zygote.jl).

To use these functions with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) you will need [NNlibCUDA.jl](https://github.com/FluxML/NNlibCUDA.jl) as well.
