<img align="right" width="200px" src="https://github.com/FluxML/NNlib.jl/raw/master/docs/src/assets/logo.png">

# NNlib.jl

[![Documentation][docs-dev-img]][docs-dev-url]
[![CI](https://github.com/FluxML/NNlib.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/FluxML/NNlib.jl/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/FluxML/NNlib.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/NNlib.jl) 

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://fluxml.ai/NNlib.jl/stable/

[docs-dev-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-dev-url]: https://fluxml.ai/NNlib.jl/dev/

This package provides a library of functions useful for neural networks, such as softmax, sigmoid, batched multiplication, convolutions and pooling. Many of these are used by [Flux.jl](https://github.com/FluxML/Flux.jl), which loads this package, but they may be used independently.

For use with automatic differentiation, this package defines gradients using [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl). These will be seen by various packages including [Zygote.jl](https://github.com/FluxML/Zygote.jl).

GPU support is provided as package extensions (see the `ext/` folder). In order to load the extensions, use the imports
```julia
using NNlib, CUDA, cuDNN
```
for CUDA support, or
```julia
using NNlib, AMDGPU
```
for AMDGPU support.
