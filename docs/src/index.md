# NNlib.jl

`NNlib` provides a library of functions useful for neural networks, such as softmax, sigmoid, batched multiplication, convolutions and pooling. Many of these are used by [Flux.jl](https://github.com/FluxML/Flux.jl), which loads this package, but they may be used independently.

For use with automatic differentiation, this package defines gradients using [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl). These will be seen by various packages including [Zygote.jl](https://github.com/FluxML/Zygote.jl).

To use these functions with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) you will need to load it and NNlib in the same Julia session.
