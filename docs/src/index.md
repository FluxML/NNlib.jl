# NNlib.jl

`NNlib` provides a library of functions useful for neural networks, such as softmax, sigmoid, batched multiplication, convolutions and pooling. Many of these are used by [Flux.jl](https://github.com/FluxML/Flux.jl), which loads this package, but they may be used independently.

For use with automatic differentiation, this package defines gradients using [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl). These will be seen by various packages including [Zygote.jl](https://github.com/FluxML/Zygote.jl).

GPU support is provided as package extensions. In order to load the extensions, use the imports
```julia
using NNlib, CUDA, cuDNN
```
for CUDA support, or
```julia
using NNlib, AMDGPU
```
for AMDGPU support.

## Threading

Various `NNlib` functions utilize available julia threads on divisible workloads. To disable this use
the `ScopedValue`-backed switch `NNlib.@disallow_spawns`
i.e.
```julia
NNlib.@disallow_spawns function_that_uses_nnlib()
```
