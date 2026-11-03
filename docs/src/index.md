# NNlib.jl

`NNlib` provides a library of functions useful for neural networks, such as softmax, sigmoid, batched multiplication, convolutions and pooling. Many of these are used by [Flux.jl](https://github.com/FluxML/Flux.jl), which loads this package, but they may be used independently.

For use with automatic differentiation (AD), this package defines gradients for its functions. AD packages are supported in two tiers:

- **First tier:** [Zygote.jl](https://github.com/FluxML/Zygote.jl) and [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl). Gradient tests check both. Zygote uses the rules defined with [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl), which other ChainRules-based packages also use. Enzyme's rules are in a package extension that `using Enzyme` loads.
- **Second tier, experimental:** [Mooncake.jl](https://github.com/chalk-lab/Mooncake.jl), loaded with `using Mooncake`. It has its own, smaller test suite. Expect missing rules and bugs, and please report them.

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
