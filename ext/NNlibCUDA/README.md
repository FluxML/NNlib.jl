# NNlibCUDA.jl

This is a glue package which extends functions from [NNlib.jl](https://github.com/FluxML/NNlib.jl) to work with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). It should be loaded automatically when using [Flux.jl](https://github.com/FluxML/Flux.jl), but not when using NNlib.jl by itself.

Julia gpu kernels are in `src/`, while wrappers around `cudnn` are in `src/cudnn/`.
