# NNlib + CUDA.jl

This is a glue package which extends functions from [NNlib.jl](https://github.com/FluxML/NNlib.jl) to work with [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl). It should be loaded automatically when NNlib and CUDA are loaded.

Wrappers around `cudnn` are in `src/cudnn/`, while Julia GPU kernels and everything else are at the top level.
