import Pkg

root_directory = dirname(@__DIR__)

nnlib = Pkg.PackageSpec(path = root_directory)
nnlibcuda = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "NNlibCUDA"))

Pkg.develop(nnlib)
Pkg.develop(nnlibcuda)

## Do this for the time being since test doesn't pick up the manifest
## for some reason. Can remove this and manifests when cuda 3.0 is released. 
# Pkg.add(url="https://github.com/JuliaGPU/GPUCompiler.jl.git", rev="master")
# Pkg.add(url="https://github.com/JuliaGPU/GPUArrays.jl.git", rev="master")
# Pkg.add(url="https://github.com/maleadt/LLVM.jl.git", rev="master")
# Pkg.add(url="https://github.com/JuliaGPU/CUDA.jl.git", rev="master")
Pkg.add(url="https://github.com/FluxML/Zygote.jl.git", rev="cl/cufunc")

Pkg.precompile()
