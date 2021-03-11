import Pkg
# import NNlibCUDA
pkgs = ["NNlibCUDA"]

# Do this for the time being since test doesn't pick up the manifest
# for some reason. Can remove this and manifests when cuda 3.0 is released. 
Pkg.add(url="https://github.com/JuliaGPU/CUDA.jl.git", rev="master")

Pkg.test(pkgs; coverage = true)
