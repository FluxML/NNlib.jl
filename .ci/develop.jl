import Pkg

root_directory = dirname(@__DIR__)

nnlib = Pkg.PackageSpec(path = root_directory)
nnlibcuda = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "NNlibCUDA"))

Pkg.develop(nnlib)
Pkg.develop(nnlibcuda)

Pkg.precompile()
