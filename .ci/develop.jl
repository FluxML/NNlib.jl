import Pkg

root_directory = dirname(@__DIR__)

nnlib = Pkg.PackageSpec(path = root_directory)
Pkg.develop(nnlib)

if VERSION >= v"1.6"
    nnlibcuda = Pkg.PackageSpec(path = joinpath(root_directory, "lib", "NNlibCUDA"))
    Pkg.develop(nnlibcuda)
end

Pkg.precompile()
