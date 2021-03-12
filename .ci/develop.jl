import Pkg

root_directory = dirname(@__DIR__)

nnlib = Pkg.PackageSpec(path = root_directory)
Pkg.develop(nnlib)
Pkg.precompile()
