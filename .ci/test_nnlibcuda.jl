import Pkg
# import NNlibCUDA
pkgs = ["NNlibCUDA"]


Pkg.test(pkgs; coverage = true)
