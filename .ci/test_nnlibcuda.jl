import Pkg

pkgs = ["NNlibCUDA"]

Pkg.test(pkgs; coverage = true)
