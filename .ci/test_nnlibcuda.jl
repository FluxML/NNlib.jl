import Pkg

pkgs = ["NNlib", "NNlibCUDA"]

Pkg.test(pkgs; coverage = true)
