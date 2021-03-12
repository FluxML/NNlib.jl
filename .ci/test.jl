import Pkg

pkgs = ["NNlib"]

Pkg.test(pkgs; coverage = true)
