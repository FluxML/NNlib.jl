import Pkg

if VERSION >= v"1.5"
    pkgs = ["NNlib", "NNlibCUDA"]
else
    pkgs = ["NNlib"]
end

Pkg.test(pkgs; coverage = true)