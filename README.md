# NNlib

[![Build Status](https://travis-ci.org/FluxML/NNlib.jl.svg?branch=master)](https://travis-ci.org/FluxML/NNlib.jl) [![Build status](https://ci.appveyor.com/api/projects/status/wo2wkv1l9cj548uh?svg=true)](https://ci.appveyor.com/project/one-more-minute/nnlib-jl)

This package will provide a library of functions useful for ML, such as softmax, sigmoid, convolutions and pooling. It doesn't provide any other "high-level" functionality like layers or AD.

Other packages can build on these functions as if they were defined in Base Julia; for example, CuArrays provides GPU kernels, and Flux provides automatic differentiation; both can work together without explicitly being aware of each other.
