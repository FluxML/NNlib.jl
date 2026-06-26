using Test
using NNlib
using Mooncake
using Mooncake.TestUtils: test_rule
using Random
using Zygote
using ForwardDiff: Dual
using Statistics: mean
using CUDA, cuDNN
import CUDA.CUSPARSE: CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO
using NNlib: batchnorm, ∇batchnorm
CUDA.allowscalar(false)

include("test_utils.jl")
include("activations.jl")
include("dropout.jl")
include("batchedadjtrans.jl")
include("batchedmul.jl")
include("conv.jl")
include("ctc.jl")
include("fold.jl")
include("pooling.jl")
include("softmax.jl")
include("batchnorm.jl")
include("scatter.jl")
include("gather.jl")
include("sampling.jl")
