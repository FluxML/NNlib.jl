using Test
using NNlib
using Zygote
using ForwardDiff: Dual
using Statistics: mean
using CUDA, cuDNN
using SparseArrays
import CUDA.CUSPARSE:
    AbstractCuSparseVector,
    CuSparseMatrixCSC,
    CuSparseMatrixCSR,
    CuSparseMatrixBSR,
    CuSparseMatrixCOO
using NNlib: batchnorm, ∇batchnorm
CUDA.allowscalar(false)

const AbstractCuSparseArray{Tv,Ti} = Union{
    AbstractCuSparseVector{Tv,Ti},
    CuSparseMatrixCSC{Tv,Ti},
    CuSparseMatrixCSR{Tv,Ti},
    CuSparseMatrixBSR{Tv,Ti},
    CuSparseMatrixCOO{Tv,Ti},
}

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
