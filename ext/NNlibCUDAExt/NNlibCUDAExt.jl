module NNlibCUDAExt

using NNlib
using CUDA
import CUDA.CUSPARSE:
    AbstractCuSparseVector,
    CuSparseMatrixCSC,
    CuSparseMatrixCSR,
    CuSparseMatrixBSR,
    CuSparseMatrixCOO
using Random, Statistics

const AbstractCuSparseArray{Tv,Ti} = Union{
    AbstractCuSparseVector{Tv,Ti},
    CuSparseMatrixCSC{Tv,Ti},
    CuSparseMatrixCSR{Tv,Ti},
    CuSparseMatrixBSR{Tv,Ti},
    CuSparseMatrixCOO{Tv,Ti},
}

include("sampling.jl")
include("activations.jl")
include("batchedadjtrans.jl")
include("batchedmul.jl")
include("ctc.jl")
include("scatter.jl")
include("utils.jl")

end # module
