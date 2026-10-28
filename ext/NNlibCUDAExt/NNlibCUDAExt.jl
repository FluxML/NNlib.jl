module NNlibCUDAExt

using NNlib
using NNlib: BatchedAdjoint, BatchedTranspose, BatchedAdjOrTrans
using CUDA
using BFloat16s: BFloat16
using Random, Statistics
import LinearAlgebra
using Adapt
using Adapt: WrappedArray

import NNlib: ctc_loss, ctc_alpha, ∇ctc_loss

include("sampling.jl")
include("activations.jl")
include("batchedadjtrans.jl")
include("batchedmul.jl")
include("groupedmul.jl")
include("ctc.jl")
include("scatter.jl")
include("utils.jl")

# A block can declare 48 KiB of shared memory without opting in to more. The kernel
# stays a little under, for the alignment of its arrays.
NNlib.attention_shmem_budget(::CUDABackend) = 47 * 1024

NNlib.attention_compute_units(::CUDABackend) =
    CUDA.attribute(CUDA.device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

include("precompile.jl")

end # module
