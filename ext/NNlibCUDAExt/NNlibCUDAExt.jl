module NNlibCUDAExt

using NNlib
using NNlib: BatchedAdjoint, BatchedTranspose, BatchedAdjOrTrans
using CUDA
using Random, Statistics
using Adapt
using Adapt: WrappedArray

import NNlib: ctc_loss, ctc_alpha, ∇ctc_loss

include("sampling.jl")
include("activations.jl")
include("batchedadjtrans.jl")
include("batchedmul.jl")
include("ctc.jl")
include("scatter.jl")
include("utils.jl")

# A block can declare 48 KiB of shared memory without opting in to more.
NNlib.attention_shmem_budget(::CUDABackend) = 48 * 1024

end # module
