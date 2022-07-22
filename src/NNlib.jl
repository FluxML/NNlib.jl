module NNlib

using Pkg
using Requires
using ChainRulesCore
import ChainRulesCore: rrule
using Base.Broadcast: broadcasted
using Base.Threads
using Statistics
using Statistics: mean
using LinearAlgebra
using LinearAlgebra: BlasFloat, Transpose, Adjoint, AdjOrTransAbsMat
using LinearAlgebra.BLAS: BlasInt, @blasfunc

const libblas = Base.libblas_name

const IntOrIntTuple = Union{Integer, NTuple{N,<:Integer} where N}
const Numeric = Union{AbstractArray{<:T}, T} where {T<:Number}

# Include APIs
include("dim_helpers.jl")
export ConvDims, DenseConvDims, PoolDims, DepthwiseConvDims

is_nnpack_available() = false

@init @require NNPACK_jll="a6bfbf70-4841-5cb9-aa18-3a8ad3c413ee"  begin
  if isdefined(NNPACK_jll, :libnnpack)
    include("nnpack/NNPACK.jl")
  else
    @warn "NNPACK not available for your platform: " *
          "$( Pkg.BinaryPlatforms.platform_name(Pkg.BinaryPlatforms.platform_key_abi()))" *
          "($( Pkg.BinaryPlatforms.triplet(Pkg.BinaryPlatforms.platform_key_abi())))
          You will be able to use only the default Julia NNlib backend"
  end
end

include("activations.jl")
for f in ACTIVATIONS
    @eval export $(f)
end
export sigmoid, hardsigmoid, logsigmoid, thresholdrelu # Aliases

include("softmax.jl")
export softmax, softmax!, ∇softmax, ∇softmax!, logsoftmax, 
    logsoftmax!, ∇logsoftmax, ∇logsoftmax!, logsumexp

include("batched/batchedadjtrans.jl")
include("batched/batchedmul.jl")
export batched_mul, batched_mul!, ⊠,  batched_vec,
    batched_transpose, batched_adjoint

include("gemm.jl")
export grid_sample, ∇grid_sample

include("conv.jl")
export conv, conv!, ∇conv_data, ∇conv_data!, ∇conv_filter, 
    ∇conv_filter!, depthwiseconv, depthwiseconv!, 
    ∇depthwiseconv_data, ∇depthwiseconv_data!, 
    ∇depthwiseconv_filter, ∇depthwiseconv_filter!

include("conv_bias_act.jl")
export conv_bias_act, conv_bias_act!

include("ctc.jl")
export ctc_loss

include("pooling.jl")
export maxpool, maxpool!, meanpool, meanpool!, 
    ∇maxpool, ∇maxpool!, ∇meanpool, ∇meanpool!

include("padding.jl")
export pad_constant, pad_repeat, pad_reflect, pad_zeros

include("upsample.jl")
export upsample_nearest, ∇upsample_nearest,
    upsample_linear, ∇upsample_linear,
    upsample_bilinear, ∇upsample_bilinear,
    upsample_trilinear, ∇upsample_trilinear,
    pixel_shuffle

include("gather.jl")
include("scatter.jl")
include("utils.jl")
include("sampling.jl")
include("functions.jl")

## Include implementations
include("impl/padding_edges.jl")

# Direct implementations of convolutional and depthwise-convolutional algorithms
include("impl/conv_direct.jl")
include("impl/depthwiseconv_direct.jl")
# im2col implementations of convolutional and depthwise-convolutional algorithms
include("impl/conv_im2col.jl")
include("impl/depthwiseconv_im2col.jl")

# Direct implementations of pooling
include("impl/pooling_direct.jl")
include("deprecations.jl")

end # module NNlib
