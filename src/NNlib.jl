module NNlib

import Atomix
import ChainRulesCore: rrule

using Base.Broadcast: broadcasted
using Base.Threads
using ChainRulesCore
using GPUArraysCore
using KernelAbstractions
using KernelAbstractions: @atomic
using LinearAlgebra
using LinearAlgebra.BLAS: @blasfunc, BlasInt
using LinearAlgebra: AdjOrTransAbsMat, Adjoint, BlasFloat, Transpose
using Pkg
using Random
using Requires
using Statistics
using Statistics: mean

const libblas = Base.libblas_name

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

include("attention.jl")
export dot_product_attention, dot_product_attention_scores, make_causal_mask

include("dropout.jl")
export dropout, dropout!

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

include("bias_act.jl")
export bias_act!

include("fold.jl")

include("ctc.jl")
export ctc_loss

include("pooling.jl")
export maxpool, maxpool!, meanpool, meanpool!, lpnormpool, lpnormpool!,
    ∇maxpool, ∇maxpool!, ∇meanpool, ∇meanpool!, ∇lpnormpool, ∇lpnormpool!

include("padding.jl")
export pad_constant, pad_repeat, pad_reflect, pad_zeros, pad_symmetric, pad_circular

include("upsample.jl")
export upsample_nearest, ∇upsample_nearest,
    upsample_linear, ∇upsample_linear,
    upsample_bilinear, ∇upsample_bilinear,
    upsample_trilinear, ∇upsample_trilinear,
    pixel_shuffle

include("gather.jl")
include("scatter.jl")
include("utils.jl")
@init @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" begin
    using .ForwardDiff
    within_gradient(x::ForwardDiff.Dual) = true
    within_gradient(x::AbstractArray{<:ForwardDiff.Dual}) = true
end

include("sampling.jl")
include("functions.jl")

include("normalization.jl")
# export batchnorm, ∇batchnorm

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

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Enzyme="7da242da-08ed-463a-9acd-ee780be4f1d9" begin
            include("../ext/NNlibEnzymeExt.jl")
        end
    end
end

end # module NNlib
