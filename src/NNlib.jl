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
using Random
using ScopedValues
using Statistics
using Statistics: mean

const Numeric = Union{AbstractArray{<:T}, T} where {T<:Number}

# internal. TODO: change to an approach where amount of threading is controlled, not just on/off
const ALLOW_SPAWNS = ScopedValue(true)
should_use_spawn() = Threads.nthreads(:default) > 1 && ALLOW_SPAWNS[]
"""
    @disallow_spawns ex

Disallow NNlib to use `@spawn` on divisible workloads. i.e. within `conv` etc.
"""
macro disallow_spawns(ex)
    quote
        @with ALLOW_SPAWNS => false $(esc(ex))
    end
end

# Include APIs
include("dim_helpers.jl")
export ConvDims, DenseConvDims, PoolDims, DepthwiseConvDims

include("activations.jl")
for f in ACTIVATIONS
    @eval export $(f)
end
export sigmoid, hardsigmoid, logsigmoid, thresholdrelu, gelu # Aliases

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

include("rotation.jl")
export imrotate, ∇imrotate

include("audio/stft.jl")
include("audio/spectrogram.jl")
include("audio/mel.jl")
export stft, istft, hann_window, hamming_window, spectrogram, melscale_filterbanks

end # module NNlib
