if isdefined(Base, :get_extension)
    using CUDA.CUDNN: handle, with_workspace, cudnnTensorDescriptor, cudnnFilterDescriptor,
                      cudnnDataType, math_mode, CUDNN_DEFAULT_REORDER,
                      CUDNN_CROSS_CORRELATION, CUDNN_NOT_PROPAGATE_NAN, CUDNN_TENSOR_NCHW,
                      dim4

    using CUDA.CUDNN: cudnnPoolingMode_t, CUDNN_POOLING_MAX, 
                      CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, cudnnPoolingForward!,
                      pooldims, cudnnPoolingBackward
    import CUDA.CUDNN: cudnnPoolingDescriptor

    using CUDA.CUDNN: scalingParameter, CUDNN_CONVOLUTION, convdims,
                      cudnnConvolutionDescriptor, cudnnConvolutionBwdDataAlgoPerf,
                      cudnnConvolutionForward!, cudnnConvolutionBwdFilterAlgoPerf,
                      cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter,
                      cudnnConvolutionBackwardBias

    using CUDA.CUDNN: cudnnActivationForward!, cudnnOpTensor!, CUDNN_ACTIVATION_TANH,
                      CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_ELU,
                      CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_CLIPPED_RELU,
                      CUDNN_OP_TENSOR_MAX, CUDNN_ACTIVATION_IDENTITY

    using CUDA.CUDNN: CUDNN_BN_MIN_EPSILON, cudnnBatchNormalizationBackward,
                      cudnnBatchNormalizationForwardInference, CUDNN_BATCHNORM_SPATIAL,
                      cudnnBatchNormalizationForwardTraining

    using CUDA.CUDNN: CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, CUDNN_SOFTMAX_FAST,
                      CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxForward!, cudnnSoftmaxBackward
else
    using ..CUDA.CUDNN: handle, with_workspace, cudnnTensorDescriptor,
                        cudnnFilterDescriptor, cudnnDataType, math_mode,
                        CUDNN_DEFAULT_REORDER, CUDNN_CROSS_CORRELATION,
                        CUDNN_NOT_PROPAGATE_NAN, CUDNN_TENSOR_NCHW, dim4

    using ..CUDA.CUDNN: cudnnPoolingMode_t, CUDNN_POOLING_MAX, 
                        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, cudnnPoolingForward!,
                        pooldims, cudnnPoolingBackward
    import ..CUDA.CUDNN: cudnnPoolingDescriptor

    using ..CUDA.CUDNN: scalingParameter, CUDNN_CONVOLUTION, convdims,
                        cudnnConvolutionDescriptor, cudnnConvolutionBwdDataAlgoPerf,
                        cudnnConvolutionForward!, cudnnConvolutionBwdFilterAlgoPerf,
                        cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter,
                        cudnnConvolutionBackwardBias

    using ..CUDA.CUDNN: cudnnActivationForward!, cudnnOpTensor!, CUDNN_ACTIVATION_TANH,
                        CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_ELU,
                        CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_CLIPPED_RELU,
                        CUDNN_OP_TENSOR_MAX, CUDNN_ACTIVATION_IDENTITY

    using ..CUDA.CUDNN: CUDNN_BN_MIN_EPSILON, cudnnBatchNormalizationBackward,
                        cudnnBatchNormalizationForwardInference, CUDNN_BATCHNORM_SPATIAL,
                        cudnnBatchNormalizationForwardTraining

    using ..CUDA.CUDNN: CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL, CUDNN_SOFTMAX_FAST,
                        CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxForward!, cudnnSoftmaxBackward
end

cudnnversion() = CUDA.CUDNN.version()

function nnlibPadding(dims)
    pd = NNlib.padding(dims)
    if !all(pd[1:2:end] .== pd[2:2:end])
        @warn "cuDNN does not support asymmetric padding; defaulting to symmetric choice" maxlog=1
    end
    return pd[1:2:end]
end
