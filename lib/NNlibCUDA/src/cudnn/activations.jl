
# Activation

using Base.Broadcast
using CUDA.CUDNN: cudnnActivationForward!, cudnnOpTensor!,
            CUDNN_ACTIVATION_TANH,CUDNN_ACTIVATION_SIGMOID,CUDNN_ACTIVATION_ELU,
            CUDNN_ACTIVATION_RELU,CUDNN_ACTIVATION_CLIPPED_RELU,CUDNN_OP_TENSOR_MAX

for (f, op) in [
    CUDA.tanh       => (src,dst)->cudnnActivationForward!(dst, src, mode=CUDNN_ACTIVATION_TANH),
    NNlib.σ         => (src,dst)->cudnnActivationForward!(dst, src, mode=CUDNN_ACTIVATION_SIGMOID),
    NNlib.elu       => (src,dst)->cudnnActivationForward!(dst, src, mode=CUDNN_ACTIVATION_ELU),
    NNlib.relu      => (src,dst)->cudnnActivationForward!(dst, src, mode=CUDNN_ACTIVATION_RELU),
    # NNlib.relu6     => (src,dst)->cudnnActivationForward!(dst, src, mode=CUDNN_ACTIVATION_CLIPPED_RELU, coef=6.0),
    # NNlib.leakyrelu => (src,dst)->cudnnOpTensor!(dst, src, src; op=CUDNN_OP_TENSOR_MAX, alpha1=0.01),
    ]
    
    @eval begin
        # in-place
        function Base.materialize!(dst::DenseCuArray{<:CUDNNFloat},
                                   bc::Broadcast.Broadcasted{<:Any,<:Any,typeof($f),<:Tuple{DenseCuArray}})
            $op(bc.args[1], dst)
            return dst
        end

        # out of place
        function Base.materialize(bc::Broadcast.Broadcasted{<:Any,<:Any,typeof($f),<:Tuple{DenseCuArray}})
            ElType = Broadcast.combine_eltypes(bc.f, bc.args)
            dst = similar(bc, ElType)
            $op(bc.args[1], dst)
            return dst
        end
    end
end

# CUDNN_ACTIVATION_IDENTITY does not work with cudnnActivationForward
# FIXME: put this optimization in GPUArrays' `copyto!` (like Base.Broadcast's `copyto!`)
Base.broadcasted(::typeof(identity), x::DenseCuArray{T}) where {T<:CUDNNFloat} = x

