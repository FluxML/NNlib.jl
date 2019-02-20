flipweight(w::Array{<:Any,4}) = w[end:-1:1,end:-1:1,:,:]

softmax!(y::A, x::A) where A<:AbstractVecOrMat{Float32} = nnp_softmax_output(x, y)

nnpack_max_pooling!(y::A, x::A, k; pad = 0, stride = 1) where A<:Array{Float32, 4} =
    nnp_max_pooling_output(y, x, k, padding = pad, stride = stride)

function nnpack_convolution_forward!(y::A1, x::A1, w::A1, b::A2; pad = 0, stride = 1, algo = UInt32(0),
                                     flipkernel = 0) where {A1<:Array{Float32, 4}, A2<:Array{Float32, 1}}
    flipkernel == 0 && (w .= flipweight(w))        
    # Use nnp_convolution_inference if the batch size is 1.
    # The wrapper for nnp_convolution_inference is not present so use nnp_convolution_output for now
    nnp_convolution_output(y, x, w, b, algo = algo, padding = pad, stride = stride)
end

function nnpack_convolution_backward_data!(dx::A, dy::A, x::A, w::A; pad = 0, stride = 1,
                                           algo = UInt32(0), flipkernel = 0) where A<:Array{Float32, 4}
    flipkernel == 0 && (w .= flipweight(w))
    nnp_convolution_input_gradient(dx, x, dy, w, padding = pad, stride = stride, algo = algo)
end

function nnpack_convolution_backward_filter!(dw::A, dy::A, x::A, w::A; pad = 0, stride = 1,
                                             algo = UInt32(0), flipkernel = 0) where A<:Array{Float32, 4}
    flipkernel == 0 && (w .= flipweight(w))
    nnp_convolution_kernel_gradient(dw, x, dy, w, padding = pad, stride = stride, algo = algo)
    flipkernel && (dw .= flipkernel(dw))
    dw
end

