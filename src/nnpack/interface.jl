flipweight(w::Array{<:Any,4}) = w[end:-1:1,end:-1:1,:,:]

function check_support(x, k, pad, stride, dilation = 1)
    fallback = false
    dilation == 1 || dilation == (1, 1) || (fallback = true)
    pad_, stride_ = expand(Val{length(k)}, pad), expand(Val{length(k)}, stride)
    ((size(x, 1) - k[1] + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - k[2] + 2 * pad_[2]) % stride_[2] == 0) || (fallback = true)
    return pad_, stride_, fallback
end

softmax!(y::A, x::A) where A<:AbstractVecOrMat{Float32} =
    nnp_softmax_output(x, y)

function maxpool!(y::A, x::A, k; pad = map(_->0,k), stride = k) where A<:Array{Float32, 4}
    pad_, stride_, fallback = check_support(x, k, pad, stride)
    if fallback
        maxpool_cpu!(y, x, k, pad = pad_, stride = stride_)
    else
        nnp_max_pooling_output(x, y, k, padding = expand(Val{length(k)}, pad), stride = expand(Val{length(k)}, stride))
    end
end

function conv!(y::A1, x::A1, w::A1; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A1<:Array{Float32, 4}
    pad_, stride_, fallback = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    flipkernel == 0 && (w .= flipweight(w))
    if fallback
        conv2d!(y, x, w, padding = pad, stride = stride, dilation = dilation, mode = 1)
    else
        nnp_convolution_output(y, x, w, zeros(Float32, size(y, 3)), algo = algo, padding = pad, stride = stride)
    end
end

function ∇conv_data!(dx::A, dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:Array{Float32, 4}
    pad_, stride_, fallback = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    if fallback
        conv2d_grad_x!(dx, x, w, dy, padding = pad_, stride = stride_, dilation = dilation)
    else
        flipkernel == 0 && (w .= flipweight(w))
        nnp_convolution_input_gradient(dx, x, dy, w, padding = pad, stride = stride, algo = algo)
    end
end

function ∇conv_filter!(dw::A, dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:Array{Float32, 4}
    pad_, stride_, fallback = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    if fallback
        conv2d_grad_w!(dw, x, w, dy, padding = pad_, stride = stride_, dilation = dilation)
    else
        flipkernel == 0 && (w .= flipweight(w))
        nnp_convolution_kernel_gradient(dw, x, dy, w, padding = pad, stride = stride, algo = algo)
        flipkernel && (dw .= flipkernel(dw))
        dw
    end
end
