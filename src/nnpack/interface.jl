function check_support(x, k, pad, stride, dilation = 1)
    fallback = false
    dilation == 1 || dilation == (1, 1) || (fallback = true)
    pad_, stride_ = expand(Val{length(k)}, pad), expand(Val{length(k)}, stride)
    ((size(x, 1) - k[1] + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - k[2] + 2 * pad_[2]) % stride_[2] == 0) || (fallback = true)
    return pad_, stride_, fallback
end

#NOTE: Commenting out the activation functions until sure what to do

# relu(x::AA1) = nnp_relu_output(x, inplace ? x : similar(x), threadpool = shared_threadpool[])

# leakyrelu(x::AA1, a = oftype(x/1, 0.01)) =
#     nnp_relu_output(x, inplace ? x : similar(x), negative_slope = a, threadpool = shared_threadpool[])

softmax!(x::A) where A<:AbstractVecOrMat{Float64} = softmax!(Float32.(x))

softmax!(x::A) where A<:AbstractVecOrMat{Float32} =
    nnp_softmax_output(x, x, threadpool = shared_threadpool[])

softmax!(y::A, x::A) where A<:AbstractVecOrMat{Float64} = softmax!(Float32.(y), Float32.(x))

softmax!(y::A, x::A) where A<:AbstractVecOrMat{Float32} =
    nnp_softmax_output(x, y, threadpool = shared_threadpool[])

softmax(x::A) where A<:AbstractVecOrMat{Float64} = softmax(Float32.(x))

softmax(x::A) where A<:AbstractVecOrMat{Float32} =
    nnp_softmax_output(x, similar(x), threadpool = shared_threadpool[])

maxpool(x::A, k; pad = map(_->0,k), stride = k) where A<:AbstractArray{Float64, 4} =
    maxpool(Float32.(x), k, pad = pad, stride = stride)

function maxpool(x::A, k; pad = map(_->0,k), stride = k) where A<:AbstractArray{Float32, 4}
    pad_, stride_, fallback = check_support(x, k, pad, stride)
    if fallback
        maxpool_cpu!(similar(x, pdims(size(x), k, pad_, stride_)), x, k, pad = pad_, stride = stride_)
    else
        maxpool!(similar(x, pdims(size(x), k, pad_, stride_)), x, k, pad = pad_, stride = stride_)
    end
end

maxpool!(y::A, x::A, k; pad = map(_->0,k), stride = k) where A<:AbstractArray{Float64, 4} =
    maxpool!(Float32.(y), Float32.(x), k, pad = pad, stride = stride)

maxpool!(y::A, x::A, k; pad = map(_->0,k), stride = k) where A<:AbstractArray{Float32, 4} =
    nnp_max_pooling_output(x, y, k, padding = expand(Val{length(k)}, pad), stride = expand(Val{length(k)}, stride), threadpool = shared_threadpool[])

conv(x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where A<:AbstractArray{Float64, 4} =
    conv(Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo)

function conv(x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where A<:AbstractArray{Float32, 4}
    pad_, stride_, fallback = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    y = similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_))
    if fallback
        conv2d!(y, x, w, padding = pad, stride = stride, dilation = dilation)
    else
        conv!(y, x, w, zeros(Float32, size(y, 3)), pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo))
    end
end

conv(x::A1, w::A1, b::A2; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A1<:AbstractArray{Float64, 4}, A2<:AbstractArray{Float64, 1}} =
    conv(Float32.(x), Float32.(w), Float32.(b), pad = pad, stride = stride, dilation = dilation, algo = algo)

function conv(x::A1, w::A1, b::A2; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A1<:AbstractArray{Float32, 4}, A2<:AbstractArray{Float32, 1}}
    pad_, stride_, fallback = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    y = similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_))
    if fallback
        conv2d!(y, x, w, padding = pad, stride = stride, dilation = dilation)
    else
        conv!(y, x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo))
    end
end

crosscor(x::A1, w::A1, b::A2; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A1<:AbstractArray{Float64, 4}, A2<:AbstractArray{Float64, 1}} =
    crosscor(Float32.(x), Float32.(w), Float32.(b), pad = pad, stride = stride, dilation = dilation, algo = algo)

function crosscor(x::A1, w::A1, b::A2; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A1<:AbstractArray{Float32, 4}, A2<:AbstractArray{Float32, 1}}
    pad_, stride_, fallback = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    y = similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_))
    if fallback
        conv2d!(y, x, w, padding = pad, stride = stride, dilation = dilation, mode = 1)
    else
        conv!(y, x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo), flipkernel = 1)
    end
end

conv!(y::A1, x::A1, w::A1, b::A2; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where {A1<:AbstractArray{Float64, 4}, A2<:AbstractArray{Float64, 1}} =
    conv!(Float32.(y), Float32.(x), Float32.(w), Float32.(b), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel)

function conv!(y::A1, x::A1, w::A1, b::A2; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where {A1<:AbstractArray{Float32, 4}, A2<:AbstractArray{Float32, 1}}
    if flipkernel == 0
        w = reverse(reverse(w, dims=1), dims=2)
    end
    nnp_convolution_output(y, x, w, b, algo = algo, padding = pad, stride = stride, threadpool = shared_threadpool[])
end

crosscor!(y::A1, x::A1, w::A1, b::A2; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A1<:AbstractArray{Float64, 4}, A2<:AbstractArray{Float64, 1}} =
    conv!(Float32.(y), Float32.(x), Float32.(w), Float32.(b), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = 1)

crosscor!(y::A1, x::A1, w::A1, b::A2; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where {A1<:AbstractArray{Float32, 4}, A2<:AbstractArray{Float32, 1}} =
    conv!(y, x, w, b, pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = 1)

∇conv_data(dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where A<:AbstractArray{Float64, 4} =
    ∇conv_data(Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo)

function ∇conv_data(dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where A<:AbstractArray{Float32, 4}
    pad_, stride_, fallback = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    if fallback
        conv2d_grad_x!(zeros(Float32, size(x)), x, w, dy, padding = pad_, stride = stride_, dilation = dilation)
    else  
        ∇conv_data!(zeros(Float32, size(x)), dy, x, w; pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo))
    end
end

∇conv_data!(dx::A, dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:AbstractArray{Float64, 4} =
    ∇conv_data!(Float32.(dx), Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel)

function ∇conv_data!(dx::A, dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:AbstractArray{Float32, 4}
    flipkernel == 0 && (w = reverse(reverse(w, dims=1), dims=2))
    nnp_convolution_input_gradient(dx, x, dy, w, padding = pad, stride = stride, algo = algo, threadpool = shared_threadpool[])
end

∇conv_filter(dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where A<:AbstractArray{Float64, 4} =
    ∇conv_filter(Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo)

function ∇conv_filter(dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0)) where A<:AbstractArray{Float32, 4}
    pad_, stride_, fallback = check_support(x, (size(w, 1), size(w, 2)), pad, stride, dilation)
    if fallback
        conv2d_grad_w!(zeros(Float32, size(w)), x, w, dy, padding = pad_, stride = stride_, dilation = dilation)
    else 
        ∇conv_filter!(zeros(Float32, size(w)), dy, x, w; pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo))
    end
end

∇conv_filter!(dw::A, dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:AbstractArray{Float64, 4} =
    ∇conv_filter!(Float32.(dw), Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel)

function ∇conv_filter!(dw::A, dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:AbstractArray{Float32, 4}
    flipkernel == 0 && (w = reverse(reverse(w, dims=1), dims=2))
    dw .= nnp_convolution_kernel_gradient(dw, x, dy, w, padding = pad, stride = stride, algo = algo, threadpool = shared_threadpool[])
    flipkernel == 0 ? reverse(reverse(dw, dims=1), dims=2) : dw
end
