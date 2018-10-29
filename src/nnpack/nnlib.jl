const AA{N} = AbstractArray{Float32,N}
const AA1 = Union{AA{2}, AA{3}, AA{4}, AA{5}}

#NOTE: Commenting out the activation functions until sure what to do

# relu(x::AA1) = nnp_relu_output(x, inplace ? x : similar(x), threadpool = shared_threadpool)

# leakyrelu(x::AA1, a = oftype(x/1, 0.01)) =
#     nnp_relu_output(x, inplace ? x : similar(x), negative_slope = a, threadpool = shared_threadpool)

softmax!(x::AbstractVecOrMat{Float64}) = Float64.(softmax!(Float32.(x)))

softmax!(x::AbstractVecOrMat{Float32}) =
    nnp_softmax_output(x, x, threadpool = shared_threadpool)

softmax!(y::AbstractVecOrMat{Float64}, x::AbstractVecOrMat{Float64}) = Float64.(softmax!(Float32.(y), Float32.(x)))

softmax!(y::AbstractVecOrMat{Float32}, x::AbstractVecOrMat{Float32}) =
    nnp_softmax_output(x, y, threadpool = shared_threadpool)

softmax(x::AbstractVecOrMat{Float64}) = Float64.(softmax(Float32.(x)))

softmax(x::AbstractVecOrMat{Float32}) =
    nnp_softmax_output(x, similar(x), threadpool = shared_threadpool)

maxpool(x::AbstractArray{Float64, 4}, k; pad = map(_->0,k), stride = k) =
    Float64.(maxpool(Float32.(x), k, pad = pad, stride = stride))

function maxpool(x::AA{4}, k; pad = map(_->0,k), stride = k)
    pad_, stride_ = expand(Val{length(k)}, pad), expand(Val{length(k)}, stride)
    ((size(x, 1) - k[1] + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - k[2] + 2 * pad_[2]) % stride_[2] == 0) || error("Choose the stride, pad and kernel size properly")
    maxpool!(similar(x, pdims(size(x), k, pad_, stride_)), x, k, pad = pad_, stride = stride_)
end

maxpool!(y::AbstractArray{Float64, 4}, x::AbstractArray{Float64, 4}, k; pad = map(_->0,k), stride = k) =
    Float64.(maxpool!(Float32.(y), Float32.(x), k, pad = pad, stride = stride))

maxpool!(y::AA{4}, x::AA{4}, k; pad = map(_->0,k), stride = k) =
    nnp_max_pooling_output(x, y, k, padding = expand(Val{length(k)}, pad), stride = expand(Val{length(k)}, stride), threadpool = shared_threadpool)

conv(x::AbstractArray{Float64, 4}, w::AbstractArray{Float64, 4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) =
    Float64.(conv(Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel))

function conv(x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0)
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ((size(x, 1) - size(w, 1) + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - size(w, 2) + 2 * pad_[2]) % stride_[2] == 0) || error("Choose the stride, pad and kernel size properly")
    y = similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_))
    b = zeros(Float32, size(y, 3))
    conv!(y, x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo), flipkernel = flipkernel)
end

conv(x::AbstractArray{Float64, 4}, w::AbstractArray{Float64, 4}, b::AbstractArray{Float64, 1}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) =
    Float64.(conv(Float32.(x), Float32.(w), Float32.(b), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel))

function conv(x::AA{4}, w::AA{4}, b::AA{1}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0)
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ((size(x, 1) - size(w, 1) + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - size(w, 2) + 2 * pad_[2]) % stride_[2] == 0) || error("Choose the stride, pad and kernel size properly")
    conv!(similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_)), x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo), flipkernel = flipkernel)
end

conv!(y::AbstractArray{Float64, 4}, x::AbstractArray{Float64, 4}, w::AbstractArray{Float64, 4}, b::AbstractArray{Float64, 1}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) =
    Float64.(conv(Float32.(y), Float32.(x), Float32.(w), Float32.(b), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel))

function conv!(y::AA{4}, x::AA{4}, w::AA{4}, b::AA{1}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0)
    flipkernel == 0 && (w = reverse(reverse(w, dims=1), dims=2))
    nnp_convolution_output(y, x, w, b, algo = algo, padding = pad, stride = stride, threadpool = shared_threadpool)
end

∇conv_data(dy::AbstractArray{Float64, 4}, x::AbstractArray{Float64, 4}, w::AbstractArray{Float64, 4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) =
    Float64.(∇conv_data(Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel))

function ∇conv_data(dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0)
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ((size(x, 1) - size(w, 1) + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - size(w, 2) + 2 * pad_[2]) % stride_[2] == 0) || error("Choose the stride, pad and kernel size properly")
    ∇conv_data!(zeros(Float32, size(x)), dy, x, w; pad = pad, stride = stride, dilation = dilation, algo = UInt32(algo), flipkernel = flipkernel)
end

∇conv_filter!(dx::AbstractArray{Float64, 4}, dy::AbstractArray{Float64, 4}, x::AbstractArray{Float64, 4}, w::AbstractArray{Float64, 4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) =
    Float64.(∇conv_filter!(Float32.(dx), Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel))

function ∇conv_data!(dx::AA{4}, dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0)
    flipkernel == 0 && (w = reverse(reverse(w, dims=1), dims=2))
    nnp_convolution_input_gradient(dx, x, dy, w, padding = pad, stride = stride, algo = algo, threadpool = shared_threadpool)
end

∇conv_filter(dy::AbstractArray{Float64, 4}, x::AbstractArray{Float64, 4}, w::AbstractArray{Float64, 4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) =
    Float64.(∇conv_filter(Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel))

function ∇conv_filter(dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0)
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ((size(x, 1) - size(w, 1) + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - size(w, 2) + 2 * pad_[2]) % stride_[2] == 0) || error("Choose the stride, pad and kernel size properly")
    ∇conv_filter!(zeros(Float32, size(w)), dy, x, w; pad = pad, stride = stride, dilation = dilation, algo = UInt32(algo), flipkernel = flipkernel)
end

∇conv_filter!(dw::AbstractArray{Float64, 4}, dy::AbstractArray{Float64, 4}, x::AbstractArray{Float64, 4}, w::AbstractArray{Float64, 4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) =
    Float64.(∇conv_filter!(Float32.(dw), Float32.(dy), Float32.(x), Float32.(w), pad = pad, stride = stride, dilation = dilation, algo = algo, flipkernel = flipkernel))

function ∇conv_filter!(dw::AA{4}, dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0)
    flipkernel == 0 && (w = reverse(reverse(w, dims=1), dims=2))
    nnp_convolution_kernel_gradient(dw, x, dy, w, padding = pad, stride = stride, algo = algo, threadpool = shared_threadpool)
end
