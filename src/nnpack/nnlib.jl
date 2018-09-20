const AA{N} = AbstractArray{Float32,N}
const AA1 = Union{AA{2}, AA{3}, AA{4}, AA{5}}

relu(x::AA1; inplace::Bool = true, nthreads = NNPACK_CPU_THREADS) =
    nnp_relu_output(x, inplace ? x : similar(x), threadpool = pthreadpool_create(nthreads))

leakyrelu(x::AA1, a = oftype(x/1, 0.01); inplace::Bool = true, nthreads = NNPACK_CPU_THREADS) =
    nnp_relu_output(x, inplace ? x : similar(x), negative_slope = a, threadpool = pthreadpool_create(nthreads))

# NOTE: softmax is slower than the nnlib version. So might comment it out

softmax!(x::AbstractVecOrMat{Float32}; inplace::Bool = true, nthreads = NNPACK_CPU_THREADS) =
    nnp_softmax_output(x, inplace ? x : similar(x), threadpool = pthreadpool_create(nthreads))

softmax!(y::AbstractVecOrMat{Float32},x::AbstractVecOrMat{Float32}; inplace::Bool = true, nthreads = NNPACK_CPU_THREADS) =
    nnp_softmax_output(x, y, threadpool = pthreadpool_create(nthreads))

softmax(x::AbstractVecOrMat{Float32}; nthreads = NNPACK_CPU_THREADS) =
    nnp_softmax_output(x, similar(x), threadpool = pthreadpool_create(nthreads))

maxpool(x::AA{4}, k; pad = map(_->0,k), stride = k, nthreads = NNPACK_CPU_THREADS) =
    maxpool!(similar(x, pdims(size(x), k, expand(Val{length(k)}, pad), expand(Val{length(k)}, stride))), x, k, pad = expand(Val{length(k)}, pad), stride = expand(Val{length(k)}, stride), threadpool = pthreadpool_create(nthreads))

maxpool!(y::AA{4}, x::AA{4}, k; pad = map(_->0,k), stride = k, threadpool = nothing) =
    nnp_max_pooling_output(x, y, k, padding = expand(Val{length(k)}, pad), stride = expand(Val{length(k)}, stride), threadpool = threadpool)

function conv(x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = 0, nthreads = NNPACK_CPU_THREADS)
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    y = similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_))
    b = zeros(Float32, size(y, 3))
    conv!(y, x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo), threadpool = pthreadpool_create(nthreads))
end

function conv(x::AA{4}, w::AA{4}, b::AA{1}; pad = 0, stride = 1, dilation = 1, algo = 0, nthreads = NNPACK_CPU_THREADS)
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    conv!(similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_)), x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo), threadpool = pthreadpool_create(nthreads))
end

conv!(y::AA{4}, x::AA{4}, w::AA{4}, b::AA{1}; pad = 0, stride = 1, dilation = 1, algo = 0, threadpool = nothing) =
    nnp_convolution_output(y, x, w, b, algo = algo, padding = pad, stride = stride, threadpool = threadpool)

function ∇conv_data(dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = 0, nthreads = NNPACK_CPU_THREADS)
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ∇conv_data!(zeros(Float32, size(x)), dy, x, w; pad = pad, stride = stride, dilation = dilation, algo = UInt32(algo), threadpool = pthreadpool_create(nthreads))
end

∇conv_data!(dx::AA{4}, dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = 0, threadpool = nothing) =
    nnp_convolution_input_gradient(dx, x, dy, w, padding = pad, stride = stride, algo = algo, threadpool = threadpool)

function ∇conv_filter(dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = 0, nthreads = NNPACK_CPU_THREADS)
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ∇conv_filter!(zeros(Float32, size(w)), dy, x, w; pad = pad, stride = stride, dilation = dilation, algo = UInt32(algo), threadpool = pthreadpool_create(nthreads))
end

∇conv_filter!(dw::AA{4}, dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = 0, threadpool = nothing) =
    nnp_convolution_kernel_gradient(dw, x, dy, w, padding = pad, stride = stride, algo = algo, threadpool = threadpool)
