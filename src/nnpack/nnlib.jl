const AA{N} = AbstractArray{Float32,N}
const AA1 = Union{AA{2}, AA{3}, AA{4}, AA{5}}

#NOTE: Commenting out the activation functions until sure what to do

# function relu(x::AA1; inplace::Bool = true, nthreads = NNPACK_CPU_THREADS)
#     threadpool = nthreads != NNPACK_CPU_THREADS ? pthreadpool_create(nthreads) : shared_threadpool
#     nnp_relu_output(x, inplace ? x : similar(x), threadpool = threadpool)
# end
#
# function leakyrelu(x::AA1, a = oftype(x/1, 0.01); inplace::Bool = true, nthreads = NNPACK_CPU_THREADS)
#     threadpool = nthreads != NNPACK_CPU_THREADS ? pthreadpool_create(nthreads) : shared_threadpool
#     nnp_relu_output(x, inplace ? x : similar(x), negative_slope = a, threadpool = threadpool)
# end

function softmax!(x::AbstractVecOrMat{Float32}; inplace::Bool = true, nthreads = NNPACK_CPU_THREADS)
    threadpool = nthreads != NNPACK_CPU_THREADS ? pthreadpool_create(nthreads) : shared_threadpool
    nnp_softmax_output(x, inplace ? x : similar(x), threadpool = threadpool)
end

function softmax!(y::AbstractVecOrMat{Float32},x::AbstractVecOrMat{Float32}; inplace::Bool = true, nthreads = NNPACK_CPU_THREADS)
    threadpool = nthreads != NNPACK_CPU_THREADS ? pthreadpool_create(nthreads) : shared_threadpool
    nnp_softmax_output(x, y, threadpool = threadpool)
end

function softmax(x::AbstractVecOrMat{Float32}; nthreads = NNPACK_CPU_THREADS)
    threadpool = nthreads != NNPACK_CPU_THREADS ? pthreadpool_create(nthreads) : shared_threadpool
    nnp_softmax_output(x, similar(x), threadpool = threadpool)
end

function maxpool(x::AA{4}, k; pad = map(_->0,k), stride = k, nthreads = NNPACK_CPU_THREADS)
    threadpool = nthreads != NNPACK_CPU_THREADS ? pthreadpool_create(nthreads) : shared_threadpool
    pad_, stride_ = expand(Val{length(k)}, pad), expand(Val{length(k)}, stride)
    ((size(x, 1) - k[1] + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - k[2] + 2 * pad_[2]) % stride_[2] == 0) || error("Choose the stride, pad and kernel size properly")
    maxpool!(similar(x, pdims(size(x), k, pad_, stride_)), x, k, pad = pad_, stride = stride_, threadpool = threadpool)
end

maxpool!(y::AA{4}, x::AA{4}, k; pad = map(_->0,k), stride = k, threadpool = nothing) =
    nnp_max_pooling_output(x, y, k, padding = expand(Val{length(k)}, pad), stride = expand(Val{length(k)}, stride), threadpool = threadpool)

function conv(x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), nthreads = NNPACK_CPU_THREADS)
    threadpool = nthreads != NNPACK_CPU_THREADS ? pthreadpool_create(nthreads) : shared_threadpool
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ((size(x, 1) - size(w, 1) + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - size(w, 2) + 2 * pad_[2]) % stride_[2] == 0) || error("Choose the stride, pad and kernel size properly")
    w = reverse(reverse(w, dims=1), dims=2) # Needs to be fixed once the flipkernel and crosscor PR is merged
    y = similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_))
    b = zeros(Float32, size(y, 3))
    conv!(y, x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo), threadpool = threadpool)
end

function conv(x::AA{4}, w::AA{4}, b::AA{1}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), nthreads = NNPACK_CPU_THREADS)
    threadpool = nthreads != NNPACK_CPU_THREADS ? pthreadpool_create(nthreads) : shared_threadpool
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ((size(x, 1) - size(w, 1) + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - size(w, 2) + 2 * pad_[2]) % stride_[2] == 0) || error("Choose the stride, pad and kernel size properly")
    w = reverse(reverse(w, dims=1), dims=2) # Needs to be fixed once the flipkernel and crosscor PR is merged
    conv!(similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_)), x, w, b, pad = pad_, stride = stride_, dilation = dilation, algo = UInt32(algo), threadpool = threadpool)
end

conv!(y::AA{4}, x::AA{4}, w::AA{4}, b::AA{1}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), threadpool = nothing) =
    nnp_convolution_output(y, x, w, b, algo = algo, padding = pad, stride = stride, threadpool = threadpool)

function ∇conv_data(dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), nthreads = NNPACK_CPU_THREADS)
    threadpool = nthreads != NNPACK_CPU_THREADS ? pthreadpool_create(nthreads) : shared_threadpool
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ((size(x, 1) - size(w, 1) + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - size(w, 2) + 2 * pad_[2]) % stride_[2] == 0) || error("Choose the stride, pad and kernel size properly")
    w = reverse(reverse(w, dims=1), dims=2) # Needs to be fixed once the flipkernel and crosscor PR is merged
    ∇conv_data!(zeros(Float32, size(x)), dy, x, w; pad = pad, stride = stride, dilation = dilation, algo = UInt32(algo), threadpool = threadpool)
end

∇conv_data!(dx::AA{4}, dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), threadpool = nothing) =
    nnp_convolution_input_gradient(dx, x, dy, w, padding = pad, stride = stride, algo = algo, threadpool = threadpool)

function ∇conv_filter(dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), nthreads = NNPACK_CPU_THREADS)
    threadpool = nthreads != NNPACK_CPU_THREADS ? pthreadpool_create(nthreads) : shared_threadpool
    dilation == 1 || dilation == (1, 1) || error("NNPACK does not support dilation > 1")
    pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
    ((size(x, 1) - size(w, 1) + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - size(w, 2) + 2 * pad_[2]) % stride_[2] == 0) || error("Choose the stride, pad and kernel size properly")
    w = reverse(reverse(w, dims=1), dims=2) # Needs to be fixed once the flipkernel and crosscor PR is merged
    ∇conv_filter!(zeros(Float32, size(w)), dy, x, w; pad = pad, stride = stride, dilation = dilation, algo = UInt32(algo), threadpool = threadpool)
end

∇conv_filter!(dw::AA{4}, dy::AA{4}, x::AA{4}, w::AA{4}; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), threadpool = nothing) =
    nnp_convolution_kernel_gradient(dw, x, dy, w, padding = pad, stride = stride, algo = algo, threadpool = threadpool)
