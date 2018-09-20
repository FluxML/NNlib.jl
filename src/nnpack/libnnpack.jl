function nnp_initialize()
    ccall((:nnp_initialize, libnnpack), nnp_status, (),)
end

function nnp_deinitialize()
    ccall((:nnp_deinitialize, libnnpack), nnp_status, (),)
end

function pthreadpool_create(n = 0)
    ccall((:pthreadpool_create, libnnpack), Ptr{Cvoid}, (Csize_t,), n)
end

function nnp_relu_output(batch_size, channels, input, output, negative_slope, threadpool)
    @check ccall((:nnp_relu_output, libnnpack), nnp_status, (Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Cfloat, pthreadpool_t), batch_size, channels, input, output, negative_slope, threadpool)
end

function nnp_relu_output(x::AbstractArray{Float32,N}, y::AbstractArray{Float32,N}; negative_slope::AbstractFloat = 0.0, threadpool = nothing) where {N}
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    # Investigate why the channel and batch dims need to specified like this
    nnp_relu_output(prod(size(x)[N-1:N]), prod(size(x)[1:N-2]), x, y, negative_slope, threadpool)
    y
end

function nnp_relu_input_gradient(batch_size, channels, grad_output, input, grad_input, negative_slope, threadpool)
    @check ccall((:nnp_relu_input_gradient, libnnpack), nnp_status, (Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Cfloat, pthreadpool_t), batch_size, channels, grad_output, input, grad_input, negative_slope, threadpool)
end

function nnp_relu_input_gradient(x::AbstractArray{Float32,N}, dy::AbstractArray{Float32,N}, dx::AbstractArray{Float32,N}; negative_slope::AbstractFloat = 0.0, threadpool = nothing) where {N}
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    # Investigate why the channel and batch dims need to specified like this
    nnp_relu_input_gradient(Csize_t(prod(size(x)[N-1:N])), prod(size(x)[1:N-2]), dy, x, dx, negative_slope, threadpool)
    dx
end

function nnp_softmax_output(batch_size, channels, input, output, threadpool)
    @check ccall((:nnp_softmax_output, libnnpack), nnp_status, (Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, pthreadpool_t), batch_size, channels, input, output, threadpool)
end

function nnp_softmax_output(x::AbstractVecOrMat{Float32}, y::AbstractVecOrMat{Float32}; threadpool = nothing)
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    nnp_softmax_output(ndims(x) == 2 ? size(x, 2) : 1, size(x, 1), x, y, threadpool)
    y
end

#FIXME: Output of fully connected not consistent with `kernel * input`

function nnp_fully_connected_output(batch_size, input_channels, output_channels, input, kernel, output, threadpool, profile)
    @check ccall((:nnp_fully_connected_output, libnnpack), nnp_status, (Csize_t, Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, pthreadpool_t, Ptr{Cvoid}), batch_size, input_channels, output_channels, input, kernel, output, threadpool, C_NULL)
end

function nnp_fully_connected_output(x::AbstractArray{Float32,2}, w::AbstractArray{Float32,2}, y::AbstractArray{Float32,2}; profile = nothing, threadpool = nothing)
    profile = profile == nothing ? nnp_profile() : profile
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    nnp_fully_connected_output(size(x, 2), size(x, 1), size(w, 1), x, w, y, threadpool, profile)
    y
end

function nnp_fully_connected_inference_f16f32(input_channels, output_channels, input, kernel, output, threadpool)
    @check ccall((:nnp_fully_connected_inference_f16f32, libnnpack), nnp_status, (Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cvoid}, Ptr{Cfloat}, pthreadpool_t), input_channels, output_channels, input, kernel, output, threadpool)
end

nnp_fully_connected_inference_f16f32(x::AbstractVector{Float32}, w::AbstractArray{Float16,2}, y::AbstractVector{Float32}; threadpool = nothing) =
    nnp_fully_connected_inference(reshape(x, size(x), 1), w, reshape(y, size(y), 1), threadpool = threadpool)

function nnp_fully_connected_inference_f16f32(x::AbstractMatrix{Float32}, w::AbstractArray{Float16,2}, y::AbstractMatrix{Float32}; threadpool = nothing)
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    nnp_fully_connected_inference(size(x, 1), size(y, 1), x, w, y, threadpool)
    y
end

function nnp_fully_connected_inference(input_channels, output_channels, input, kernel, output, threadpool)
    @check ccall((:nnp_fully_connected_inference, libnnpack), nnp_status, (Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, pthreadpool_t), input_channels, output_channels, input, kernel, output, threadpool)
end

nnp_fully_connected_inference(x::AbstractVector{Float32}, w::AbstractArray{Float32,2}; threadpool = nothing) =
    nnp_fully_connected_inference(reshape(x, size(x), 1), w, threadpool = threadpool)

function nnp_fully_connected_inference(x::AbstractMatrix{Float32}, w::AbstractMatrix{Float32}, y::AbstractMatrix{Float32}; threadpool = nothing)
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    nnp_fully_connected_inference(size(x, 1), size(y, 1), x, w, y, threadpool)
    y
end

function nnp_max_pooling_output(batch_size, channels, input_size, input_padding, pooling_size, pooling_stride, input, output, threadpool)
    @check ccall((:nnp_max_pooling_output, libnnpack), nnp_status, (Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size, nnp_size, Ptr{Cfloat}, Ptr{Cfloat}, pthreadpool_t), batch_size, channels, input_size, input_padding, pooling_size, pooling_stride, input, output, threadpool)
end

function nnp_max_pooling_output(x::AbstractArray{Float32,4}, y::AbstractArray{Float32,4}, kernel::Tuple; padding = 0, stride = 1, threadpool = nothing)
    input_size = nnp_size(Csize_t.((size(x, 1), size(x, 2)))...)
    pooling_size = nnp_size(Csize_t.(kernel)...)
    input_padding = nnp_padding(Csize_t(padding[2]), Csize_t(padding[1]), Csize_t(padding[2]), Csize_t(padding[1]))
    pooling_stride = nnp_size(Csize_t.(padtuple(x, stride))...)
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    nnp_max_pooling_output(size(x, 4), size(x, 3), input_size, input_padding, pooling_size, pooling_stride, x, y, threadpool)
    y
end

#TODO: Add wrapper for convolution inference

function nnp_convolution_input_gradient(algorithm, batch_size, input_channels, output_channels, input_size, input_padding, kernel_size, grad_output, kernel, grad_input, workspace_buffer, workspace_size, activation, activation_parameters, threadpool, profile)
    @check ccall((:nnp_convolution_kernel_gradient, libnnpack), nnp_status, (nnp_convolution_algorithm, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cvoid}, Csize_t, nnp_activation, Ptr{Cvoid}, pthreadpool_t, Ptr{Cvoid}), algorithm, batch_size, input_channels, output_channels, input_size, input_padding, kernel_size, grad_output, kernel, grad_input, workspace_buffer, workspace_size, activation, activation_parameters, threadpool, C_NULL)
end

function nnp_convolution_input_gradient(dx::AbstractArray{Float32,4}, x::AbstractArray{Float32,4}, dy::AbstractArray{Float32,4}, w::AbstractArray{Float32,4}; algo::nnp_convolution_algorithm = UInt32(0), workspace_buffer = nothing, workspace_size = 0, padding = 0, stride = 1, threadpool = nothing, profile = nothing)
    input_size = nnp_size(Csize_t.((size(x,1), size(x,2)))...)
    kernel_size = nnp_size(Csize_t.((size(w,1),size(w,2)))...)
    input_padding = nnp_padding(Csize_t(padding[2]), Csize_t(padding[1]), Csize_t(padding[2]), Csize_t(padding[1]))
    profile = profile == nothing ? nnp_profile() : profile
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    workspace_buffer = workspace_buffer === nothing ? C_NULL : workspace_buffer
    nnp_convolution_input_gradient(UInt32(algo), size(x,4), size(x,3), size(w,4), input_size, input_padding, kernel_size, dy, w, dx, workspace_buffer, workspace_size, UInt32(0), C_NULL, threadpool, profile)
    dx
end

function nnp_convolution_kernel_gradient(algorithm, batch_size, input_channels, output_channels, input_size, input_padding, kernel_size, input, grad_output, grad_kernel, workspace_buffer, workspace_size, activation, activation_parameters, threadpool, profile)
    @check ccall((:nnp_convolution_kernel_gradient, libnnpack), nnp_status, (nnp_convolution_algorithm, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cvoid}, Csize_t, nnp_activation, Ptr{Cvoid}, pthreadpool_t, Ptr{Cvoid}), algorithm, batch_size, input_channels, output_channels, input_size, input_padding, kernel_size, input, grad_output, grad_kernel, workspace_buffer, workspace_size, activation, activation_parameters, threadpool, C_NULL)
end

function nnp_convolution_kernel_gradient(dw::AbstractArray{Float32,4}, x::AbstractArray{Float32,4}, dy::AbstractArray{Float32,4}, w::AbstractArray{Float32,4}; algo::nnp_convolution_algorithm = UInt32(0), workspace_buffer = nothing, workspace_size = 0, padding = 0, stride = 1, threadpool = nothing, profile = nothing)
    input_size = nnp_size(Csize_t.((size(x,1), size(x,2)))...)
    kernel_size = nnp_size(Csize_t.((size(w,1),size(w,2)))...)
    input_padding = nnp_padding(Csize_t(padding[2]), Csize_t(padding[1]), Csize_t(padding[2]), Csize_t(padding[1]))
    profile = profile == nothing ? nnp_profile() : profile
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    workspace_buffer = workspace_buffer === nothing ? C_NULL : workspace_buffer
    nnp_convolution_kernel_gradient(UInt32(algo), size(x,4), size(x,3), size(w,4), input_size, input_padding, kernel_size, x, dy, dw, workspace_buffer, workspace_size, UInt32(0), C_NULL, threadpool, profile)
    dw
end

function nnp_convolution_output(algorithm, batch_size, input_channels, output_channels, input_size, input_padding, kernel_size, input, kernel, bias, output, workspace_buffer, workspace_size, activation, activation_parameters, threadpool, profile)
    @check ccall((:nnp_convolution_output, libnnpack), nnp_status, (nnp_convolution_algorithm, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Cvoid}, Csize_t, nnp_activation, Ptr{Cvoid}, pthreadpool_t, Ptr{Cvoid}), algorithm, batch_size, input_channels, output_channels, input_size, input_padding, kernel_size, input, kernel, bias, output, workspace_buffer, workspace_size, activation, activation_parameters, threadpool, C_NULL)
end

function nnp_convolution_output(y::AbstractArray{Float32,4}, x::AbstractArray{Float32,4}, w::AbstractArray{Float32,4}, b::AbstractArray{Float32,1}; algo::nnp_convolution_algorithm = UInt32(0), workspace_buffer = nothing, workspace_size = 0, padding = 0, stride = 1, threadpool = nothing, profile = nothing)
    input_size = nnp_size(Csize_t.((size(x,1), size(x,2)))...)
    kernel_size = nnp_size(Csize_t.((size(w,1),size(w,2)))...)
    input_padding = nnp_padding(Csize_t(padding[2]), Csize_t(padding[1]), Csize_t(padding[2]), Csize_t(padding[1]))
    profile = profile == nothing ? nnp_profile() : profile
    threadpool = threadpool === nothing ? pthreadpool_create() : threadpool
    workspace_buffer = workspace_buffer === nothing ? C_NULL : workspace_buffer
    nnp_convolution_output(UInt32(algo), size(x,4), size(x,3), size(w,4), input_size, input_padding, kernel_size, x, w, b, y, workspace_buffer, workspace_size, UInt32(0), C_NULL, threadpool, profile)
    y
end
