function nnpack_supported_operation(x::AbstractArray{<:Real, 4}, k, pad, stride, dilation)
    fallback = false
    # NNPACK does not support dilated convolutions
    dilation == 1 || dilation == (1, 1) || (fallback = true)
    # Expand the pad and stride to have same dimensions as k
    pad_, stride_ = expand(Val{length(k)}, pad), expand(Val{length(k)}, stride)
    (size(x, 1) - k[1] + 2 * pad_[1]) % stride_[1] == 0 && (size(x, 2) - k[2] + 2 * pad_[2]) % stride_[2] == 0 || (fallback = true)
    # Return the pad_ and stride_ as well
    return pad_, stride_, fallback
end

function nnpack_speed_check(x::AbstractArray{<:Real, 4}, k, pad, stride, dilation)
    # Add heurestics here to determine whether or not to use NNPACK
    # For now just return true
    return true
end

# NNPACK supports only Float32 operations. So Float64 will have it default behaviour

# Pooling
function maxpool!(y::A, x::A, k; pad = map(_ -> 0, k), stride = k) where A<:Array{Float32, 4}
    pad_, stride_, use_default = nnpack_supported_operation(x, k, pad, stride, 1)
    use_nnpack = !use_default
    # Only use NNPACK if we get speed improvement
    use_nnpack && (use_nnpack = nnpack_speed_check(x, k, pad, stride, 1))
    if use_nnpack
        nnpack_max_pooling!(y, x, k, pad = pad_, stride = stride_)
    else
        maxpool_cpu!(y, x, k, pad = pad_, stride = stride_)
    end
end

# Convolutions
function conv!(y::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:Array{Float32, 4}
    k = (size(w, 1), size(w, 2))
    pad_, stride_, use_default = nnpack_supported_operation(x, k, pad, stride, 1)
    use_nnpack = !use_default
    use_nnpack && (use_nnpack = nnpack_speed_check(x, k, pad, stride, 1))
    if use_nnpack
        nnpack_convolution_forward!(y, x, w, zeros(Float32, size(y, 3)), algo = algo, pad = pad, stride = stride, flipkernel = flipkernel)
    else
        conv2d!(y, x, w, padding = pad_, stride = stride_, dilation = dilation, mode = flipkernel)
    end
end

function ∇conv_data!(dx::A, dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:Array{Float32, 4}
    k = (size(w, 1), size(w, 2))
    pad_, stride_, use_default = nnpack_supported_operation(x, k, pad, stride, 1)
    use_nnpack = !use_default
    use_nnpack && (use_nnpack = nnpack_speed_check(x, k, pad, stride, 1))
    if use_nnpack
        nnpack_convolution_backward_data!(dx, x, dy, w, pad = pad_, stride = stride_, algo = algo, flipkernel = flipkernel)
    else
        conv2d_grad_x!(dx, x, w, dy, padding = pad_, stride = stride_, dilation = dilation, mode = flipkernel)
    end
end

function ∇conv_filter!(dw::A, dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1, algo = UInt32(0), flipkernel = 0) where A<:Array{Float32, 4}
    k = (size(w, 1), size(w, 2))
    pad_, stride_, use_default = nnpack_supported_operation(x, k, pad, stride, 1)
    use_nnpack = !use_default
    use_nnpack && (use_nnpack = nnpack_speed_check(x, k, pad, stride, 1))
    if use_nnpack
        nnpack_convolution_backward_filter!(dw, x, dy, w, pad = pad_, stride = stride_, algo = algo, flipkernel = flipkernel)
    else
        conv2d_grad_w!(dw, x, w, dy, padding = pad_, stride = stride_, dilation = dilation, mode = flipkernel)
    end
end
