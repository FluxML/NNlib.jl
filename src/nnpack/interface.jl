export is_nnpack_available,
       # Pooling
       maxpool_nnpack!, nnpack_supported_operation,
       # Convolution
       conv_nnpack!, ∇conv_data_nnpack!, ∇conv_filter_nnpack!

"""
    is_nnpack_available()

Checks if the current hardware is supported by NNPACK.

While the platform itself may be supported by NNPACK, certain hardware
configurations (such as processors lacking SSE) are not.
"""
function is_nnpack_available()
    return nnp_initialize() != nnp_status_unsupported_hardware
end

# Conv
function conv_nnpack!(y::A1, x::A1, w::A1, cdims::ConvDims;
                                       b::A2 = zeros(Float32, size(x, 3)),
                                       algo = UInt32(0)) where {A1<:Array{Float32, 4},
                                                                A2<:Array{Float32, 1}}
    check_dims(size(x), size(w), size(y), cdims)
    threadpool = select_threadpool(cdims, size(y, 4))

    if flipkernel(cdims) == 0
        w = flipweight(w)
    end

    nnp_convolution_output(y, x, w, b, algo = algo, padding = padding(cdims),
                           stride = stride(cdims), threadpool = threadpool)
end

function ∇conv_data_nnpack!(dx::A, dy::A, w::A, cdims::ConvDims;
                                             algo = UInt32(0)) where{A<:Array{Float32, 4}}
    check_dims(size(dx), size(w), size(dy), cdims)
    threadpool = select_threadpool(cdims, size(dy, 4))
    
    if flipkernel(cdims) == 0
        w = flipweight(w)
    end

    nnp_convolution_input_gradient(dx, dy, w, algo = algo, padding = padding(cdims),
                                   stride = stride(cdims), threadpool = threadpool)
end

function ∇conv_filter_nnpack!(dw::A, x::A, dy::A, cdims::ConvDims;
                                               algo = UInt32(0)) where{A<:Array{Float32, 4}}
    check_dims(size(x), size(dw), size(dy), cdims)
    threadpool = select_threadpool(cdims, size(dy, 4))
    
    nnp_convolution_kernel_gradient(dw, x, dy, algo = algo, padding = padding(cdims),
                                    stride = stride(cdims), threadpool = threadpool)

    if flipkernel(cdims) == 0
        dw .= flipweight(dw)
    end

    dw
end


# Pooling
function maxpool_nnpack!(y::A, x::A, pdims::PoolDims) where {A<:Array{Float32, 4}}
    check_dims(size(x), size(y), pdims)
    threadpool = select_threadpool(pdims, size(y, 4))
    nnp_max_pooling_output(y, x, kernel_size(pdims), padding = padding(pdims),
                           stride = stride(pdims), threadpool = threadpool)
end

"""
    nnpack_supported_operation(cdims::ConvDims)

Returns `true` if nnpack supports the conv/pooling operation for the given
parameters.  For convolution this can be known at compile-time, however for
pooling, we cannot describe the stride domain constraint purely with types,
so we must do it at runtime with this method.
"""
function nnpack_supported_operation(pdims::PoolDims{2, K, S, P, (1, 1)}) where {K, S, P}
    # Ensure that the kernel striding perfectly covers the padded input size.
    stride_domain = input_size(pdims)[1:2] .+ (P[1] + P[2], P[3] + P[4]) .- K
    return stride_domain .% S == (0, 0)
end

NNPACK_CDIMS = DenseConvDims{2,K,C_in,C_out,(1,1),P,(1,1),F} where {K,C_in,C_out,P,F}
nnpack_supported_operation(cdims::NNPACK_CDIMS) = true

# Say false by default
nnpack_supported_operation(dims) = false
