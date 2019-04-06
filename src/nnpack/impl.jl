@inline flipweight(w::Array{<:Any, 4}) = w[end:-1:1,end:-1:1,:,:]

function maxpool_nnpack!(y::A, x::A, pdims::PoolDims) where {A<:Array{Float32, 4}}
    check_dims(size(x), size(y), pdims)

    pad = padding(pdims)
    stride_ = stride(pdims)
    kernel = kernel_size(pdims)
    
    nnp_max_pooling_output(y, x, kernel, padding = pad, stride = stride_)
end

@timeit_debug to function conv_nnpack!(y::A1, x::A1, w::A1, cdims::ConvDims;
                                       b::A2 = zeros(Float32, size(x, 3)),
                                       algo = UInt32(0)) where {A1<:Array{Float32, 4},
                                                                A2<:Array{Float32, 1}}
    check_dims(size(x), size(w), size(y), cdims)
    
    flipkernel_ = flipkernel(cdims)
    if flipkernel_ == 0
        w .= flipweight(w)
    end

    pad = padding(cdims)
    stride_ = stride(cdims)

    nnp_convolution_output(y, x, w, b, algo = algo, padding = pad, stride = stride_)
end

@timeit_debug to function ∇conv_data_nnpack!(dx::A, dy::A, w::A, cdims::ConvDims;
                                             algo = UInt32(0)) where{A<:Array{Float32, 4}}
    check_dims(size(dx), size(w), size(dy), cdims)
    
    flipkernel_ = flipkernel(cdims)
    if flipkernel_ == 0
        w .= flipweight(w)
    end

    pad = padding(cdims)
    stride_ = stride(cdims)

    nnp_convolution_input_gradient(dx, dy, w, algo = algo, padding = pad, stride = stride_)
end

@timeit_debug to function ∇conv_filter_nnpack!(dw::A, x::A, dy::A, cdims::ConvDims;
                                               algo = UInt32(0)) where{A<:Array{Float32, 4}}
    check_dims(size(x), size(dw), size(dy), cdims)
    
    flipkernel_ = flipkernel(cdims)
    pad = padding(cdims)
    stride_ = stride(cdims)

    nnp_convolution_kernel_gradient(dw, x, dy, algo = algo, padding = pad, stride = stride_)

    if flipkernel_ == 0
        dw .= flipweight(dw)
    end

    dw
end

