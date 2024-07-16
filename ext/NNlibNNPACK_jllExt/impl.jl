function maxpool_nnpack!(y::A, x::A, pdims::PoolDims) where {A<:Array{Float32, 4}}
    check_dims(size(x), size(y), pdims)
    threadpool = select_threadpool(pdims, size(y, 4))
    nnp_max_pooling_output(y, x, kernel_size(pdims), padding = padding(pdims),
                           stride = stride(pdims), threadpool = threadpool)
end

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

