function select_threadpool(cdims::ConvDims, batch_size::Int)
    inp_size = input_size(cdims)[1]
    if batch_size >= 32
        return shared_threadpool_dict[Int(NNPACK_CPU_THREADS)][]
    elseif batch_size >= 16 && inp_size >= 64
        return shared_threadpool_dict[Int(NNPACK_CPU_THREADS)][]
    elseif inp_size <= 32
        return C_NULL
    elseif inp_size >= 128
        return shared_threadpool_dict[Int(NNPACK_CPU_THREADS)][]
    elseif inp_size * batch_size >= 256
        return shared_threadpool_dict[Int(NNPACK_CPU_THREADS)][]
    end
    return C_NULL
end

function select_threadpool(pdims::PoolDims, batch_size::Int)
    inp_size = input_size(pdims)[1]
    if batch_size >= 32
        return shared_threadpool_dict[Int(NNPACK_CPU_THREADS)][]
    elseif batch_size >= 16 && inp_size >= 64
        return shared_threadpool_dict[Int(NNPACK_CPU_THREADS)][]
    elseif inp_size <= 32
        return C_NULL
    elseif inp_size >= 128
        return shared_threadpool_dict[Int(NNPACK_CPU_THREADS)][]
    elseif inp_size * batch_size >= 256
        return shared_threadpool_dict[Int(NNPACK_CPU_THREADS)][]
    end
    return C_NULL
end
