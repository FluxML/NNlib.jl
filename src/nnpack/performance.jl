function select_threadpool(cdims::DenseConvDims, batch_size::Int)
    return C_NULL
end

function select_threadpool(pdims::PoolDims, batch_size::Int)
    inp_size = input_size(pdims)[1] 
    if batch_size >= 32
        return shared_threadpool_dict[4][]
    elseif batch_size >= 16 && inp_size >= 64
        return shared_threadpool_dict[4][]
    elseif inp_size <= 32
        return C_NULL
    elseif inp_size >= 128
        return shared_threadpool_dict[4][]
    elseif inp_size * batch_size >= 256
        return shared_threadpool_dict[4][]
    end    
    return C_NULL
end