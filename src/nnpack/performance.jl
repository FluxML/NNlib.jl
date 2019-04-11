function select_threadpool(cdims::DenseConvDims, batch_size::Int)
    return C_NULL
end

function select_threadpool(pdims::PoolDims, batch_size::Int)
    return C_NULL
end
