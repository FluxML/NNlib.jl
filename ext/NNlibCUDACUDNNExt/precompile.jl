using PrecompileTools: @compile_workload

# Host code only: precompiling must not need a GPU. The kernels are compiled on first use.
@static if isdefined(CUDA, :DeviceMemory)
    @compile_workload begin
        M = CuArray{Float32, 2, CUDA.DeviceMemory}
        A = CuArray{Float32, 4, CUDA.DeviceMemory}
        cdims = typeof(DenseConvDims((8, 8, 3, 2), (3, 3, 3, 4)))
        pdims = typeof(PoolDims((8, 8, 4, 2), (2, 2)))
        precompile(conv, (A, A, cdims))
        precompile(∇conv_data, (A, A, cdims))
        precompile(∇conv_filter, (A, A, cdims))
        for (pool, ∇pool) in ((maxpool, ∇maxpool), (meanpool, ∇meanpool))
            precompile(pool, (A, pdims))
            precompile(pool, (A, NTuple{2, Int}))
            precompile(∇pool, (A, A, A, pdims))
        end
        for (f, ∇f) in ((softmax, ∇softmax), (logsoftmax, ∇logsoftmax))
            precompile(f, (M,))
            precompile(∇f, (M, M))
        end
        precompile(sdpa, (A, A, A))
        for σ in (relu, sigmoid, tanh, elu)
            precompile(broadcast, (typeof(σ), A))
        end
    end
end
