using PrecompileTools: @compile_workload

# Host code only: precompiling must not need a GPU. The kernels are compiled on first use.
# Ops that cuDNN takes over are precompiled in NNlibCUDACUDNNExt.
@static if isdefined(CUDA, :DeviceMemory)
    @compile_workload begin
        V = CuArray{Float32, 1, CUDA.DeviceMemory}
        M = CuArray{Float32, 2, CUDA.DeviceMemory}
        B = CuArray{Float32, 3, CUDA.DeviceMemory}
        A = CuArray{Float32, 4, CUDA.DeviceMemory}
        precompile(batched_mul, (B, B))
        precompile(upsample_nearest, (A, NTuple{2, Int}))
        precompile(∇upsample_nearest, (A, NTuple{2, Int}))
        precompile(dropout, (A, Float64))
        for σ in (identity, relu, tanh, sigmoid)
            precompile(bias_act!, (typeof(σ), A, A))
            precompile(bias_act!, (typeof(σ), M, V))
        end
        for σ in (leakyrelu, gelu, swish, softplus, tanh_fast)
            precompile(broadcast, (typeof(σ), A))
        end
    end
end
