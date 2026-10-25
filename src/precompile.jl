using PrecompileTools: @compile_workload, @setup_workload

# Float32 CPU arrays only. Ops that launch KernelAbstractions kernels (normalization,
# linear upsampling, ...) are left out: on KernelAbstractions 0.10 the CPU backend is
# POCL, which would compile them through GPUCompiler during precompilation.
@setup_workload begin
    x = rand(Float32, 8, 8, 3, 2)
    w = rand(Float32, 3, 3, 3, 4)
    b = rand(Float32, 1, 1, 4, 1)
    m = rand(Float32, 5, 3)
    v = rand(Float32, 5)
    @compile_workload begin
        y = conv(x, w; pad = 1)
        cdims = DenseConvDims(x, w; padding = 1)
        ∇conv_data(y, w, cdims)
        ∇conv_filter(x, y, cdims)

        pdims = PoolDims(y, (2, 2))
        p = maxpool(y, (2, 2))
        ∇maxpool(p, p, y, pdims)
        p = meanpool(y, (2, 2))
        ∇meanpool(p, p, y, pdims)

        u = upsample_nearest(y, (2, 2))
        ∇upsample_nearest(u, (2, 2))

        s = softmax(m)
        ∇softmax(s, s)
        s = logsoftmax(m)
        ∇logsoftmax(s, s)

        batched_mul(rand(Float32, 3, 4, 2), rand(Float32, 4, 5, 2))
        dropout(y, 0.5)

        for σ in (identity, relu, tanh, sigmoid)
            bias_act!(σ, copy(y), b)
            bias_act!(σ, copy(m), v)
        end
        for σ in (relu, leakyrelu, elu, gelu, swish, softplus, sigmoid, tanh_fast)
            σ.(y)
        end
    end
end
