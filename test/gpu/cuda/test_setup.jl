# Setup for the CUDA test files: backend imports + the CUDA `gputest` (which takes
# CPU inputs and moves them to the device). Loaded into every CUDA-run worker by
# `init_code` in `test/runtests.jl`, AFTER `test_module.jl` — so the shared imports
# (Test, NNlib, Zygote, Random, Statistics, Mooncake, ...) are already in scope here.
# Distinct from `test_gradients` in `test_module.jl`, which the shared
# `common_testsuite/` suites use.

using CUDA, cuDNN
using BFloat16s: BFloat16             # bare `BFloat16` in cuda/bfloat16.jl
import CUDA.CUSPARSE: CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO
using ForwardDiff: Dual              # bare `Dual` in cuda/{activations,softmax}.jl
using Mooncake.TestUtils: test_rule  # bare `test_rule` in cuda/batchnorm.jl
using NNlib: batchnorm, ∇batchnorm   # NNlib internals exercised by cuda/batchnorm.jl
CUDA.allowscalar(false)

function gputest(f, xs...; checkgrad=true, rtol=1e-7, atol=1e-10, broken=false, broken_grad=false, kws...)
    cpu_in = xs
    gpu_in = CuArray.(xs)

    cpu_out = f(cpu_in...; kws...)
    gpu_out = f(gpu_in...; kws...)
    @test collect(cpu_out) ≈ collect(gpu_out) rtol=rtol atol=atol broken=broken

    if checkgrad
        # use mean instead of sum to prevent error accumulation (for larger
        # tensors) which causes error to go above atol
        cpu_grad = gradient((x...) -> mean(f(x...; kws...)), cpu_in...)
        gpu_grad = gradient((x...) -> mean(f(x...; kws...)), gpu_in...)
        for (cpu_g, gpu_g) in zip(cpu_grad, gpu_grad)
            if cpu_g === nothing
                @test gpu_g === nothing
            else
                @test collect(cpu_g) ≈ collect(gpu_g) rtol=rtol atol=atol broken=broken_grad
            end
        end
    end
end
