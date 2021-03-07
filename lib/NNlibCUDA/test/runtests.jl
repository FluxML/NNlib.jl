using Test
using NNlib
using Zygote
using CUDA
using CUDA: @allowscalar
CUDA.allowscalar(false)

function gputest(f, xs...)
    cpu_in = xs
    gpu_in = CuArray.(xs)

    cpu_out = f(cpu_in...)
    gpu_out = f(gpu_in...)
    @test collect(cpu_out) ≈ collect(gpu_out)
    
    cpu_grad = gradient((x...) -> sum(f(x...)), cpu_in...)
    gpu_grad = gradient((x...) -> sum(f(x...)), gpu_in...)
    for (cpu_g, gpu_g) in zip(cpu_grad, gpu_grad)
        if cpu_g === nothing
            @test gpu_g === nothing
        else
            @test collect(cpu_g) ≈ collect(gpu_g)
        end
    end
end


if CUDA.has_cuda()
    include("activations.jl")
    include("batchedmul.jl")
    include("upsample.jl")
end
