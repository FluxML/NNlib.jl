# Setup for the Metal test files: backend imports + the Metal `gputest` + `DEVICE`.
# Loaded into every Metal-run worker by `init_code` in `test/runtests.jl`.

using NNlib
using Test
using Metal
using Zygote: gradient
using MLDataDevices: gpu_device
using ForwardDiff: Dual
using Statistics: mean

Metal.allowscalar(false)

function gputest(device, f, xs...; checkgrad=true, atol=1e-6, kws...)
    cpu_in = xs
    gpu_in = device(xs)

    cpu_out = f(cpu_in...; kws...)
    gpu_out = f(gpu_in...; kws...)
    @test collect(cpu_out) ≈ collect(gpu_out)

    if checkgrad
        cpu_grad = gradient((x...) -> sum(f(x...; kws...)), cpu_in...)
        gpu_grad = gradient((x...) -> sum(f(x...; kws...)), gpu_in...)
        for (cpu_g, gpu_g) in zip(cpu_grad, gpu_grad)
            if cpu_g === nothing
                @test gpu_g === nothing
            else
                @test collect(cpu_g) ≈ collect(gpu_g) atol=atol
            end
        end
    end
    return true
end

DEVICE = gpu_device(force=true)
