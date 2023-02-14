using NNlib: batched_adjoint, batched_mul, batched_mul!, batched_transpose
using NNlib: is_strided, storage_type
using LinearAlgebra

AMDGPU.allowscalar(false)

function gputest(f, xs...; checkgrad=true, atol=1e-6, kws...)
    cpu_in = xs
    gpu_in = ROCArray.(xs)

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
end

@testset "Storage types" begin
    include("storage_type.jl")
end

@testset "Batched repr" begin
    include("batched_repr.jl")
end

@testset "Batched multiplication" begin
    include("batched_mul.jl")
end

@testset "Convolution" begin
    include("conv.jl")
end

@testset "Pooling" begin
    include("pool.jl")
end

@testset "Softmax" begin
    include("softmax.jl")
end

@testset "Activations" begin
    include("activations.jl")
end

@testset "Dropout" begin
    include("dropout.jl")
end

@testset "Attention" begin
    include("attention.jl")
end
