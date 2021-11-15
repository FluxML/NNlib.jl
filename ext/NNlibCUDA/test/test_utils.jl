function gputest(f, xs...; checkgrad=true, atol=1e-10, kws...)
    cpu_in = xs
    gpu_in = CuArray.(xs)

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
                @test collect(cpu_g) ≈ collect(gpu_g)  atol=atol
            end
        end
    end
end
