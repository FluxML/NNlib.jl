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
