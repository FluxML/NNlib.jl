@testset "Compare CPU & GPU" begin
    channels, batch = 3, 2
    for T in (Float16, Float32), nd in (1, 2, 3)
        x = rand(T, fill(8, nd)..., channels, batch)
        pdims = PoolDims(x, 2)
        # NOTE: Disable grad check for maxpool as *sometimes*
        # it does not *completely* agree with CPU :/
        gputest(x -> NNlib.maxpool(x, pdims), x; checkgrad=false)
        gputest(x -> NNlib.meanpool(x, pdims), x)
    end
end

@testset "complex meanpool (issue #610)" begin
    # Complex `meanpool` is implemented once in core (NNlib pools the real and
    # imaginary parts separately), so MIOpen needs no complex-specific method: the
    # real parts dispatch to MIOpen's real `meanpool` and its rrule. Gradients flow
    # through AD (there is no standalone `∇meanpool` for `ROCArray`); we use a
    # real-valued loss since Zygote rejects gradients of complex outputs. `maxpool`
    # is unsupported (`max` is undefined for complex).
    channels, batch = 3, 2
    for nd in (1, 2, 3)
        x = rand(ComplexF32, fill(8, nd)..., channels, batch)
        pdims = PoolDims(x, 2)

        # forward: GPU matches CPU and stays a complex ROCArray
        y_c = NNlib.meanpool(x, pdims)
        y_g = NNlib.meanpool(ROCArray(x), pdims)
        @test y_g isa ROCArray{ComplexF32}
        @test collect(y_g) ≈ y_c

        # gradient via AD: GPU matches CPU and stays complex
        loss(z) = abs2(sum(NNlib.meanpool(z, pdims)))
        g_c = gradient(loss, x)[1]
        g_g = gradient(loss, ROCArray(x))[1]
        @test g_g isa ROCArray{ComplexF32}
        @test collect(g_g) ≈ g_c rtol=1f-3

        # maxpool has no complex extension (max is undefined for complex)
        @test_throws Exception NNlib.maxpool(ROCArray(x), pdims)
    end
end
