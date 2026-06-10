@testset "pooling" begin

    # Test for agreement between CPU NNlib and CuDNN versions, across a variety of kwargs
    for num_spatial_dims in (1, 2, 3)
        # Initialize data we'll run our tests over
        C_in = 3
        batch_size = 1
        x = rand(Float64, fill(8, num_spatial_dims)..., C_in, batch_size)
       
        # Test that pooling is equivalent across GPU/CPU
        pdims = PoolDims(x, 2)
        y = maxpool(x, pdims)
        dy = ones(size(y))
        gputest(x -> maxpool(x, pdims), x)
        gputest((dy, y, x) -> ∇maxpool(dy, y, x, pdims), dy, y, x, checkgrad=false)
        gputest(x -> maxpool(x, pdims), x)
        gputest((dy, y, x) -> ∇maxpool(dy, y, x, pdims), dy, y, x, checkgrad=false)

        # meanpool count_include_pad (issue #218): with padding the two modes differ,
        # check both agree with cuDNN (forward and backward).
        pdims_pad = PoolDims(x, 2; padding=1, stride=2)
        for cip in (true, false)
            ym = meanpool(x, pdims_pad; count_include_pad=cip)
            dym = ones(size(ym))
            gputest(x -> meanpool(x, pdims_pad; count_include_pad=cip), x)
            gputest((dy, y, x) -> ∇meanpool(dy, y, x, pdims_pad; count_include_pad=cip),
                    dym, ym, x, checkgrad=false)
        end
    end
end

@testset "complex meanpool (issue #610)" begin

    # Complex `meanpool` is implemented once in core (NNlib pools the real and
    # imaginary parts separately), so cuDNN needs no complex-specific method; the
    # real parts dispatch to the real cuDNN `meanpool`. We check the result and its
    # gradient agree with the CPU path. The gradient goes through AD (the `meanpool`
    # rrule is real-only, so AD differentiates the split); we use a real-valued loss
    # since Zygote rejects gradients of complex outputs. `maxpool` is unsupported
    # (`max` is undefined for complex) and errors on both CPU and GPU.
    for num_spatial_dims in (1, 2, 3)
        C_in = 3
        batch_size = 2
        x = rand(ComplexF64, fill(8, num_spatial_dims)..., C_in, batch_size)

        for pdims in (PoolDims(x, 2), PoolDims(x, 2; padding=1, stride=2))
            for cip in (true, false)
                # forward: GPU matches CPU and stays complex
                y_c = meanpool(x, pdims; count_include_pad=cip)
                y_g = meanpool(CuArray(x), pdims; count_include_pad=cip)
                @test y_g isa CuArray{ComplexF64}
                @test collect(y_g) ≈ y_c

                # gradient via AD: GPU matches CPU and stays complex
                loss(z) = abs2(sum(meanpool(z, pdims; count_include_pad=cip)))
                g_c = gradient(loss, x)[1]
                g_g = gradient(loss, CuArray(x))[1]
                @test g_g isa CuArray{ComplexF64}
                @test collect(g_g) ≈ g_c
            end
        end

        # maxpool has no complex extension (max is undefined for complex)
        @test_throws Exception maxpool(CuArray(x), PoolDims(x, 2))
    end
end
