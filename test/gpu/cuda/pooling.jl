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
        @test test_gradients(x -> maxpool(x, pdims), x; test_gpu=true)
        gputest((dy, y, x) -> ∇maxpool(dy, y, x, pdims), dy, y, x, checkgrad=false)
        @test test_gradients(x -> maxpool(x, pdims), x; test_gpu=true)
        gputest((dy, y, x) -> ∇maxpool(dy, y, x, pdims), dy, y, x, checkgrad=false)

        # asymmetric padding, which only cuDNN's graph API supports
        pdims_asym = PoolDims(x, 3; stride=2,
                              padding=ntuple(i -> isodd(i) ? 1 : 0, 2num_spatial_dims))
        gputest(x -> maxpool(x, pdims_asym), x)
        for cip in (true, false)
            gputest(x -> meanpool(x, pdims_asym; count_include_pad=cip), x)
        end

        # meanpool count_include_pad (issue #218): with padding the two modes differ,
        # check both agree with cuDNN (forward and backward).
        pdims_pad = PoolDims(x, 2; padding=1, stride=2)
        for cip in (true, false)
            ym = meanpool(x, pdims_pad; count_include_pad=cip)
            dym = ones(size(ym))
            @test test_gradients(x -> meanpool(x, pdims_pad; count_include_pad=cip), x; test_gpu=true)
            gputest((dy, y, x) -> ∇meanpool(dy, y, x, pdims_pad; count_include_pad=cip),
                    dym, ym, x, checkgrad=false)
        end

        # ceil_mode (issue #839), which cuDNN supports with symmetric padding
        xc = rand(Float64, fill(6, num_spatial_dims)..., C_in, 2)
        for pdims_ceil in (PoolDims(xc, 3; stride=2, ceil_mode=true),
                           PoolDims(xc, 3; padding=1, stride=2, ceil_mode=true))
            @test NNlib.ceil_mode(pdims_ceil)
            gputest(x -> maxpool(x, pdims_ceil), xc)
            for cip in (true, false)
                gputest(x -> meanpool(x, pdims_ceil; count_include_pad=cip), xc)
            end
        end
        pdims_ceil = PoolDims(xc, 2; stride=2, ceil_mode=true,
                              padding=ntuple(i -> isodd(i) ? 1 : 0, 2num_spatial_dims))
        @test NNlib.ceil_mode(pdims_ceil)
        @test_throws ArgumentError maxpool(CuArray(xc), pdims_ceil)
        @test_throws ArgumentError meanpool(CuArray(xc), pdims_ceil)
    end
end

@testset "asymmetric padding: NaN, and padding past `pre`" begin
    # Before cuDNN.jl 6.5, the graph API propagates NaN in max pooling, unlike the CPU and
    # the fixed-function API, and mean pooling counts the padding past `pre` wrong
    # (JuliaGPU/CUDA.jl#3473).
    broken = pkgversion(cuDNN) < v"6.5"
    for num_spatial_dims in (1, 2, 3)
        x = rand(Float64, fill(7, num_spatial_dims)..., 3, 2)
        pdims = PoolDims(x, 3; stride=2, padding=ntuple(i -> isodd(i) ? 0 : 2, 2num_spatial_dims))
        gputest(x -> meanpool(x, pdims), x; broken, broken_grad=broken)
        x[3, ntuple(Returns(1), num_spatial_dims + 1)...] = NaN
        pdims = PoolDims(x, 3; stride=2, padding=ntuple(i -> isodd(i) ? 1 : 0, 2num_spatial_dims))
        gputest(x -> maxpool(x, pdims), x; broken, broken_grad=broken)
    end
end

@testset "window, padding or stride larger than the input (issue #862)" begin
    for num_spatial_dims in (1, 2, 3), (n, k, pad, stride) in ((1, 3, 1, 1), (2, 3, 1, 1), (2, 3, 1, 2),
                                                                (4, 5, 2, 1), (1, 2, 1, 1), (2, 3, 2, 3))
        x = rand(Float64, fill(n, num_spatial_dims)..., 3, 2)
        pdims = PoolDims(x, k; padding=pad, stride)
        gputest(x -> maxpool(x, pdims), x)
        for cip in (true, false)
            gputest(x -> meanpool(x, pdims; count_include_pad=cip), x)
        end
    end
    x = rand(Float64, 1, 5, 3, 2)
    pdims = PoolDims(x, 3; padding=1, stride=1)
    gputest(x -> maxpool(x, pdims), x)
    gputest(x -> meanpool(x, pdims), x)
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

@testset "non-dense views (issue #863)" begin
    # cuDNN.jl 6.5 takes views with positive strides, older versions copy them (and the
    # gradients always do): compare with the dense copies.
    @testset "$(nd)d" for nd in (1, 2, 3)
        cs = ntuple(_ -> Colon(), nd)
        x = CUDA.randn(Float32, fill(7, nd)..., 6, 2)
        @testset "$I" for I in ((cs..., 3:4, :), (2:6, cs[2:end]..., 1:2, :),
                                (6:-1:2, cs[2:end]..., 1:2, :))
            v = view(x, I...)
            # symmetric padding goes through the fixed-function API, asymmetric through the graph API
            for pdims in (PoolDims(v, 2), PoolDims(v, 2; padding=ntuple(i -> isodd(i) ? 1 : 0, 2nd))),
                pool in (maxpool, meanpool)
                @test pool(v, pdims) ≈ pool(x[I...], pdims)
                @test gradient(x -> sum(abs2, pool(view(x, I...), pdims)), x)[1] ≈
                      gradient(x -> sum(abs2, pool(x[I...], pdims)), x)[1]
            end
        end
    end
end
