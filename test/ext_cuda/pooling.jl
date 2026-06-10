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

    # Mean pooling is linear, so the cuDNN extension supports complex inputs by
    # pooling the real and imaginary parts independently. We check it agrees with
    # the (generic) CPU path. `maxpool` is intentionally NOT supported, since
    # `max` is undefined for complex numbers; it errors on both CPU and GPU.
    # `gputest`'s gradient check goes through `mean`, which Zygote rejects for
    # complex outputs, so we use `checkgrad=false` and compare `∇meanpool` directly.
    for num_spatial_dims in (1, 2, 3)
        C_in = 3
        batch_size = 2
        x = rand(ComplexF64, fill(8, num_spatial_dims)..., C_in, batch_size)

        # no padding
        pdims = PoolDims(x, 2)
        y = meanpool(x, pdims)
        dy = rand(ComplexF64, size(y))
        @test eltype(meanpool(CuArray(x), pdims)) == ComplexF64
        gputest(x -> meanpool(x, pdims), x, checkgrad=false)
        gputest((dy, y, x) -> ∇meanpool(dy, y, x, pdims), dy, y, x, checkgrad=false)

        # with padding: count_include_pad changes the result (issue #218)
        pdims_pad = PoolDims(x, 2; padding=1, stride=2)
        for cip in (true, false)
            ym = meanpool(x, pdims_pad; count_include_pad=cip)
            dym = rand(ComplexF64, size(ym))
            gputest(x -> meanpool(x, pdims_pad; count_include_pad=cip), x, checkgrad=false)
            gputest((dy, y, x) -> ∇meanpool(dy, y, x, pdims_pad; count_include_pad=cip),
                    dym, ym, x, checkgrad=false)
        end

        # maxpool has no complex extension (max is undefined for complex)
        @test_throws Exception maxpool(CuArray(x), pdims)
    end
end
