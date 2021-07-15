using NNlib: DenseConvDims

@testset "convolution" begin
    a, b, c = rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4), rand(Float64, 9, 9, 4, 1)
    da, db, dc = CuArray(a), CuArray(b), CuArray(c)
    cdims = DenseConvDims(a, b)
    @test NNlib.conv(a, b, cdims) ≈ collect(NNlib.conv(da, db, cdims))
    @test ∇conv_data(c, b, cdims) ≈ collect(∇conv_data(dc, db, cdims))
    @test ∇conv_filter(a, c, cdims) ≈ collect(∇conv_filter(da, dc, cdims))

    # Test for agreement between CPU NNlib and CuDNN versions, across a variety of kwargs
    for num_spatial_dims in (1, 2, 3)
        # Initialize data we'll run our tests over
        C_in = 3
        C_out = 4
        batch_size = 1
        x = rand(Float64, fill(8, num_spatial_dims)..., C_in, batch_size)
        w = rand(Float64, fill(2, num_spatial_dims)..., C_in, C_out)
        b = rand(Float64, fill(1, num_spatial_dims)..., C_in, C_out)
        options = (Dict(), Dict(:dilation => 2), Dict(:flipkernel => true), Dict(:stride => 2), Dict(:padding => 1))

        # @denizyuret: algo option deprecated for nnlib, handling in cudnn
        # algos = (1, 0, 1, 1,)
        # for (opts, algo) in zip(options, algos)

        for opts in options  
            cdims = DenseConvDims(x, w; opts...)
            y = NNlib.conv(x, w, cdims)

            # Test that basic convolution is equivalent across GPU/CPU
            gputest((x, w) -> NNlib.conv(x, w, cdims), x, w)
            gputest((y, w) -> NNlib.∇conv_data(y, w, cdims), y, w)
            gputest((x, y) -> NNlib.∇conv_filter(x, y, cdims), x, y, checkgrad=false) # TODO fix grad

            # Scaling factors
            gputest((x, w) -> NNlib.conv(x, w, cdims; alpha=2.0), x, w, checkgrad=false) # TODO
            gputest((y, w) -> NNlib.∇conv_data(y, w, cdims; alpha=2.0), y, w, checkgrad=false) # TODO
            gputest((x, y) -> NNlib.∇conv_filter(x, y, cdims; alpha=2.0), x, y, checkgrad=false) # TODO
            
            gputest((y, x, w) -> NNlib.conv!(copy(y), x, w, cdims; beta=2.0), y, x, w, checkgrad=false) # TODO
            # @test_broken gputest((x, y, w) -> NNlib.∇conv_data!(copy(x), y, w, cdims; beta=2.0), x, y, w, checkgrad=false) #TODO
            gputest((w, x, y) -> NNlib.∇conv_filter!(copy(w), x, y, cdims; beta=2.0), w, x, y, checkgrad=false) # TODO
        end

        # CPU implementation of ∇conv_bias!
        db = zeros(Float64, 1, 1, 3, 1)
        dy = randn(Float64, 8, 8, 3, 1)
        function NNlibCUDA.∇conv_bias!(db, dy)
            db .= sum(dy, dims=1:(ndims(dy)-2))
            return db
        end
        gputest(NNlibCUDA.∇conv_bias!, db, dy, checkgrad=false)
    end
end
