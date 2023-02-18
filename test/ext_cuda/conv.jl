using NNlib: DenseConvDims

@testset "convolution" begin
    a, b, c = rand(Float64, 10, 10, 3, 1), rand(Float64, 2, 2, 3, 4), rand(Float64, 9, 9, 4, 1)
    da, db, dc = CuArray(a), CuArray(b), CuArray(c)
    cdims = DenseConvDims(a, b)
    @test NNlib.conv(a, b, cdims) ≈ collect(NNlib.conv(da, db, cdims))
    @test ∇conv_data(c, b, cdims) ≈ collect(∇conv_data(dc, db, cdims))
    @test ∇conv_filter(a, c, cdims) ≈ collect(∇conv_filter(da, dc, cdims))

    # Test Conv Bias Activation
    bias = rand(Float64, 1, 1, 4, 1)
    dbias = CuArray(bias)
    @test conv_bias_act(a, b, cdims, bias, NNlib.relu) ≈ collect(conv_bias_act(da, db, cdims, dbias, NNlib.relu))
    @test conv_bias_act(a, b, cdims, bias, identity) ≈ collect(conv_bias_act(da, db, cdims, dbias, identity))

    # Test for agreement between CPU NNlib and CuDNN versions, across a variety of kwargs
    options = Dict{Any, Any}.((
        (), (:dilation => 2), (:flipkernel => true), (:stride => 2),
        (:padding => 1),
        (:padding => (1,0)),
        (:padding => (0,1)),
        (:padding => (2,3)),
    ))
    C_in_ = 3
    C_out = 4
    batch_size = 1

    for groups in (1, 2, 4), num_spatial_dims in (1, 2, 3)
        # Make `C_in = C_out` when using grouped convolution.
        C_in = groups == 1 ? C_in_ : C_out
        # Initialize data we'll run our tests over
        x = rand(Float64, fill(8, num_spatial_dims)..., C_in, batch_size)
        w = rand(Float64, fill(2, num_spatial_dims)..., C_in ÷ groups, C_out)

        for opts in options
            opts[:groups] = groups
            
            if :padding in keys(opts)
                padding = opts[:padding]
                if 1 < length(padding) && length(padding) != 2num_spatial_dims
                    opts[:padding] = ntuple(i -> padding[mod1(i,2)] .+ 2div(i-1,2), 2num_spatial_dims)   
                end
            end

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
    end
end
