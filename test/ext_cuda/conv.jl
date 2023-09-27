using NNlib: DenseConvDims

@testset "convolution" begin
@testset "$T" for T in (Float64, ComplexF64)
    a, b, c = rand(T, 10, 10, 3, 1), rand(T, 2, 2, 3, 4), rand(T, 9, 9, 4, 1)
    da, db, dc = CuArray(a), CuArray(b), CuArray(c)
    cdims = DenseConvDims(a, b)
    @test NNlib.conv(a, b, cdims) ≈ collect(NNlib.conv(da, db, cdims))
    @test ∇conv_data(c, b, cdims) ≈ collect(∇conv_data(dc, db, cdims))
    @test ∇conv_filter(a, c, cdims) ≈ collect(∇conv_filter(da, dc, cdims))

    if T <: Complex
        @testset "mixed real and complex" begin
            @test NNlib.conv(real(a), b, cdims) ≈ collect(NNlib.conv(real(da), db, cdims))
            @test NNlib.conv(a, real(b), cdims) ≈ collect(NNlib.conv(da, real(db), cdims))
            @test ∇conv_data(c, real(b), cdims) ≈ collect(∇conv_data(dc, real(db), cdims))
            @test ∇conv_filter(real(a), c, cdims) ≈ collect(∇conv_filter(real(da), dc, cdims))
        end
    end

    # Test Conv Bias Activation
    bias = rand(T, 1, 1, 4, 1)
    dbias = CuArray(bias)
    act = T <: Complex ? abs2 : NNlib.relu 
    @test conv_bias_act(a, b, cdims, bias, act) ≈ collect(conv_bias_act(da, db, cdims, dbias, act))
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

    # we use this activation for the gpu tests
    # as we can't take gradients of complex quantities
    act = T <: Complex ? x-> abs2(x) : identity
    @testset "groups=$groups, num_spatial_dims=$num_spatial_dims" for groups in (1, 2, 4), num_spatial_dims in (1, 2, 3)
        # Make `C_in = C_out` when using grouped convolution.
        C_in = groups == 1 ? C_in_ : C_out
        # Initialize data we'll run our tests over
        x = rand(T, fill(8, num_spatial_dims)..., C_in, batch_size)
        w = rand(T, fill(2, num_spatial_dims)..., C_in ÷ groups, C_out)

        @testset "opts #$i" for (i,opts) in enumerate(options)
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
            @testset "cpu==gpu" begin
                @testset "conv" begin
                    gputest((x, w) -> act.(NNlib.conv(x, w, cdims)), x, w)
                    if T <: Complex
                        gputest((x, w) -> act.(NNlib.conv(x, w, cdims)), real(x), w)
                        gputest((x, w) -> act.(NNlib.conv(x, w, cdims)), x, real(w))
                    end
                end
                @testset "∇conv_data" begin
                    gputest((y, w) -> act.(NNlib.∇conv_data(y, w, cdims)), y, w)
                    if T <: Complex
                        gputest((y, w) -> act.(NNlib.∇conv_data(y, w, cdims)), y, real(w))
                    end
                end
                @testset "∇conv_filter" begin
                    gputest((x, y) -> act.(NNlib.∇conv_filter(x, y, cdims)), x, y) 
                    if T <: Complex
                        gputest((x, y) -> act.(NNlib.∇conv_filter(x, y, cdims)), real(x), y)
                    end
                end
            end

            # Scaling factors
            @testset "scale-alpha" begin
                gputest((x, w) -> act.(NNlib.conv(x, w, cdims; alpha=T(2.0))), x, w, checkgrad=false) # TODO
                gputest((y, w) -> act.(NNlib.∇conv_data(y, w, cdims; alpha=T(2.0))), y, w, checkgrad=false) # TODO
                gputest((x, y) -> act.(NNlib.∇conv_filter(x, y, cdims; alpha=T(2.0))), x, y, checkgrad=false) # TODO 

                if T <: Complex
                    gputest((x, w) -> act.(NNlib.conv(x, w, cdims; alpha=T(2.0))), real(x), w, checkgrad=false) 
                    gputest((x, w) -> act.(NNlib.conv(x, w, cdims; alpha=T(2.0))), x, real(w), checkgrad=false) # TODO
                    gputest((y, w) -> act.(NNlib.∇conv_data(y, w, cdims; alpha=T(2.0))), y, real(w), checkgrad=false) # TODO
                    gputest((x, y) -> act.(NNlib.∇conv_filter(x, y, cdims; alpha=T(2.0))), real(x), y, checkgrad=false) # TODO
                end
            end

            @testset "scale-beta" begin
                gputest((y, x, w) -> act.(NNlib.conv!(copy(y), x, w, cdims; beta=T(2.0))), y, x, w, checkgrad=false, broken=false)
                gputest((w, x, y) -> act.(NNlib.∇conv_filter!(copy(w), x, y, cdims; beta=T(2.0))), w, x, y, checkgrad=false, broken=false) 
                gputest((x, y, w) -> act.(NNlib.∇conv_data!(copy(x), y, w, cdims; beta=T(2.0))), x, y, w, checkgrad=false, broken=false) 

                if T <: Complex
                    gputest((y, x, w) -> act.(NNlib.conv!(copy(y), x, w, cdims; beta=T(2.0))), y, real(x), w, checkgrad=false) 
                    gputest((y, x, w) -> act.(NNlib.conv!(copy(y), x, w, cdims; beta=T(2.0))), y, x, real(w), checkgrad=false) 
                    gputest((x, y, w) -> act.(NNlib.∇conv_data!(copy(x), y, w, cdims; beta=T(2.0))), x, y, real(w), checkgrad=false) 
                    gputest((w, x, y) -> act.(NNlib.∇conv_filter!(copy(w), x, y, cdims; beta=T(2.0))), w, real(x), y, checkgrad=false)
                end
            end

        end
    end
end
end
