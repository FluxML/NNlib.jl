using NNlib, Test
using NNlib: input_size, kernel_size, channels_in, channels_out, channel_multiplier,
             stride, padding, dilation, flipkernel, output_size

@testset "ConvDims" begin
    for T in (DenseConvDims, DepthwiseConvDims)
        @testset "$(T)" begin
            x = randn(5,4,3,2)

            if T == DenseConvDims
                w = randn(1,2,3,4)
            elseif T == DepthwiseConvDims
                w = randn(1,2,4,3)
            end
            
            # First, getters:
            cdims = T(x, w)
            @test input_size(cdims) == size(x)[1:2]
            @test kernel_size(cdims) == size(w)[1:2]
            @test channels_in(cdims) == size(x, 3)
            @test stride(cdims) == (1,1)
            @test dilation(cdims) == (1,1)
            @test padding(cdims) == (0,0,0,0)
            @test flipkernel(cdims) == false
            @test output_size(cdims) == (5,3)

            # Special-case channel output tests
            if T == DenseConvDims
                @test channels_out(cdims) == size(w, 4)
            elseif T == DepthwiseConvDims
                @test channel_multiplier(cdims) == size(w, 3)
                @test channels_out(cdims) == size(w,3)*size(w,4)
            end

            # Next, scalar settings:
            cdims = T(x, w; stride=2, dilation=2, padding=3, flipkernel=true)
            @test stride(cdims) == (2,2)
            @test dilation(cdims) == (2,2)
            @test padding(cdims) == (3,3,3,3)
            @test flipkernel(cdims) == true
            @test output_size(cdims) == (6,4)
            
            # Next, tuple settings
            cdims = T(x, w; stride=(1, 2), dilation=(1, 2), padding=(0,1))
            @test stride(cdims) == (1,2)
            @test dilation(cdims) == (1,2)
            @test padding(cdims) == (0,0,1,1)
            @test output_size(cdims) == (5,2)

            # Special case for 4-d padding spec:
            cdims = T(x, w; padding=(1,2,3,4))
            @test padding(cdims) == (1,2,3,4)
            @test output_size(cdims) == (8,10)

            # Make sure we throw on invalid settings:
            # Invalid dimensionality of settings:
            @test_throws DimensionMismatch T(x, w; stride=(1,))
            @test_throws DimensionMismatch T(x, w; stride=(1, 1, 1))
            @test_throws DimensionMismatch T(x, w; padding=(1, 1, 1))
            @test_throws DimensionMismatch T(x, w; padding=(1, 1, 1, 1, 1))
            @test_throws DimensionMismatch T(x, w; dilation=(1,))
            @test_throws DimensionMismatch T(x, w; dilation=(1, 1, 1))
            # Dilation will cause us to reach beyond the end of input + padding here:
            @test_throws DimensionMismatch T(x, w; dilation=(1, 5))
            # Channel mismatch:
            if T == DenseConvDims
                @test_throws DimensionMismatch T(x, w[:,:,1:1,:])
            elseif T == DepthwiseConvDims
                @test_throws DimensionMismatch T(x, w[:,:,:,1:1])
            end
        end
    end
end

conv_answer_dict = Dict(
    # Known-good answers for 1d convolution operations
    1 => Dict(
        "y_pad"  => [1, 4,  7, 10, 13, 10.],
        "y_dil"  => [5, 8, 11.],
        "y_flip" => [5, 8, 11, 14.],

        "dx"        => [ 8, 18, 27, 36, 13.],
        "dx_stride" => [ 8,  4, 20, 10,  0.],
        "dx_pad"    => [ 9, 18, 27, 36, 33.],
        "dx_dil"    => [10, 16, 27,  8, 11.],
        "dx_flip"   => [ 5, 18, 27, 36, 28.],

        "dw"        => [134, 100.],
        "dw_stride" => [ 48,  34.],
        "dw_pad"    => [135, 150.],
        "dw_dil"    => [102,  54.],
        "dw_flip"   => [110, 148.],
    ),

    # Known-good answers for 2d convolution operations
    2 => Dict(
        "y_pad" => [
            1  9  29  49  48;
            4 29  79 129 115;
            7 39  89 139 122;
            10 49  99 149 129;
            13 59 109 159 136;
            10 40  70 100  80.
        ],
        "y_dil" => [
            48   98;
            58  108;
            68  118.
        ],
        "y_flip" => [
            51  101  151;
            61  111  161;
            71  121  171;
            81  131  181.
        ],

        "dx" => [
            116  374   674  258;
            243  700  1200  407;
            313  800  1300  437;
            383  900  1400  467;
            177  386   586  159.
        ],
        "dx_stride" => [
            116  58  516  258;
            87  29  387  129;
            196  98  596  298;
            147  49  447  149;
            0   0    0    0.
        ],
        "dx_pad" => [
            152  470   850   911;
            261  700  1200  1240;
            340  800  1300  1319;
            419  900  1400  1398;
            370  746  1126  1087.
        ],
        "dx_dil" => [
            192  392   96  196;
            232  432  116  216;
            416  766  184  334;
            174  324   58  108;
            204  354   68  118.
        ],
        "dx_flip" => [
            51  254   454   453;
            163  700  1200  1087;
            193  800  1300  1157;
            223  900  1400  1227;
            162  586   886   724.
        ],

        "dw" => [
            17378  11738;
            16250  10610.
        ],
        "dw_stride" => [
            5668  3888;
            5312  3532.
        ],
        "dw_pad" => [
            18670  22550;
            19850  23430.
        ],
        "dw_dil" => [
            8632  3652;
            7636  2656.
        ],
        "dw_flip" => [
            12590  19550;
            13982  20942.
        ],
    ),

    # Known-good answers for 3d convolution operations (these are getting rather large)
    3 => Dict(
        "y_pad"  => reshape([
            1, 4, 7, 10, 13, 10, 9, 29, 39, 49, 59, 40, 29, 79, 89, 99, 109, 70, 49, 129,
            139, 149, 159, 100, 48, 115, 122, 129, 136, 80, 26, 80, 94, 108, 122, 80, 126,
            322, 358, 394, 430, 260, 206, 502, 538, 574, 610, 360, 286, 682, 718, 754, 790,
            460, 220, 502, 524, 546, 568, 320, 146, 360, 374, 388, 402, 240, 446, 1042, 1078,
            1114, 1150, 660, 526, 1222, 1258, 1294, 1330, 760, 606, 1402, 1438, 1474, 1510,
            860, 420, 942, 964, 986, 1008, 560, 205, 456, 467, 478, 489, 270, 517, 1133, 1159,
            1185, 1211, 660, 577, 1263, 1289, 1315, 1341, 730, 637, 1393, 1419, 1445, 1471,
            800, 392, 847, 862, 877, 892, 480.
        ], (6,5,4)),
        "y_dil"  => reshape([608, 644, 680, 788, 824, 860.], (3,2,1)),
        "y_flip" => reshape([
            686, 722, 758, 794, 866, 902, 938, 974, 1046, 1082, 1118, 1154, 1406, 1442,
            1478, 1514, 1586, 1622, 1658, 1694, 1766, 1802, 1838, 1874.
        ], (4,3,2)),

        "dx"        => reshape([
            2576, 5118, 5658, 6198, 3010, 5948, 11576, 12512, 13448, 6420, 8468, 16256,
            17192, 18128, 8580, 4092, 7718, 8114, 8510, 3950, 9624, 18316, 19108, 19900,
            9340, 18680, 34992, 36288, 37584, 17320, 22280, 41472, 42768, 44064, 20200,
            9776, 17756, 18260, 18764, 8340, 4168, 7438, 7690, 7942, 3450, 6972, 11896,
            12256, 12616, 5140, 8052, 13696, 14056, 14416, 5860, 2804, 4278, 4386, 4494,
            1510.
        ], (5,4,3)),
        "dx_stride" => reshape([
            2576, 2254, 3152, 2758, 0, 1932, 1610, 2364, 1970, 0, 5456, 4774, 6032,
            5278, 0, 4092, 3410, 4524, 3770, 0, 1288, 966, 1576, 1182, 0, 644, 322,
            788, 394, 0, 2728, 2046, 3016, 2262, 0, 1364, 682, 1508, 754, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.
        ], (5,4,3)),
        "dx_pad"    => reshape([
            4220, 6343, 7116, 7889, 6550, 8490, 12276, 13312, 14348, 11606, 12350,
            17456, 18492, 19528, 15546, 11989, 16664, 17469, 18274, 14333, 16200,
            22628, 23616, 24604, 19392, 25336, 34992, 36288, 37584, 29320, 30216,
            41472, 42768, 44064, 34200, 26236, 35664, 36652, 37640, 28940, 22816,
            30831, 31636, 32441, 24794, 32522, 43668, 44704, 45740, 34742, 36462,
            48848, 49884, 50920, 38602, 29501, 39264, 40037, 40810, 30733.
        ], (5,4,3)),
        "dx_dil"    => reshape([
            4864, 5152, 9696, 4508, 4760, 6304, 6592, 12396, 5768, 6020, 3648,
            3864, 7120, 3220, 3400, 4728, 4944, 9100, 4120, 4300, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2432, 2576, 4544, 1932, 2040, 
            3152, 3296, 5804, 2472, 2580, 1216, 1288, 1968, 644, 680, 1576, 1648,
            2508, 824, 860.
        ], (5,4,3)),
        "dx_flip"   => reshape([
            686, 2094, 2202, 2310, 1588, 2924, 7544, 7904, 8264, 5124, 3644, 9344,
            9704, 10064, 6204, 3138, 7430, 7682, 7934, 4616, 4836, 11980, 12484,
            12988, 7792, 14936, 34992, 36288, 37584, 21640, 17816, 41472, 42768,
            44064, 25240, 12620, 28412, 29204, 29996, 16728, 7030, 15646, 16042,
            16438, 9084, 17772, 38968, 39904, 40840, 22276, 19932, 43648, 44584,
            45520, 24796, 12362, 26742, 27282, 27822, 14992.
        ], (5,4,3)),

        "dw"        => reshape([1.058184e6, 1.0362e6,    948264,    926280,
                                    618504,   596520,    508584,    486600], (2,2,2)),
        "dw_stride" => reshape([    74760,     72608,     64000,     61848,
                                    31720,     29568,     20960,     18808.], (2,2,2)),
        "dw_pad"    => reshape([1.26055e6, 1.30805e6, 1.40327e6, 1.44923e6,
                                1.73731e6, 1.77589e6, 1.83259e6, 1.86731e6], (2,2,2)),
        "dw_dil"    => reshape([   250320,    241512,    206280,    197472,
                                    74160,     65352,     30120,     21312.], (2,2,2)),
        "dw_flip"   => reshape([    639480,   670200,    793080,    823800,
                                    1.25388e6, 1.2846e6, 1.40748e6,  1.4382e6], (2,2,2)),
    ),
)

@testset "Dense Convolution" begin
    # Start with some easy-to-debug cases that we have worked through and _know_ work
    for rank in (1,2,3)
        @testset "conv$(rank)d" begin
            # Pull out known-good answers for y = conv(x, w)
            y_pad = conv_answer_dict[rank]["y_pad"]
            y_dil = conv_answer_dict[rank]["y_dil"]
            y_flip = conv_answer_dict[rank]["y_flip"]

            # We can always derive y_plain and y_stride from the other answers.
            y_plain = y_pad[((2:(size(y_pad,idx)-1)) for idx in 1:rank)...]
            y_stride = y_pad[((2:2:(size(y_pad,idx)-1)) for idx in 1:rank)...]

            # Same for dx and dw:
            dx = conv_answer_dict[rank]["dx"]
            dx_stride = conv_answer_dict[rank]["dx_stride"]
            dx_pad = conv_answer_dict[rank]["dx_pad"]
            dx_dil = conv_answer_dict[rank]["dx_dil"]
            dx_flip = conv_answer_dict[rank]["dx_flip"]

            dw = conv_answer_dict[rank]["dw"]
            dw_stride = conv_answer_dict[rank]["dw_stride"]
            dw_pad = conv_answer_dict[rank]["dw_pad"]
            dw_dil = conv_answer_dict[rank]["dw_dil"]
            dw_flip = conv_answer_dict[rank]["dw_flip"]

            # We generate x and w from the shapes we know they must be
            x = reshape(Float64[1:prod(size(dx));], size(dx)..., 1, 1)
            w = reshape(Float64[1:prod(size(dw));], size(dw)..., 1, 1)

            # A "drop channels and batch dimension" helper
            ddims(x) = dropdims(x, dims=(rank+1, rank+2))
            
            for conv in (NNlib.conv, NNlib.conv_im2col, NNlib.conv_direct)
                @testset "$(conv)" begin
                    # First, your basic convolution with no parameters
                    cdims = DenseConvDims(x, w)
                    @test ddims(conv(x, w, cdims)) == y_plain

                    # Next, test convolution on views and alternate datatypes:
                    @test ddims(conv(view(x, repeat([:], ndims(x))...), w, cdims)) == y_plain
                    @test ddims(conv(Float32.(x), Float32.(w), cdims)) == Float32.(y_plain)

                    # Next, introduce stride:
                    cdims = DenseConvDims(x, w; stride=2)
                    @test ddims(conv(x, w, cdims)) == y_stride

                    # Next, introduce dilation:
                    cdims = DenseConvDims(x, w; dilation=2)
                    @test ddims(conv(x, w, cdims)) == y_dil

                    # Next, introduce padding:
                    cdims = DenseConvDims(x, w; padding=1)
                    @test ddims(conv(x, w, cdims)) == y_pad

                    # Next, test crosscor/conv with a flipped kernel
                    cdims = DenseConvDims(x, w; flipkernel=true)
                    @test ddims(conv(x, w, cdims)) == y_flip
                end
            end

            # Test all implementations/interfaces
            for (∇conv_filter, ∇conv_data) in (
                    (NNlib.∇conv_filter,        NNlib.∇conv_data),
                    (NNlib.∇conv_filter_im2col, NNlib.∇conv_data_im2col),
                    (NNlib.∇conv_filter_direct, NNlib.∇conv_data_direct),
                )
                @testset "$(∇conv_filter)/$(∇conv_data)" begin
                    # First, your basic convolution with no parameters
                    cdims = DenseConvDims(x, w)
                    dy = NNlib.conv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx

                    # Next, test convolution on views and alternate datatypes:
                    @test ddims(∇conv_filter(x, view(dy, repeat([:], ndims(dy))...), cdims)) == dw
                    @test ddims(∇conv_data(view(dy, repeat([:], ndims(dy))...), w,   cdims)) == dx

                    @test ddims(∇conv_filter(Float32.(x), Float32.(dy), cdims)) == dw
                    @test ddims(∇conv_data(Float32.(dy),  Float32.(w),  cdims)) == dx

                    # Next, introduce stride:
                    cdims = DenseConvDims(x, w; stride=2)
                    dy = NNlib.conv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_stride
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_stride

                    # Next, introduce dilation:
                    cdims = DenseConvDims(x, w; dilation=2)
                    dy = NNlib.conv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_dil
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_dil

                    # Next, introduce padding:
                    cdims = DenseConvDims(x, w; padding=1)
                    dy = NNlib.conv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_pad
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_pad

                    # Next, test crosscor/conv with a flipped kernel
                    cdims = DenseConvDims(x, w; flipkernel=true)
                    dy = NNlib.conv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_flip
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_flip
                end
            end
        end
    end

    @testset "fuzzing" begin
        if get(ENV,"NNLIB_TEST_FUZZING","false") != "true"
            @info("Skipping Convolutional fuzzing tests, set NNLIB_TEST_FUZZING=true to run them")
            return
        end
        @info("Starting Convolutional fuzzing tests; this can take a few minutes...")
        # Now that we're fairly certain things are working, let's fuzz things a little bit:
        for x_size in (
                # 1d tests
                (1,), (3,), (7,),
                # 2d tests
                (1, 3), (3, 3), (12, 3), (20, 17),
                # 3d tests
                (1, 1, 3), (3, 5, 4), (20, 17, 14),
            ),
            C_in in (1, 3),
            batch in (1, 5)

            # Allocate x in this outer loop to save on allocations and speed things up
            x = rand(x_size..., C_in, batch)
            dx_direct = similar(x)
            dx_im2col = similar(x)

            for w_size in (
                    (1,), (3,), (7,),
                    (1,1), (1,3), (3,4), (7, 4),
                    (1,1,1), (1,1,3,), (3,4,3), (7,3,2)),
                C_out in (1, 4)

                # Give some output to the user that something is in fact happening.
                print(".")

                # Allocate w in this outer loop to save on allocations and speed things up
                w = rand(w_size..., C_in, C_out)
                dw_direct = similar(w)
                dw_im2col = similar(w)

                for S_size in (1, 2, 4, (1,2), (4,1), (2,1,4)),
                    P_size in (0, 1, 2, (0,3,0,3), (4,1,4,2), (1,2,3,4,5,6)),
                    D_size in (1, 2, 4, (1,2), (3,2), (4,2,3))

                    # Skip tests that are impossible due to mismatched sizes
                    try    
                        DenseConvDims(x, w;
                            stride=S_size, padding=P_size, dilation=D_size,
                        )
                    catch e
                        if isa(e, DimensionMismatch) || isa(e, MethodError)
                            continue
                        end
                        rethrow(e)
                    end

                    # Do the actual convolution, comparing convolution implementations
                    cdims = DenseConvDims(x, w; stride=S_size, padding=P_size, dilation=D_size)

                    # We use mutating calls with explicitly different initial values, so as
                    # to be sure to catch when we're leaving pieces of the output untouched.
                    y_direct = ones(output_size(cdims)..., C_out, batch) .* 666.666
                    y_im2col = ones(output_size(cdims)..., C_out, batch) .* 777.777

                    # Do the convolutions
                    NNlib.conv_direct!(y_direct, x, w, cdims)
                    NNlib.conv_im2col!(y_im2col, x, w, cdims)

                    # Compare!
                    @test y_direct ≈ y_im2col
                    dy = y_im2col

                    # Now push backwards; first for the filter.  Again, we initialize our
                    # memory so that segments that never get touched are immediately noticable
                    fill!(dw_direct, 666.666)
                    fill!(dw_im2col, 777.777)
                    NNlib.∇conv_filter_direct!(dw_direct, x, dy, cdims)
                    NNlib.∇conv_filter_im2col!(dw_im2col, x, dy, cdims)
                    @test dw_direct ≈ dw_im2col

                    # And then for the input
                    fill!(dx_direct, 666.666)
                    fill!(dx_im2col, 777.777)
                    NNlib.∇conv_data_direct!(dx_direct, dy, w, cdims)
                    NNlib.∇conv_data_im2col!(dx_im2col, dy, w, cdims)
                    @test dx_direct ≈ dx_im2col
                end
            end
        end
        println()
    end
end


@testset "Depthwise Convolution" begin
    # Start with some easy-to-debug cases that we have worked through and _know_ work
    for rank in (1,) #2,3)
        @testset "depthwiseconv$(rank)d" begin
            # Pull out known-good answers for y = depthwiseconv(x, w)
            y_pad = conv_answer_dict[rank]["y_pad"]
            y_dil = conv_answer_dict[rank]["y_dil"]
            y_flip = conv_answer_dict[rank]["y_flip"]

            # We can always derive y_plain and y_stride from the other answers.
            y_plain = y_pad[((2:(size(y_pad,idx)-1)) for idx in 1:rank)...]
            y_stride = y_pad[((2:2:(size(y_pad,idx)-1)) for idx in 1:rank)...]

            # Same for dx and dw:
            dx = conv_answer_dict[rank]["dx"]
            dx_stride = conv_answer_dict[rank]["dx_stride"]
            dx_pad = conv_answer_dict[rank]["dx_pad"]
            dx_dil = conv_answer_dict[rank]["dx_dil"]
            dx_flip = conv_answer_dict[rank]["dx_flip"]

            dw = conv_answer_dict[rank]["dw"]
            dw_stride = conv_answer_dict[rank]["dw_stride"]
            dw_pad = conv_answer_dict[rank]["dw_pad"]
            dw_dil = conv_answer_dict[rank]["dw_dil"]
            dw_flip = conv_answer_dict[rank]["dw_flip"]

            # We generate x and w from the shapes we know they must be
            x = reshape(Float64[1:prod(size(dx));], size(dx)..., 1, 1)
            w = reshape(Float64[1:prod(size(dw));], size(dw)..., 1, 1)

            # A "drop channels and batch dimension" helper
            ddims(x) = dropdims(x, dims=(rank+1, rank+2))
            
            for conv in (NNlib.depthwiseconv, NNlib.depthwiseconv_im2col, NNlib.depthwiseconv_direct)
                @testset "$(conv)" begin
                    # First, your basic convolution with no parameters
                    cdims = DepthwiseConvDims(x, w)
                    @test ddims(conv(x, w, cdims)) == y_plain

                    # Next, test convolution on views and alternate datatypes:
                    @test ddims(conv(view(x, repeat([:], ndims(x))...), w, cdims)) == y_plain
                    @test ddims(conv(Float32.(x), Float32.(w), cdims)) == Float32.(y_plain)

                    # Next, introduce stride:
                    cdims = DepthwiseConvDims(x, w; stride=2)
                    @test ddims(conv(x, w, cdims)) == y_stride

                    # Next, introduce dilation:
                    cdims = DepthwiseConvDims(x, w; dilation=2)
                    @test ddims(conv(x, w, cdims)) == y_dil

                    # Next, introduce padding:
                    cdims = DepthwiseConvDims(x, w; padding=1)
                    @test ddims(conv(x, w, cdims)) == y_pad

                    # Next, test crosscor/conv with a flipped kernel
                    cdims = DepthwiseConvDims(x, w; flipkernel=true)
                    @test ddims(conv(x, w, cdims)) == y_flip
                end
            end

            # Test all implementations/interfaces
            for (∇conv_filter, ∇conv_data) in (
                    (NNlib.∇depthwiseconv_filter,        NNlib.∇depthwiseconv_data),
                    (NNlib.∇depthwiseconv_filter_im2col, NNlib.∇depthwiseconv_data_im2col),
                    (NNlib.∇depthwiseconv_filter_direct, NNlib.∇depthwiseconv_data_direct),
                )
                @testset "$(∇conv_filter)/$(∇conv_data)" begin
                    # First, your basic convolution with no parameters
                    cdims = DepthwiseConvDims(x, w)
                    dy = NNlib.depthwiseconv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx

                    # Next, test convolution on views and alternate datatypes:
                    @test ddims(∇conv_filter(x, view(dy, repeat([:], ndims(dy))...), cdims)) == dw
                    @test ddims(∇conv_data(view(dy, repeat([:], ndims(dy))...), w,   cdims)) == dx

                    @test ddims(∇conv_filter(Float32.(x), Float32.(dy), cdims)) == dw
                    @test ddims(∇conv_data(Float32.(dy),  Float32.(w),  cdims)) == dx

                    # Next, introduce stride:
                    cdims = DepthwiseConvDims(x, w; stride=2)
                    dy = NNlib.depthwiseconv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_stride
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_stride

                    # Next, introduce dilation:
                    cdims = DepthwiseConvDims(x, w; dilation=2)
                    dy = NNlib.depthwiseconv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_dil
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_dil

                    # Next, introduce padding:
                    cdims = DepthwiseConvDims(x, w; padding=1)
                    dy = NNlib.depthwiseconv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_pad
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_pad

                    # Next, test crosscor/conv with a flipped kernel
                    cdims = DepthwiseConvDims(x, w; flipkernel=true)
                    dy = NNlib.depthwiseconv(x, w, cdims)
                    @test ddims(∇conv_filter(x, dy, cdims)) == dw_flip
                    @test ddims(∇conv_data(dy, w,  cdims)) == dx_flip
                end
            end
        end
    end

    @testset "fuzzing" begin
        if get(ENV,"NNLIB_TEST_FUZZING","false") != "true"
            @info("Skipping Depthwise Convolutional fuzzing tests, set NNLIB_TEST_FUZZING=true to run them")
            return
        end
        @info("Starting Depthwise Convolutional fuzzing tests; this can take a few minutes...")
        # Now that we're fairly certain things are working, let's fuzz things a little bit:
        for x_size in (
                # 1d tests
                (1,), (3,), (7,),
                # 2d tests
                (1, 3), (3, 3), (12, 3), (20, 17),
                # 3d tests
                (1, 1, 3), (3, 5, 4), (20, 17, 14),
            ),
            C_in in (1, 3),
            batch in (1, 5)

            # Allocate x in this outer loop to save on allocations and speed things up
            x = rand(x_size..., C_in, batch)
            dx_direct = similar(x)
            dx_im2col = similar(x)

            for w_size in (
                    (1,), (3,), (7,),
                    (1,1), (1,3), (3,4), (7, 4),
                    (1,1,1), (1,1,3,), (3,4,3), (7,3,2)),
                C_mult in (1, 4)

                # Give some output to the user that something is in fact happening.
                print(".")

                # Allocate w in this outer loop to save on allocations and speed things up
                w = rand(w_size..., C_mult, C_in)
                dw_direct = similar(w)
                dw_im2col = similar(w)

                for S_size in (1, 2, 4, (1,2), (4,1), (2,1,4)),
                    P_size in (0, 1, 2, (0,3,0,3), (4,1,4,2), (1,2,3,4,5,6)),
                    D_size in (1, 2, 4, (1,2), (3,2), (4,2,3))

                    # Skip tests that are impossible due to mismatched sizes
                    try    
                        DepthwiseConvDims(x, w;
                            stride=S_size, padding=P_size, dilation=D_size,
                        )
                    catch e
                        if isa(e, DimensionMismatch) || isa(e, MethodError)
                            continue
                        end
                        rethrow(e)
                    end

                    # Do the actual convolution, comparing convolution implementations
                    cdims = DepthwiseConvDims(x, w; stride=S_size, padding=P_size, dilation=D_size)

                    # We use mutating calls with explicitly different initial values, so as
                    # to be sure to catch when we're leaving pieces of the output untouched.
                    y_direct = ones(output_size(cdims)..., channels_out(cdims), batch) .* 666.666
                    y_im2col = ones(output_size(cdims)..., channels_out(cdims), batch) .* 777.777

                    # Do the convolutions
                    NNlib.depthwiseconv_direct!(y_direct, x, w, cdims)
                    NNlib.depthwiseconv_im2col!(y_im2col, x, w, cdims)

                    # Compare!
                    @test y_direct ≈ y_im2col
                    dy = y_im2col

                    # Now push backwards; first for the filter.  Again, we initialize our
                    # memory so that segments that never get touched are immediately noticable
                    fill!(dw_direct, 666.666)
                    fill!(dw_im2col, 777.777)
                    NNlib.∇depthwiseconv_filter_direct!(dw_direct, x, dy, cdims)
                    NNlib.∇depthwiseconv_filter_im2col!(dw_im2col, x, dy, cdims)
                    @test dw_direct ≈ dw_im2col

                    # And then for the input
                    fill!(dx_direct, 666.666)
                    fill!(dx_im2col, 777.777)
                    NNlib.∇depthwiseconv_data_direct!(dx_direct, dy, w, cdims)
                    NNlib.∇depthwiseconv_data_im2col!(dx_im2col, dy, w, cdims)
                    @test dx_direct ≈ dx_im2col
                end
            end
        end
        println()
    end
end