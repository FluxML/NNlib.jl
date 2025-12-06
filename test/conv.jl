using NNlib, Test
using NNlib: input_size, kernel_size, channels_in, channels_out, channel_multiplier,
             stride, padding, dilation, flipkernel, output_size,
             groupcount
using Random: AbstractRNG, SamplerType

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

# A "drop channels and batch dimension" helper
ddims(x) = dropdims(x, dims=(ndims(x)-1, ndims(x)))

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

            convs = [NNlib.conv, NNlib.conv_im2col, NNlib.conv_direct,]
            for conv in convs
                @testset "$(conv)" begin
                    cdims = DenseConvDims(x, w)
                    # First, your basic convolution with no parameters
                    @test isapprox(ddims(conv(x, w, cdims)), y_plain, rtol = 1.0e-7)

                    # Next, test convolution on views and alternate datatypes:
                    @test isapprox(ddims(conv(view(x, repeat([:], ndims(x))...), w, cdims)), y_plain, rtol = 1.0e-7)
                    @test isapprox(ddims(conv(Float32.(x), Float32.(w), cdims)), Float32.(y_plain), rtol = 1.0e-7)

                    # Next, introduce stride:
                    cdims = DenseConvDims(x, w; stride=2)
                    @test isapprox(ddims(conv(x, w, cdims)), y_stride, rtol = 1.0e-7)

                    # Next, introduce dilation:
                    cdims = DenseConvDims(x, w; dilation=2)
                    @test isapprox(ddims(conv(x, w, cdims)), y_dil, rtol = 1.0e-7)

                    # Next, introduce padding:
                    cdims = DenseConvDims(x, w; padding=1)
                    @test isapprox(ddims(conv(x, w, cdims)), y_pad, rtol = 1.0e-7)

                    # Next, test crosscor/conv with a flipped kernel
                    cdims = DenseConvDims(x, w; flipkernel=true)
                    @test isapprox(ddims(conv(x, w, cdims)), y_flip, rtol = 1.0e-7)
                end
            end

            # Test all in-place implementations/interfaces
            convs = [NNlib.conv!, NNlib.conv_im2col!, NNlib.conv_direct!,]
            for conv! in convs
                α, β = 2e0, -1e0

                @testset "$(conv!)" begin
                    # First, your basic convolution with no parameters
                    cdims = DenseConvDims(x, w)
                    y0 = rand(rng, -9e0:9e0, size(y_plain)..., 1, 1)
                    @test isapprox(ddims(conv!(copy(y0), x, w, cdims; alpha=α, beta=β)), α*y_plain + β*y0, rtol = 1.0e-7)

                    # Next, test convolution on views and alternate datatypes:
                    @test isapprox(ddims(conv!(copy(y0), view(x, repeat([:], ndims(x))...), w, cdims; alpha=α, beta=β)), α*y_plain + β*y0, rtol = 1.0e-7)
                    @test isapprox(ddims(conv!(Float32.(copy(y0)), Float32.(x), Float32.(w), cdims; alpha=Float32(α), beta=Float32(β))), Float32.(α*y_plain + β*y0), rtol = 1.0e-7)

                    # Next, introduce stride:
                    cdims = DenseConvDims(x, w; stride=2)
                    y0 = rand(rng, -9e0:9e0, size(y_stride)..., 1, 1)
                    @test isapprox(ddims(conv!(copy(y0), x, w, cdims; alpha=α, beta=β)), α*y_stride + β*y0, rtol = 1.0e-7)

                    # Next, introduce dilation:
                    cdims = DenseConvDims(x, w; dilation=2)
                    y0 = rand(rng, -9e0:9e0, size(y_dil)..., 1, 1)
                    @test isapprox(ddims(conv!(copy(y0), x, w, cdims; alpha=α, beta=β)), α*y_dil + β*y0, rtol = 1.0e-7)

                    # Next, introduce padding:
                    cdims = DenseConvDims(x, w; padding=1)
                    y0 = rand(rng, -9e0:9e0, size(y_pad)..., 1, 1)
                    @test isapprox(ddims(conv!(copy(y0), x, w, cdims; alpha=α, beta=β)), α*y_pad + β*y0, rtol = 1.0e-7)

                    # Next, test crosscor/conv with a flipped kernel
                    cdims = DenseConvDims(x, w; flipkernel=true)
                    y0 = rand(rng, -9e0:9e0, size(y_flip)..., 1, 1)
                    @test isapprox(ddims(conv!(copy(y0), x, w, cdims; alpha=α, beta=β)), α*y_flip + β*y0, rtol = 1.0e-7)
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
                    @test isapprox(ddims(∇conv_filter(x, dy, cdims)), dw, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data(dy, w,  cdims)), dx, rtol = 1.0e-7)

                    # Next, test convolution on views and alternate datatypes:
                    @test isapprox(ddims(∇conv_filter(x, view(dy, repeat([:], ndims(dy))...), cdims)), dw, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data(view(dy, repeat([:], ndims(dy))...), w,   cdims)), dx, rtol = 1.0e-7)

                    @test isapprox(ddims(∇conv_filter(Float32.(x), Float32.(dy), cdims)), dw, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data(Float32.(dy),  Float32.(w),  cdims)), dx, rtol = 1.0e-7)

                    # Next, introduce stride:
                    cdims = DenseConvDims(x, w; stride=2)
                    dy = NNlib.conv(x, w, cdims)
                    @test isapprox(ddims(∇conv_filter(x, dy, cdims)), dw_stride, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data(dy, w,  cdims)), dx_stride, rtol = 1.0e-7)

                    # Next, introduce dilation:
                    cdims = DenseConvDims(x, w; dilation=2)
                    dy = NNlib.conv(x, w, cdims)
                    @test isapprox(ddims(∇conv_filter(x, dy, cdims)), dw_dil, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data(dy, w,  cdims)), dx_dil, rtol = 1.0e-7)

                    # Next, introduce padding:
                    cdims = DenseConvDims(x, w; padding=1)
                    dy = NNlib.conv(x, w, cdims)
                    @test isapprox(ddims(∇conv_filter(x, dy, cdims)), dw_pad, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data(dy, w,  cdims)), dx_pad, rtol = 1.0e-7)

                    # Next, test crosscor/conv with a flipped kernel
                    cdims = DenseConvDims(x, w; flipkernel=true)
                    dy = NNlib.conv(x, w, cdims)
                    @test isapprox(ddims(∇conv_filter(x, dy, cdims)), dw_flip, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data(dy, w,  cdims)), dx_flip, rtol = 1.0e-7)
                end
            end

            # Test im2col

            for beta in (-2.0, -1.0, 0.0, 0.5, 1.0, 2.0)
                cache_dx, cache_dy, cache_w = ([0.17;;; 0.19;;; 0.23], [0.11;;; 0.13;;; 0.15], [1.0;;;])
                dx_old = copy(cache_dx)
                cdims = DenseConvDims(cache_dx, cache_w)
                NNlib.∇conv_data_im2col!(cache_dx, cache_dy, cache_w, cdims; alpha=1.0, beta)
                @test isapprox(cache_dx, dx_old * beta + cache_dy, rtol = 1.0e-7)
            end

            # Test all in-place implementations/interfaces
            for (∇conv_filter!, ∇conv_data!) in (
                    (NNlib.∇conv_filter!,        NNlib.∇conv_data!),
                    (NNlib.∇conv_filter_im2col!, NNlib.∇conv_data_im2col!),
                    (NNlib.∇conv_filter_direct!, NNlib.∇conv_data_direct!),
                )
                #α, β = 2*rand(rng) - 1, 2*rand(rng) - 1
                α, β = 2e0, -1e0

                @testset "$(∇conv_filter!)/$(∇conv_data!)" begin
                    # First, your basic convolution with no parameters
                    cdims = DenseConvDims(x, w)
                    dy = NNlib.conv(x, w, cdims)
                    @test isapprox(ddims(∇conv_filter!(copy(w), x, dy, cdims; alpha=α, beta=β)), α*dw + β*w, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data!(copy(x), dy, w,   cdims; alpha=α, beta=β)), α*dx + β*x, rtol = 1.0e-7)

                    # Next, test convolution on views and alternate datatypes:
                    @test isapprox(ddims(∇conv_filter!(copy(w), x, view(dy, repeat([:], ndims(dy))...), cdims; alpha=α, beta=β)), α*dw + β*w, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data!(copy(x), view(dy, repeat([:], ndims(dy))...), w,   cdims; alpha=α, beta=β)), α*dx + β*x, rtol = 1.0e-7)

                    @test isapprox(ddims(∇conv_filter!(Float32.(copy(w)), Float32.(x), Float32.(dy), cdims; alpha=Float32(α), beta=Float32(β))), α*dw + β*w, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data!(Float32.(copy(x)), Float32.(dy),  Float32.(w),  cdims; alpha=Float32(α), beta=Float32(β))), α*dx + β*x, rtol = 1.0e-7)

                    # Next, introduce stride:
                    cdims = DenseConvDims(x, w; stride=2)
                    dy = NNlib.conv(x, w, cdims)
                    flag_ = ∇conv_filter! == NNlib.∇conv_filter_direct! && rank in (1,3)
                    @test isapprox(ddims(∇conv_filter!(copy(w), x, dy, cdims; alpha=α, beta=β)), α*dw_stride + β*w, rtol = 1.0e-7) broken=flag_
                    @test isapprox(ddims(∇conv_data!(copy(x), dy, w,   cdims; alpha=α, beta=β)), α*dx_stride + β*x, rtol = 1.0e-7)

                    # Next, introduce dilation:
                    cdims = DenseConvDims(x, w; dilation=2)
                    dy = NNlib.conv(x, w, cdims)
                    flag_ = ∇conv_data! == NNlib.∇conv_data_direct! && rank == 3
                    @test isapprox(ddims(∇conv_filter!(copy(w), x, dy, cdims; alpha=α, beta=β)), α*dw_dil + β*w, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data!(copy(x), dy, w,   cdims; alpha=α, beta=β)), α*dx_dil + β*x, rtol = 1.0e-7) broken=flag_

                    # Next, introduce padding:
                    cdims = DenseConvDims(x, w; padding=1)
                    dy = NNlib.conv(x, w, cdims)
                    @test isapprox(ddims(∇conv_filter!(copy(w), x, dy, cdims; alpha=α, beta=β)), α*dw_pad + β*w, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data!(copy(x), dy, w,   cdims; alpha=α, beta=β)), α*dx_pad + β*x, rtol = 1.0e-7)

                    # Next, test crosscor/conv with a flipped kernel
                    cdims = DenseConvDims(x, w; flipkernel=true)
                    dy = NNlib.conv(x, w, cdims)
                    @test isapprox(ddims(∇conv_filter!(copy(w), x, dy, cdims; alpha=α, beta=β)), α*dw_flip + β*w, rtol = 1.0e-7)
                    @test isapprox(ddims(∇conv_data!(copy(x), dy, w,   cdims; alpha=α, beta=β)), α*dx_flip + β*x, rtol = 1.0e-7)
                end
            end
        end
    end
end

@testset "Complex Dense Convolution" begin
    # For now only 1 dimensional 1x1 convolution
    x = reshape(complex.(Float64[1:4;], Float64[1:4;] .+ 1), 1, 4, 1)
    w = reshape(complex.(Float64[1:4;] .+ 2, Float64[1:4;] .+ 3), 1, 4, 1)
    cdims = DenseConvDims(x, w)
    convs = [NNlib.conv, NNlib.conv_im2col, NNlib.conv_direct,]
    for conv in convs
        @testset "$(conv)" begin
            @test isapprox(ddims(conv(x, w, cdims)), [transpose(vec(w)) * vec(x)], rtol = 1.0e-7)
        end
    end
    dy = NNlib.conv(x, w, cdims)
    for (∇conv_filter, ∇conv_data) in (
        (NNlib.∇conv_filter,        NNlib.∇conv_data),
        (NNlib.∇conv_filter_im2col, NNlib.∇conv_data_im2col),
        (NNlib.∇conv_filter_direct, NNlib.∇conv_data_direct),
    )
        @testset "$(∇conv_filter)/$(∇conv_data)" begin
            @test isapprox(∇conv_filter(x, dy, cdims), conj(x) .* dy, rtol = 1.0e-7)
            @test isapprox(∇conv_data(dy, w, cdims), dy .* conj(w), rtol = 1.0e-7)
        end
    end
end

if get(ENV, "NNLIB_TEST_FUZZING", "false") == "true"
    # @info("Skipping Convolutional fuzzing tests, set NNLIB_TEST_FUZZING=true to run them")
    @testset "fuzzing" begin
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
else
    @info "Skipping Convolutional fuzzing tests, set NNLIB_TEST_FUZZING=true to run them"
end

@testset "Depthwise Convolution" begin
    # Start with some easy-to-debug cases that we have worked through and _know_ work.
    # NOTE: these examples are all single-channel... which doesn't really stress test
    # the important parts of depthwise convolution!
    for rank in (1,2,3)
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

            for conv in (NNlib.depthwiseconv, NNlib.depthwiseconv_im2col, NNlib.depthwiseconv_direct)
                @testset "$(conv)" begin
                    # First, your basic convolution with no parameters
                    cdims = DepthwiseConvDims(x, w)
                    @test ddims(conv(x, w, cdims)) == y_plain

                    # Next, test convolution on views and alternate datatypes:
                    @test isapprox(ddims(conv(view(x, repeat([:], ndims(x))...), w, cdims)), y_plain, rtol = 1.0e-7)
                    @test isapprox(ddims(conv(Float32.(x), Float32.(w), cdims)), Float32.(y_plain), rtol = 1.0e-7)

                    # Next, introduce stride:
                    cdims = DepthwiseConvDims(x, w; stride=2)
                    @test isapprox(ddims(conv(x, w, cdims)), y_stride, rtol = 1.0e-7)

                    # Next, introduce dilation:
                    cdims = DepthwiseConvDims(x, w; dilation=2)
                    @test isapprox(ddims(conv(x, w, cdims)), y_dil, rtol = 1.0e-7)

                    # Next, introduce padding:
                    cdims = DepthwiseConvDims(x, w; padding=1)
                    @test isapprox(ddims(conv(x, w, cdims)), y_pad, rtol = 1.0e-7)

                    # Next, test crosscor/conv with a flipped kernel
                    cdims = DepthwiseConvDims(x, w; flipkernel=true)
                    @test isapprox(ddims(conv(x, w, cdims)), y_flip, rtol = 1.0e-7)
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

    # Do some real depthwise convolution tests
    x = Float64.(reshape(1:2, (1,2,1)))
    w = Float64.(reshape(1:6, (3,1,2)))
    cdims = DepthwiseConvDims(x, w; padding=1)
    for conv in (NNlib.depthwiseconv, NNlib.depthwiseconv_im2col, NNlib.depthwiseconv_direct)
        @test conv(x, w, cdims)[:] ≈ [2, 10]  rtol=1e-7
    end
end


if get(ENV,"NNLIB_TEST_FUZZING","false") == "true"
    @testset "fuzzing" begin
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
else
    @info "Skipping Depthwise Convolutional fuzzing tests, set NNLIB_TEST_FUZZING=true to run them"
end

@testset "Grouped Convolutions" begin
   x′ = rand(Float32, 28, 28, 100, 2)
   w′ = rand(Float32, 3, 3, 20, 15)

   @test_throws DimensionMismatch DenseConvDims(x′, w′)
   cdims = DenseConvDims(x′, w′, groups = 5)

   @test groupcount(cdims) == 5

   y = conv(x′, w′, cdims)
   _, back = Zygote.pullback((x, w) -> sum(conv(x, w, cdims)), x′, w′)
   gs_x, gs_w = back(1.f0)


   ips = Iterators.partition(1:100, 20)
   ops = Iterators.partition(1:15, 3)
   for (i,o) in zip(ips,ops)
      _, back_reg = Zygote.pullback((x, w) -> sum(conv(x, w)), x′[:,:,i,:], w′[:,:,:,o])
      gs_x_reg, gs_w_reg = back_reg(1.f0)
      @test conv(x′[:,:,i,:], w′[:,:,:,o]) ≈ y[:,:,o,:]
      @test gs_x_reg ≈ gs_x[:,:,i,:]
      @test gs_w_reg ≈ gs_w[:,:,:,o]
   end

   # Currently hangs due to a FiniteDifferences issue
   @test_skip gradtest((x, w) -> sum(conv(x, w, cdims)), x′, w′)
end

@testset "conv_wrapper" begin
    x = rand(10, 10, 3, 10)
    w = rand(2, 2, 3, 16)
    w1 = rand(3, 4, 3, 16)
    @test size(conv(x, w)) == (9, 9, 16, 10)
    @test size(conv(x, w; stride = (2, 2), pad = (2, 2))) == (7, 7, 16, 10)
    @test size(conv(x, w1; stride = (1, 2), pad = (2, 3))) == (12, 7, 16, 10)
    @test size(conv(x, w; stride = (1, 2), pad = (2, 3), dilation = (2, 2))) == (12, 7, 16, 10)
    @test size(conv(x, w; stride = (1, 2), pad = (2, 3), dilation = (2, 2), flipped = true)) == (12, 7, 16, 10)
end

# https://github.com/FluxML/NNlib.jl/issues/369
@testset "conv_wrapper with groups - not equal types that trigger direct backend" begin
    x = rand(Float32, 10, 10, 32, 8)
    w = rand(Float64, 2, 2, 16, 4)
    g = 2
    @test conv(x, w; groups=g) ≈ conv(x, Float32.(w); groups=g)
    @test conv(x, w; stride = (2, 2), pad = (2, 2), groups=g) ≈ conv(x, w; stride = (2, 2), pad = (2, 2), groups=g)
    @test conv(x, w; stride = (1, 2), pad = (2, 3), dilation = (2, 2), groups=g) ≈ conv(x, w; stride = (1, 2), pad = (2, 3), dilation = (2, 2), groups=g)
    @test conv(x, w; stride = (1, 2), pad = (2, 3), dilation = (2, 2), flipped = true, groups=g) ≈ conv(x, w; stride = (1, 2), pad = (2, 3), dilation = (2, 2), flipped = true, groups=g)
end

@testset "depthwiseconv_wrapper" begin
    x = rand(10, 10, 3, 10)
    w = rand(2, 2, 3, 3)
    w1 = rand(3, 4, 3, 3)
    @test size(depthwiseconv(x, w)) == (9, 9, 9, 10)
    @test size(depthwiseconv(x, w; stride = (2, 2), pad = (2, 2))) == (7, 7, 9, 10)
    @test size(depthwiseconv(x, w1; stride = (1, 2), pad = (2, 3))) == (12, 7, 9, 10)
    @test size(depthwiseconv(x, w1; stride = (1, 2), pad = (2, 3), dilation = (2, 2))) == (10, 5, 9, 10)
    @test size(depthwiseconv(x, w1; stride = (1, 2), pad = (2, 3), dilation = (2, 2), flipped = true)) == (10, 5, 9, 10)
end

# https://github.com/FluxML/NNlib.jl/pull/171
@testset "conv_direct! - Check Sizes" begin
    x_size = (6, 7, 8, 5, 3)
    y_size = (5, 6, 7, 4, 3)
    w_size = (2, 2, 2, 5, 4)
    x = randn(Float32, x_size);
    y = randn(Float32, y_size);
    w = randn(Float32, w_size);
    cdims = DenseConvDims(x_size, w_size)
    @test size(NNlib.conv_direct!(y, x, w, cdims)) == y_size
    @test size(NNlib.∇conv_data_direct!(x, y, w, cdims)) == x_size
    @test size(NNlib.∇conv_filter_direct!(w, x, y, cdims)) == w_size
end

# https://github.com/FluxML/NNlib.jl/issues/490
# https://github.com/FluxML/NNlib.jl/issues/405
@testset "conv_direct! - Unusual input types" begin
    # Create test type that can't be indexed when undefined.
    # This simulates the worst-case scenario for custom types.
    struct MyFloat <: Real
        set::Set{Float32}
    end

    # Test that direct indexing fails when undefined.
    v = Array{MyFloat}(undef, 3)
    @test_throws UndefRefError v[1]

    # Define minimal set of functions required for conv_direct!
    MyFloat(x::MyFloat) = x
    MyFloat(x::Real) = MyFloat(Set(Float32(x)))

    Base.:+(x::MyFloat, y::MyFloat) = MyFloat(only(x.set) + only(y.set))
    Base.:*(x::MyFloat, y::MyFloat) = MyFloat(only(x.set) * only(y.set))
    Base.promote_rule(::Type{MyFloat}, ::Type{Float32})   = MyFloat
    Base.rand(::AbstractRNG, ::SamplerType{MyFloat}) = MyFloat(rand(Float32))
    Base.zero(::MyFloat) = MyFloat(zero(Float32))
    Base.zero(::Type{MyFloat}) = MyFloat(zero(Float32))

    # Test conv_direct!
    x_size = (6, 7, 8, 5, 3)
    y_size = (5, 6, 7, 4, 3)
    w_size = (2, 2, 2, 5, 4)
    x = rand(MyFloat, x_size);
    w = randn(Float32, w_size);
    y = Array{MyFloat}(undef, y_size...);
    cdims = DenseConvDims(x_size, w_size)
    y_out = NNlib.conv_direct!(y, x, w, cdims)

    @test eltype(y_out) == MyFloat
    @test size(y_out) == y_size
end

@testset "AutoDiff: spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
  x = rand(rng, repeat([5], spatial_rank)..., 3, 2)
  w = rand(rng, repeat([3], spatial_rank)..., 3, 3)
  cdims = DenseConvDims(x, w)
  gradtest((x, w) -> conv(x, w, cdims), x, w)
  gradtest((x, w) -> sum(conv(x, w, cdims)), x, w)  # https://github.com/FluxML/Flux.jl/issues/1055

  y = conv(x, w, cdims)
  gradtest((y, w) -> ∇conv_data(y, w, cdims), y, w)
  gradtest((y, w) -> sum(∇conv_data(y, w, cdims)), y, w)
  gradtest((x, y) -> ∇conv_filter(x, y, cdims), x, y)
  gradtest((x, y) -> sum(∇conv_filter(x, y, cdims)), x, y)

  dcdims = DepthwiseConvDims(x, w)
  gradtest((x, w) -> depthwiseconv(x, w, dcdims), x, w)

  # FIXME fails
  y = depthwiseconv(x, w, dcdims)
  gradtest((y, w) -> ∇depthwiseconv_data(y, w, dcdims), y, w)
  gradtest((y, w) -> sum(∇depthwiseconv_data(y, w, dcdims)), y, w)
end

@static if Test_Enzyme

@testset "EnzymeRules: conv! spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
  x = rand(rng, repeat([5], spatial_rank)..., 3, 2)
  w = rand(rng, repeat([3], spatial_rank)..., 3, 3)

  cdims = DenseConvDims(x, w)

  curconv = conv
  curconv! = conv!
  dst = curconv(x, w, cdims)

  for Tret in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tdst in (EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tx in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tw in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated)

    Tret == EnzymeCore.Const && continue # ERROR
    EnzymeTestUtils.are_activities_compatible(Tret, Tdst, Tx, Tw) || continue

    EnzymeTestUtils.test_reverse(curconv!, Tret, (dst, Tdst), (x, Tx), (w, Tw), (cdims, EnzymeCore.Const), atol=1e-6, rtol=1e-6)
  end
end

@testset "EnzymeRules: ∇conv_data! spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
  x = rand(rng, repeat([5], spatial_rank)..., 3, 2)
  w = rand(rng, repeat([3], spatial_rank)..., 3, 3)
  cdims = DenseConvDims(x, w)
  y = conv(x, w, cdims)
  dy = randn(rng, size(y)...)

  dx = ∇conv_data(dy, w, cdims)

  for Tret in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tdst in (EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Ty in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tw in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated)

    Tret == EnzymeCore.Const && continue # ERROR
    EnzymeTestUtils.are_activities_compatible(Tret, Tdst, Ty, Tw) || continue

    EnzymeTestUtils.test_reverse(∇conv_data!, Tret, (dx, Tdst), (dy, Ty), (w, Tw), (cdims, EnzymeCore.Const), atol=1e-6, rtol=1e-6)
  end
end

@testset "EnzymeRules: ∇conv_filter! spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
  x = rand(rng, repeat([5], spatial_rank)..., 3, 2)
  w = rand(rng, repeat([3], spatial_rank)..., 3, 3)
  cdims = DenseConvDims(x, w)
  y = conv(x, w, cdims)
  dy = randn(rng, size(y)...)

  dw = ∇conv_filter(x, dy, cdims)

  for Tret in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tdst in (EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tx in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Ty in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated)

    Tret == EnzymeCore.Const && continue # ERROR
    EnzymeTestUtils.are_activities_compatible(Tret, Tdst, Tx, Ty) || continue

    EnzymeTestUtils.test_reverse(∇conv_filter!, Tret, (dw, Tdst), (x, Tx), (dy, Ty), (cdims, EnzymeCore.Const), atol=1e-6, rtol=1e-6)
  end
end

@testset "EnzymeRules: depthwiseconv! spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
  x = rand(rng, repeat([5], spatial_rank)..., 3, 2)
  w = rand(rng, repeat([3], spatial_rank)..., 3, 3)

  cdims = DepthwiseConvDims(x, w)

  curconv = depthwiseconv
  curconv! = depthwiseconv!
  dst = curconv(x, w, cdims)

  for Tret in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tdst in (EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tx in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tw in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated)

    Tret == EnzymeCore.Const && continue # ERROR
    EnzymeTestUtils.are_activities_compatible(Tret, Tdst, Tx, Tw) || continue

    EnzymeTestUtils.test_reverse(curconv!, Tret, (dst, Tdst), (x, Tx), (w, Tw), (cdims, EnzymeCore.Const), atol=1e-6, rtol=1e-6)
  end
end

end
