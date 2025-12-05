# using NNlib, Test

maxpool_answer_dict = Dict(
    1 => Dict(
        "y"          => [2, 4.],
        "y_nostride" => [2, 3, 4, 5.],
        "y_pad"      => [1, 3, 5.],

        "dx"          => [0, 2, 0, 4, 0.],
        "dx_nostride" => [0, 2, 3, 4, 5.],
        "dx_pad"      => [1, 0, 3, 0, 5.],
    ),
    2 => Dict(
        "y" => [
            7 17.;
            9 19.
        ],
        "y_nostride" => [
            7  12 17;
            8  13 18;
            9  14 19;
            10 15 20.
        ],
        "y_pad" => [
            1  11 16;
            3  13 18;
            5  15 20.
        ],

        "dx" => [
            0 0 0  0;
            0 7 0 17;
            0 0 0  0;
            0 9 0 19;
            0 0 0  0.
        ],
        "dx_nostride" => [
            0  0  0  0;
            0  7 12 17;
            0  8 13 18;
            0  9 14 19;
            0 10 15 20.
        ],
        "dx_pad"      => [
            1 0 11 16;
            0 0  0  0;
            3 0 13 18;
            0 0  0  0;
            5 0 15 20.
        ],
    ),
    3 => Dict(
        "y" => reshape([
            27, 29,
            37, 39.
        ], (2, 2, 1)),
        "y_nostride" => reshape([
            27, 28, 29, 30,
            32, 33, 34, 35,
            37, 38, 39, 40,

            47, 48, 49, 50,
            52, 53, 54, 55,
            57, 58, 59, 60.
        ], (4, 3, 2)),
        "y_pad" => reshape([
             1,  3, 5,
            11, 13, 15,
            16, 18, 20,

            41, 43, 45,
            51, 53, 55,
            56, 58, 60.
        ], (3, 3, 2)),

        "dx" => reshape([
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,

            0,  0, 0,  0, 0,
            0, 27, 0, 29, 0,
            0,  0, 0,  0, 0,
            0, 37, 0, 39, 0,

            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0.
        ], (5, 4, 3)),
        "dx_nostride" => reshape([
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,

            0,  0,  0,  0,  0,
            0, 27, 28, 29, 30,
            0, 32, 33, 34, 35,
            0, 37, 38, 39, 40,

            0,  0,  0,  0,  0,
            0, 47, 48, 49, 50,
            0, 52, 53, 54, 55,
            0, 57, 58, 59, 60.
        ], (5, 4, 3)),
        "dx_pad" => reshape([
             1, 0,  3, 0,  5,
             0, 0,  0, 0,  0,
            11, 0, 13, 0, 15,
            16, 0, 18, 0, 20,

            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,

            41, 0, 43, 0, 45,
             0, 0,  0, 0,  0,
            51, 0, 53, 0, 55,
            56, 0, 58, 0, 60.
        ], (5, 4, 3)),
    )
)

meanpool_answer_dict = Dict(
    1 => Dict(
        "y"          => [1.5, 3.5],
        "y_nostride" => [1.5, 2.5, 3.5, 4.5],
        "y_pad"      => [0.5, 2.5, 4.5],

        "dx"          => [0.75, 0.75, 1.75, 1.75,  0.0],
        "dx_nostride" => [0.75,  2.0,  3.0,  4.0, 2.25],
        "dx_pad"      => [0.25, 1.25, 1.25, 2.25, 2.25],
    ),
    2 => Dict(
        "y" => [
            4.0 14.0;
            6.0 16.0
        ],
        "y_nostride" => [
            4.0  9.0 14.0
            5.0 10.0 15.0
            6.0 11.0 16.0
            7.0 12.0 17.0
        ],
        "y_pad" => [
            0.25  4.25 4.0
            1.25 10.0  8.75
            2.25 12.0  9.75
        ],

        "dx" => [
            1.0 1.0 3.5 3.5;
            1.0 1.0 3.5 3.5;
            1.5 1.5 4.0 4.0;
            1.5 1.5 4.0 4.0;
            0.0 0.0 0.0 0.0
        ],
        "dx_nostride" => [
            1.0  3.25  5.75 3.5;
            2.25 7.0  12.0  7.25;
            2.75 8.0  13.0  7.75;
            3.25 9.0  14.0  8.25;
            1.75 4.75  7.25 4.25
        ],
        "dx_pad"      => [
            0.0625 1.0625 1.0625 1.0;
            0.3125 2.5    2.5    2.1875;
            0.3125 2.5    2.5    2.1875;
            0.5625 3.0    3.0    2.4375;
            0.5625 3.0    3.0    2.4375
        ],
    ),
    3 => Dict(
        "y" => reshape([
            14.0, 16.0,
            24.0, 26.0
        ], (2, 2, 1)),
        "y_nostride" => reshape([
            14.0, 15.0, 16.0, 17.0,
            19.0, 20.0, 21.0, 22.0,
            24.0, 25.0, 26.0, 27.0,

            34.0, 35.0, 36.0, 37.0,
            39.0, 40.0, 41.0, 42.0,
            44.0, 45.0, 46.0, 47.0
        ], (4, 3, 2)),
        "y_pad" => reshape([
            0.125, 0.625, 1.125,
            2.125, 5.0,   6.0,
            2.0,   4.375, 4.875,

             7.75, 16.25, 17.25,
            19.25, 40.0,  42.0,
            11.5,  23.75, 24.75,
        ], (3, 3, 2)),

        "dx" => reshape([
            1.75, 1.75, 2.0, 2.0, 0.0,
            1.75, 1.75, 2.0, 2.0, 0.0,
            3.0, 3.0, 3.25, 3.25, 0.0,
            3.0, 3.0, 3.25, 3.25, 0.0,

            1.75, 1.75, 2.0, 2.0, 0.0,
            1.75, 1.75, 2.0, 2.0, 0.0,
            3.0, 3.0, 3.25, 3.25, 0.0,
            3.0, 3.0, 3.25, 3.25, 0.0,

            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ], (5, 4, 3)),
        "dx_nostride" => reshape([
            1.75,   3.625,  3.875,  4.125, 2.125,
            4.125,  8.5,    9.0,    9.5,   4.875,
            5.375, 11.0,   11.5,   12.0,   6.125,
            3.0,    6.125,  6.375,  6.625, 3.375,

             6.0,  12.25, 12.75, 13.25,  6.75,
            13.25, 27.0,  28.0,  29.0,  14.75,
            15.75, 32.0,  33.0,  34.0,  17.25,
             8.5,  17.25, 17.75, 18.25,  9.25,

             4.25,   8.625,  8.875,  9.125,  4.625,
             9.125, 18.5,   19.0,   19.5,    9.875,
            10.375, 21.0,   21.5,   22.0,   11.125,
             5.5,   11.125, 11.375, 11.625,  5.875
        ], (5, 4, 3)),
        "dx_pad" => reshape([
            0.015625, 0.078125, 0.078125, 0.140625, 0.140625,
            0.265625, 0.625,    0.625,    0.75,     0.75,
            0.265625, 0.625,    0.625,    0.75,     0.75,
            0.25,     0.546875, 0.546875, 0.609375, 0.609375,

            0.96875, 2.03125, 2.03125, 2.15625, 2.15625,
            2.40625, 5.0,     5.0,     5.25,    5.25,
            2.40625, 5.0,     5.0,     5.25,    5.25,
            1.4375,  2.96875, 2.96875, 3.09375, 3.09375,

            0.96875, 2.03125, 2.03125, 2.15625, 2.15625,
            2.40625, 5.0,     5.0,     5.25,    5.25,
            2.40625, 5.0,     5.0,     5.25,    5.25,
            1.4375,  2.96875, 2.96875, 3.09375, 3.09375
        ], (5, 4, 3)),
    )
)

lpnormpool_answer_dict = Dict(
    1 => Dict(
        "y"           => [2.019312856150994, 4.221163518110637],
        "y_nostride"  => [
            2.080083823051904, 3.2710663101885897,
            4.497941445275415, 5.738793548317167
        ],
        "y_pad"       => [1.0, 3.605551275463989, 6.4031242374328485],
        "dx"          => [
            0.17258020254042603, 1.9525221042381296,
            1.2774501198988355, 3.496467771732918, 0.0
        ],
        "dx_nostride" => [
            0.48074985676913606, 3.1458422620080637,
            4.752311710531486, 6.345225258061685, 4.356316321455918
        ],
        "dx_pad"       => [1.0, 2.0, 3.0, 4.0, 5.0],
        "p"           => 4.5,
        "p_nostride"  => 3.0,
        "p_pad"       => 2.0
    ),
    2 => Dict(
        "y"           => [
            8.71909  24.9703;
            11.7336  28.3804
        ],
        "y_nostride"  => [
            11.1128  23.134   35.5704;
            13.4219  25.6082  38.0707;
            15.8033  28.0907  40.5735;
            18.2249  30.5795  43.0782
        ],
        "y_pad"       => [
            1.0      11.3616  16.0;
            3.19158  15.9662  21.3545;
            5.56869  18.7771  23.7903
        ],
        "dx"          => [
            0.33866   4.97727  7.30092  12.8076;
            0.957876  6.27208  8.31879  14.0269;
            1.51693   6.6057   8.79844  14.3351;
            2.33547   7.8822   9.83293  15.5461;
            0.0       0.0      0.0      0.0 
        ],
        "dx_nostride" => [
            3.33359  19.9471  35.7329  23.8564;
            9.89551  44.627   76.2257  50.0307;
           13.231    50.9101  82.5686  53.2022;
           16.4888   57.223   88.9133  56.3742;
            9.54591  30.9869  46.8371  29.3524
        ],
        "dx_pad"      => [
            1.0       2.30261  10.4791   16.0;
            0.992125  2.0321    7.81903  12.075;
            2.73398   2.83743   9.5512   13.9299;
            2.43512   2.98652   9.0132   13.5608;
            4.25398   3.8865   10.7099   15.4161
        ],
        "p"           => 2.5,
        "p_nostride"  => 1.5,
        "p_pad"       => 3.5
    )
)

for rank in (1, 2, 3)
    @testset "pool$(rank)d" begin
        for (pool, ∇pool, answer_dict) in (
                # Main API name
                (maxpool, ∇maxpool, maxpool_answer_dict),
                (meanpool, ∇meanpool, meanpool_answer_dict),

                # _direct name
                (NNlib.maxpool_direct, NNlib.∇maxpool_direct, maxpool_answer_dict),
                (NNlib.meanpool_direct, NNlib.∇meanpool_direct, meanpool_answer_dict),)

            @testset "$(pool)$(rank)d" begin
                y = answer_dict[rank]["y"]
                y_nostride = answer_dict[rank]["y_nostride"]
                y_pad = answer_dict[rank]["y_pad"]
                dx = answer_dict[rank]["dx"]
                dx_nostride = answer_dict[rank]["dx_nostride"]
                dx_pad = answer_dict[rank]["dx_pad"]

                x = reshape(Float64[1:prod(size(dx));], size(dx)..., 1, 1)

                # A "drop channels and batch dimension" helper
                ddims(x) = dropdims(x, dims=(rank + 1, rank + 2))

                # Let's ensure that a 1x1x1 pooling kernel always just returns `x`
                @test pool(x, PoolDims(x, 1)) == x

                # Test vanilla pooling
                pdims = PoolDims(x, 2)
                y_hat = pool(x, pdims)
                @test ddims(y_hat) == y
                @test ddims(∇pool(y_hat, y_hat, x, pdims)) == dx

                # Strided pooling
                pdims = PoolDims(x, 2; stride=1)
                y_hat = pool(x, pdims)
                @test ddims(y_hat) == y_nostride
                @test ddims(∇pool(y_hat, y_hat, x, pdims)) == dx_nostride

                # Padded pooling
                pdims = PoolDims(x, 2; padding=1)
                y_hat = pool(x, pdims)
                @test ddims(y_hat) == y_pad
                @test ddims(∇pool(y_hat, y_hat, x, pdims)) == dx_pad
            end
        end
    end
end

for rank in (1, 2)
    for (pool, ∇pool, answer_dict) in (
            (lpnormpool, ∇lpnormpool, lpnormpool_answer_dict),
            (NNlib.lpnormpool_direct, NNlib.∇lpnormpool_direct, lpnormpool_answer_dict),)
        @testset "$(pool)$(rank)d" begin
            y = answer_dict[rank]["y"]
            y_nostride = answer_dict[rank]["y_nostride"]
            y_pad = answer_dict[rank]["y_pad"]
            dx = answer_dict[rank]["dx"]
            dx_nostride = answer_dict[rank]["dx_nostride"]
            dx_pad = answer_dict[rank]["dx_pad"]
            p = answer_dict[rank]["p"]
            p_nostride = answer_dict[rank]["p_nostride"]
            p_pad = answer_dict[rank]["p_pad"]

            x = reshape(Float64[1:prod(size(dx));], size(dx)..., 1, 1)

            ddims(x) = dropdims(x, dims=(rank + 1, rank + 2))

            @test pool(x, PoolDims(x, 1); p=p) ≈ x atol = 1e-3

            # Test vanilla pooling
            pdims = PoolDims(x, 2)
            y_hat = pool(x, pdims; p=p)
            @test ddims(y_hat) ≈ y atol = 1e-3
            @test ddims(∇pool(y_hat, y_hat, x, pdims; p=p)) ≈ dx atol = 1e-3

            # Strided pooling
            pdims = PoolDims(x, 2; stride=1)
            y_hat = pool(x, pdims; p=p_nostride)
            @test ddims(y_hat) ≈ y_nostride atol = 1e-3
            @test ddims(∇pool(y_hat, y_hat, x, pdims; p=p_nostride)) ≈ dx_nostride atol = 1e-3

            # Padded pooling
            pdims = PoolDims(x, 2; padding=1)
            y_hat = pool(x, pdims; p=p_pad)
            @test ddims(y_hat) ≈ y_pad atol = 1e-3
            @test ddims(∇pool(y_hat, y_hat, x, pdims; p=p_pad)) ≈ dx_pad atol = 1e-3
        end
    end
end

@testset "Pooling - Check Sizes" begin
    x = rand(10, 10, 3, 10)
    @test size(maxpool(x, (2, 2))) == (5, 5, 3, 10)
    @test size(maxpool(x, (2, 2); pad=(1, 1), stride=(2, 2))) == (6, 6, 3, 10)
    @test size(meanpool(x, (2, 2))) == (5, 5, 3, 10)
    @test size(meanpool(x, (2, 2); pad=(1, 1), stride=(2, 2))) == (6, 6, 3, 10)
end

# Add another test for 2d maxpool that uses an odd-length size:
@testset "Issue #133" begin
    x = reshape([(1.:9.)...], 3, 3, 1, 1)
    pdims = PoolDims(size(x), (2, 2), padding=(1, 1), stride=(2, 2))
    y = maxpool(x, pdims)

    dy = y .* 0 .+ 1
    dx = ∇maxpool(dy, y, x, pdims)
    @test dx[:,:,1,1] == [1.0 0.0 1.0; 0.0 0.0 0.0; 1.0 0.0 1.0]
end

# test "true" strided case, see https://github.com/FluxML/NNlib.jl/issues/205


# obtained with
# using FiniteDifferences
maxpool_answer_nature = Dict(
    "rank1" => Dict(
        # kernel size 2, stride 1, pad 0
        "k2s1p0" => (size = (2,),
            stride = 1,
            pad = 0,

            x = reshape([
                0.0299635,  0.233456,  0.596161,   0.161514,  0.0094027
            ], 5, 1, 1), # width, channel, batch_size

            dx_maxpool = reshape([
                 0.0, 1.0, 2.0, 1.0, 0.0
            ], 5, 1, 1),

            dx_meanpool = reshape([
                 0.5, 1.0, 1.0, 1.0, 0.5
            ], 5, 1, 1),),
        "k2s1p1" => (size = (2,),
            stride = 1,
            pad = 1,

            x = reshape([
                0.0299635,  0.233456,  0.596161,   0.161514,  0.0094027
            ], 5, 1, 1),

            dx_maxpool = reshape([
                 1.0, 1.0, 2.0, 1.0, 1.0
            ], 5, 1, 1),

            dx_meanpool = reshape([
                 1.0, 1.0, 1.0, 1.0, 1.0
            ], 5, 1, 1),),
        "k3s1p1" => (size = (3,),
            stride = 1,
            pad = 1,

            x = reshape([
                0.0299635,  0.233456,  0.596161,   0.161514,  0.0094027
            ], 5, 1, 1),

            dx_maxpool = reshape([
                 0.0, 1.0, 3.0, 1.0, 0.0
            ], 5, 1, 1),

            dx_meanpool = reshape([
                 0.6666666666, 1.0, 1.0, 1.0, 0.6666666666
            ], 5, 1, 1),),
        "k3s2p1" => (size = (3,),
            stride = 2,
            pad = 1,

            x = reshape([
                0.0299635,  0.233456,  0.596161,   0.161514,  0.0094027
            ], 5, 1, 1),

            dx_maxpool = reshape([
                 0.0, 1.0, 1.0, 1.0, 0.0
            ], 5, 1, 1),

            dx_meanpool = reshape([
                 0.333333333,
                 0.666666666,
                 0.333333333,
                 0.666666666,
                 0.333333333,
            ], 5, 1, 1),)
    ),
    "rank2" => Dict(
        # kernel size 2, stride 1, pad 0
        "k2s1p0" => (size = (2, 2),
            stride = 1,
            pad = 0,

            x = reshape([
                0.0299635  0.233456  0.596161   0.161514  0.0094027
                0.389984   0.235158  0.579525   0.301893  0.561358
                0.0830242  0.483759  0.914904   0.253871  0.820061
                0.425287   0.53451   0.0405225  0.729861  0.403925
                0.473724   0.571418  0.558427   0.552183  0.561624
            ], 5, 5, 1, 1),

            dx_maxpool = reshape([
                0.0  0.0  2.0  0.0  0.0
                1.0  0.0  0.0  0.0  1.0
                0.0  1.0  4.0  0.0  2.0
                0.0  1.0  0.0  2.0  0.0
                0.0  2.0  0.0  0.0  0.0
            ], 5, 5, 1, 1),

            dx_meanpool = reshape([
                0.25  0.5  0.5  0.5  0.25
                0.5   1.0  1.0  1.0  0.5
                0.5   1.0  1.0  1.0  0.5
                0.5   1.0  1.0  1.0  0.5
                0.25  0.5  0.5  0.5  0.25
            ], 5, 5, 1, 1)),
        "k2s1p1" => (size = (2, 2),
            stride = 1,
            pad = 1,

            x = reshape([
                0.0299635  0.233456  0.596161   0.161514  0.0094027
                0.389984   0.235158  0.579525   0.301893  0.561358
                0.0830242  0.483759  0.914904   0.253871  0.820061
                0.425287   0.53451   0.0405225  0.729861  0.403925
                0.473724   0.571418  0.558427   0.552183  0.561624
            ], 5, 5, 1, 1),

            dx_maxpool = reshape([
                1.0  1.0  4.0  1.0  1.0
                3.0  0.0  0.0  0.0  2.0
                0.0  1.0  4.0  0.0  4.0
                1.0  1.0  0.0  2.0  0.0
                2.0  4.0  1.0  0.0  3.0
            ], 5, 5, 1, 1),

            dx_meanpool = reshape([
                1.0  1.0  1.0  1.0  1.0
                1.0  1.0  1.0  1.0  1.0
                1.0  1.0  1.0  1.0  1.0
                1.0  1.0  1.0  1.0  1.0
                1.0  1.0  1.0  1.0  1.0
            ], 5, 5, 1, 1)),
        "k3s1p1" => (size = (3, 3),
            stride = 1,
            pad = 1,

            x = reshape([
                0.0299635  0.233456  0.596161   0.161514  0.0094027
                0.389984   0.235158  0.579525   0.301893  0.561358
                0.0830242  0.483759  0.914904   0.253871  0.820061
                0.425287   0.53451   0.0405225  0.729861  0.403925
                0.473724   0.571418  0.558427   0.552183  0.561624
            ], 5, 5, 1, 1),

            dx_maxpool = reshape([
                0.0  0.0  3.0  0.0  0.0
                1.0  0.0  0.0  0.0  1.0
                0.0  1.0  9.0  0.0  3.0
                0.0  1.0  0.0  3.0  0.0
                0.0  3.0  0.0  0.0  0.0
            ], 5, 5, 1, 1),

            dx_meanpool = reshape([
                0.444444  0.666667  0.666667  0.666667  0.444444
                0.666667  1.0       1.0       1.0       0.666667
                0.666667  1.0       1.0       1.0       0.666667
                0.666667  1.0       1.0       1.0       0.666667
                0.444444  0.666667  0.666667  0.666667  0.444444
            ], 5, 5, 1, 1)),
        "k3s2p1" => (size = (3, 3),
            stride = 2,
            pad = 1,

            x = reshape([
                0.0299635  0.233456  0.596161   0.161514  0.0094027
                0.389984   0.235158  0.579525   0.301893  0.561358
                0.0830242  0.483759  0.914904   0.253871  0.820061
                0.425287   0.53451   0.0405225  0.729861  0.403925
                0.473724   0.571418  0.558427   0.552183  0.561624
            ], 5, 5, 1, 1),

            dx_maxpool = reshape([
                0.0  0.0  1.0  0.0  0.0
                1.0  0.0  0.0  0.0  1.0
                0.0  0.0  1.0  0.0  1.0
                0.0  1.0  0.0  2.0  0.0
                0.0  1.0  0.0  0.0  0.0
            ], 5, 5, 1, 1),

            dx_meanpool = reshape([
                0.111111  0.222222  0.111111  0.222222  0.111111
                0.222222  0.444444  0.222222  0.444444  0.222222
                0.111111  0.222222  0.111111  0.222222  0.111111
                0.222222  0.444444  0.222222  0.444444  0.222222
                0.111111  0.222222  0.111111  0.222222  0.111111
            ], 5, 5, 1, 1))
    ),
    "rank3" => Dict(
        # kernel size 2, stride 1, pad 0
        "k2s1p0" => (size = (2, 2, 2),
            stride = 1,
            pad = 0,

            x = reshape(cat([
                    0.82584   0.416818  0.92668   0.471931
                    0.798798  0.131608  0.344556  0.79681
                    0.716898  0.320672  0.24453   0.288568
                    0.261484  0.258469  0.121916  0.0685961
                ],
                [
                    0.73934   0.16631    0.525109   0.0223458
                    0.164918  0.790875   0.444085   0.469671
                    0.116848  0.359845   0.0653075  0.804886
                    0.525431  0.0402844  0.846814   0.84876
                ],
                [
                    0.709245  0.325828  0.715952  0.719116
                    0.576722  0.405659  0.770104  0.259131
                    0.640221  0.28811   0.129229  0.97571
                    0.953795  0.1316    0.94538   0.705337
                ],dims=3), 4,4,3,1,1),

            dx_maxpool = reshape(cat([
                     1.0  0.0  2.0  0.0
                     1.0  0.0  0.0  0.0
                     1.0  0.0  0.0  0.0
                     0.0  0.0  0.0  0.0
                ],
                [
                     0.0  0.0  0.0  0.0
                     0.0  5.0  0.0  0.0
                     0.0  0.0  0.0  1.0
                     0.0  0.0  1.0  1.0
                ],
                [
                     0.0  0.0  0.0  0.0
                     0.0  0.0  1.0  0.0
                     0.0  0.0  0.0  2.0
                     1.0  0.0  1.0  0.0
                ],dims=3), 4,4,3,1,1),

            dx_meanpool = reshape(cat([
                     0.125  0.25  0.25  0.125
                     0.25   0.5   0.5   0.25
                     0.25   0.5   0.5   0.25
                     0.125  0.25  0.25  0.125
                ],
                [
                     0.25  0.5  0.5  0.25
                     0.5   1.0  1.0  0.5
                     0.5   1.0  1.0  0.5
                     0.25  0.5  0.5  0.25
                ],
                [
                     0.125  0.25  0.25  0.125
                     0.25   0.5   0.5   0.25
                     0.25   0.5   0.5   0.25
                     0.125  0.25  0.25  0.125
                ],dims=3), 4,4,3,1,1)),
        "k2s1p1" => (size = (2, 2, 2),
            stride = 1,
            pad = 1,

            x = reshape(cat([
                    0.82584   0.416818  0.92668   0.471931
                    0.798798  0.131608  0.344556  0.79681
                    0.716898  0.320672  0.24453   0.288568
                    0.261484  0.258469  0.121916  0.0685961
                ],
                [
                    0.73934   0.16631    0.525109   0.0223458
                    0.164918  0.790875   0.444085   0.469671
                    0.116848  0.359845   0.0653075  0.804886
                    0.525431  0.0402844  0.846814   0.84876
                ],
                [
                    0.709245  0.325828  0.715952  0.719116
                    0.576722  0.405659  0.770104  0.259131
                    0.640221  0.28811   0.129229  0.97571
                    0.953795  0.1316    0.94538   0.705337
                ],dims=3), 4,4,3,1,1),

            dx_maxpool = reshape(cat([
                     8.0  0.0  8.0  2.0
                     4.0  0.0  1.0  4.0
                     4.0  1.0  0.0  2.0
                     2.0  1.0  1.0  1.0
                ],
                [
                     3.0  0.0  0.0  0.0
                     0.0  5.0  0.0  0.0
                     0.0  0.0  0.0  2.0
                     2.0  0.0  2.0  5.0
                ],
                [
                     4.0  0.0  2.0  6.0
                     0.0  0.0  4.0  0.0
                     3.0  0.0  0.0  8.0
                     8.0  0.0  6.0  1.0
                ],dims=3), 4,4,3,1,1),

            dx_meanpool = reshape(cat([
                     1.0  1.0  1.0  1.0
                     1.0  1.0  1.0  1.0
                     1.0  1.0  1.0  1.0
                     1.0  1.0  1.0  1.0
                ],
                [
                     1.0  1.0  1.0  1.0
                     1.0  1.0  1.0  1.0
                     1.0  1.0  1.0  1.0
                     1.0  1.0  1.0  1.0
                ],
                [
                     1.0  1.0  1.0  1.0
                     1.0  1.0  1.0  1.0
                     1.0  1.0  1.0  1.0
                     1.0  1.0  1.0  1.0
                ],dims=3), 4,4,3,1,1)),
        "k3s1p1" => (size = (3, 3, 2),
            stride = 1,
            pad = 1,

            x = reshape(cat([
                    0.82584   0.416818  0.92668   0.471931
                    0.798798  0.131608  0.344556  0.79681
                    0.716898  0.320672  0.24453   0.288568
                    0.261484  0.258469  0.121916  0.0685961
                ],
                [
                    0.73934   0.16631    0.525109   0.0223458
                    0.164918  0.790875   0.444085   0.469671
                    0.116848  0.359845   0.0653075  0.804886
                    0.525431  0.0402844  0.846814   0.84876
                ],
                [
                    0.709245  0.325828  0.715952  0.719116
                    0.576722  0.405659  0.770104  0.259131
                    0.640221  0.28811   0.129229  0.97571
                    0.953795  0.1316    0.94538   0.705337
                ],dims=3), 4,4,3,1,1),

            dx_maxpool = reshape(cat([
                     4.0  0.0  12.0  0.0
                     3.0  0.0   0.0  2.0
                     3.0  1.0   0.0  1.0
                     0.0  0.0   0.0  0.0
                ],
                [
                     0.0  0.0  0.0  0.0
                     0.0  5.0  0.0  0.0
                     0.0  0.0  0.0  0.0
                     0.0  0.0  2.0  4.0
                ],
                [
                     2.0  0.0  0.0   0.0
                     0.0  0.0  5.0   0.0
                     0.0  0.0  0.0  12.0
                     8.0  0.0  0.0   0.0
                ],dims=3), 4,4,3,1,1),

            dx_meanpool = reshape(cat([
                     0.444444  0.666667  0.666667  0.444444
                     0.666667  1.0       1.0       0.666667
                     0.666667  1.0       1.0       0.666667
                     0.444444  0.666667  0.666667  0.444444
                ],
                [
                     0.444444  0.666667  0.666667  0.444444
                     0.666667  1.0       1.0       0.666667
                     0.666667  1.0       1.0       0.666667
                     0.444444  0.666667  0.666667  0.444444
                ],
                [
                     0.444444  0.666667  0.666667  0.444444
                     0.666667  1.0       1.0       0.666667
                     0.666667  1.0       1.0       0.666667
                     0.444444  0.666667  0.666667  0.444444
                ],dims=3), 4,4,3,1,1)),
        "k3s2p1" => (size = (3, 3, 2),
            stride = 2,
            pad = 1,

            x = reshape(cat([
                    0.82584   0.416818  0.92668   0.471931
                    0.798798  0.131608  0.344556  0.79681
                    0.716898  0.320672  0.24453   0.288568
                    0.261484  0.258469  0.121916  0.0685961
                ],
                [
                    0.73934   0.16631    0.525109   0.0223458
                    0.164918  0.790875   0.444085   0.469671
                    0.116848  0.359845   0.0653075  0.804886
                    0.525431  0.0402844  0.846814   0.84876
                ],
                [
                    0.709245  0.325828  0.715952  0.719116
                    0.576722  0.405659  0.770104  0.259131
                    0.640221  0.28811   0.129229  0.97571
                    0.953795  0.1316    0.94538   0.705337
                ],dims=3), 4,4,3,1,1),

            dx_maxpool = reshape(cat([
                     1.0  0.0  1.0  0.0
                     1.0  0.0  0.0  1.0
                     0.0  0.0  0.0  0.0
                     0.0  0.0  0.0  0.0
                ],
                [
                     0.0  0.0  0.0  0.0
                     0.0  2.0  0.0  0.0
                     0.0  0.0  0.0  0.0
                     0.0  0.0  0.0  0.0
                ],
                [
                     0.0  0.0  0.0  0.0
                     0.0  0.0  0.0  0.0
                     0.0  0.0  0.0  1.0
                     1.0  0.0  0.0  0.0
                ],dims=3), 4,4,3,1,1),

            dx_meanpool = reshape(cat([
                     0.0555556  0.111111  0.0555556  0.0555556
                     0.111111   0.222222  0.111111   0.111111
                     0.0555556  0.111111  0.0555556  0.0555556
                     0.0555556  0.111111  0.0555556  0.0555556
                ],
                [
                     0.0555556  0.111111  0.0555556  0.0555556
                     0.111111   0.222222  0.111111   0.111111
                     0.0555556  0.111111  0.0555556  0.0555556
                     0.0555556  0.111111  0.0555556  0.0555556
                ],
                [
                     0.0555556  0.111111  0.0555556  0.0555556
                     0.111111   0.222222  0.111111   0.111111
                     0.0555556  0.111111  0.0555556  0.0555556
                     0.0555556  0.111111  0.0555556  0.0555556
                ],dims=3), 4,4,3,1,1))
    )
)


@testset "more maxpool and meanpool tests" begin
    # issue #205
    function check(config, T)
        # CHECK DEFAULT
        pdims = PoolDims(config.x, config.size; stride=config.stride, padding=config.pad)
        x = T.(config.x)
        y_maxpool = NNlib.maxpool(x, pdims)
        y_meanpool = NNlib.meanpool(x, pdims)
        dy = ones(T, size(y_maxpool)...) # size(y_maxpool) == size(y_meanpool)
        @test isapprox(config.dx_maxpool, NNlib.∇maxpool(dy, y_maxpool, x, pdims), rtol=1e-5)
        @test isapprox(config.dx_meanpool, NNlib.∇meanpool(dy, y_meanpool, x, pdims), rtol=1e-5)
        # CHECK DIRECT
        y_maxpool_dir = NNlib.maxpool_direct(x, pdims)
        y_meanpool_dir = NNlib.meanpool_direct(x, pdims)
        @test y_maxpool_dir ≈ y_maxpool  atol = 1e-6
        @test isapprox(config.dx_maxpool, NNlib.∇maxpool_direct(dy, y_maxpool_dir, x, pdims), rtol=1e-5)
        @test isapprox(config.dx_meanpool, NNlib.∇meanpool_direct(dy, y_meanpool_dir, x, pdims), rtol=1e-5)
    end

    for (rank_name, config_dict) in maxpool_answer_nature
        for (setting_name, config) in config_dict
            for T in (Float32, Float64)
                check(config, T)
            end
        end
    end

    # issue 210
    x, k = rand(Float32, 5, 2, 1, 3), (2, 1)
    pdims1 = NNlib.PoolDims(x, k, padding=1, stride=1)
    pdims2 = NNlib.PoolDims(x, k, padding=(1, 0, 0, 0), stride=1)
    @test maxpool(x, pdims1) isa Array{Float32,4}
    @test maxpool(x, pdims2) isa Array{Float32,4}

    # issue #229
    x = ones(Float32, 4, 4, 1, 1) .* -1
    pool = meanpool(x, PoolDims(x, 2, padding=1))
    valid = reshape([
    -0.25,  -0.5,  -0.25,
    -0.5,   -1.0,  -0.5,
    -0.25,  -0.5,  -0.25], (3, 3, 1, 1))
    @test all(pool .== valid)

    # issue #484
    # Description: some in-place pooling functions only accepted arrays with the same eltype.
    # The strict method signatures were based on assumption on the return type of `similar`.
    # For ReverseDiff, this caused problems, e.g. with taking derivatives of pooling 
    # operations.
    # Now, if explicitly calling an in-place pooling functions, a different `yT` is allowed.
    for xT in (Int32, Int64, Float16, Float32, Float64, BigFloat)
        for (xsz, psz) in (     # test a few different data and kernel sizes
            ((1,1), (1,1)),
            ((1,2), (1,1)), ((1,2), (1,2)),
            ((2,1), (1,1)), ((2,1), (2,1)),
            ((2,2), (1,1)), ((2,2), (1,2)), ((2,2), (2,1)),
        )
            x = ones(xT, xsz..., 1, 1)
            pdims = PoolDims(x, psz)
            for yT in (Float16, Float32, Float64, BigFloat) 
                # `yT` is the target eltype and we do not test integer types here
                # because those cannot always store the pooling results.
                y = similar(x, yT, NNlib.output_size(pdims)..., NNlib.channels_out(pdims), size(x, 4))
                @test maxpool!(y, x, pdims) isa Array{yT}
                @test meanpool!(y, x, pdims) isa Array{yT}
                @test lpnormpool!(y, x, pdims; p=2) isa Array{yT}
                @test lpnormpool!(y, x, pdims; p=1.0) isa Array{yT}
            end
        end
    end
    
    # This is how to test #484 with ReverseDiff:
    x = reshape(Float32[ 1 2; 3 4 ], (2,2,1,1))
    @test only(maxpool(x, (2,2))) == 4
    # define typemin, because of https://github.com/JuliaDiff/ReverseDiff.jl/issues/225
    Base.typemin(tr::Type{<:T}) where{V, T<:RD.TrackedReal{V, <:Any, <:Any}} = T(typemin(V))
    @test RD.gradient(_x -> only(maxpool(_x,(2,2))), x)[:,:,1,1] == [0 0; 0 1]
    @test only(meanpool(x, (2,2))) == 2.5
    @test all(==(0.25), RD.gradient(_x -> only(meanpool(_x,(2,2))), x))
end

@testset "AutoDiff: spatial_rank=$spatial_rank" for spatial_rank in (1, 2)
  x = rand(rng, repeat([10], spatial_rank)..., 3, 2)
  pdims = PoolDims(x, 2)
  gradtest(x -> maxpool(x, pdims), x; skip = spatial_rank==2)
  gradtest(x -> meanpool(x, pdims), x)
  gradtest(x -> sum(maxpool(x, pdims)), x, skip = spatial_rank==2)
  gradtest(x -> sum(meanpool(x, pdims)), x)

  #https://github.com/FluxML/NNlib.jl/issues/188
  k = ntuple(_ -> 2, spatial_rank)  # Kernel size of pool in ntuple format
  gradtest(x -> maxpool(x, k), x; skip = spatial_rank==2)
  gradtest(x -> meanpool(x, k), x)
  gradtest(x -> sum(maxpool(x, k)), x, skip = spatial_rank==2)
  gradtest(x -> sum(meanpool(x, k)), x)
end

@static if Test_Enzyme

@testset "EnzymeRules: pooling! $pool spatial_rank=$spatial_rank " for spatial_rank in (1, 2),
                                                                                (pool, pool!) in ((maxpool, maxpool!), (meanpool, meanpool!))

  x = rand(rng, repeat([10], spatial_rank)..., 3, 2)
  pdims = PoolDims(x, 2)
  y = pool(x, pdims)

  for Tret in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tdst in (EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
    Tsrc in (EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated)

    Tret == EnzymeCore.Const && continue # ERROR
    EnzymeTestUtils.are_activities_compatible(Tret, Tdst, Tsrc) || continue

    EnzymeTestUtils.test_reverse(pool!, Tret, (y, Tdst), (x, Tsrc), (pdims, EnzymeCore.Const))
  end

end

end