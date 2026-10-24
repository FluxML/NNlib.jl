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

# An unbound type parameter makes GPUCompiler, and so Reactant, reject the method (issue #849)
@testset "PoolDims has no unbound type parameters" begin
    @test !any(m -> occursin("PoolDims", String(m.name)), Test.detect_unbound_args(NNlib))
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

@testset "∇maxpool picks the exact argmax (issue #771)" begin
    # Brute-force reference: all of a window's gradient goes to its first largest
    # element, scanning the first spatial dimension fastest.
    function ref_∇maxpool(dy, x, pdims)
        N = ndims(x) - 2
        K, S, D = NNlib.kernel_size(pdims), NNlib.stride(pdims), NNlib.dilation(pdims)
        P = NNlib.padding(pdims)        # (w_lo, w_hi, h_lo, h_hi, d_lo, d_hi)
        dx = zero(x)
        for oidx in CartesianIndices(dy)
            c, batch = oidx[N+1], oidx[N+2]
            m, imax = nothing, nothing
            for kidx in CartesianIndices(K)
                iin = ntuple(d -> (oidx[d] - 1) * S[d] - P[2d-1] + 1 + (kidx[d] - 1) * D[d], N)
                all(d -> 1 <= iin[d] <= size(x, d), 1:N) || continue
                if m === nothing || x[iin..., c, batch] > m
                    m, imax = x[iin..., c, batch], iin
                end
            end
            dx[imax..., c, batch] += dy[oidx]
        end
        return dx
    end

    # The first element is within `≈` of the last one, which is the maximum
    x = reshape(Float32[0.95492, 0.15382, 0.03295, 0.95500], 2, 2, 1, 1)
    pdims = PoolDims(x, 2)
    y = maxpool(x, pdims)
    @test ∇maxpool(one.(y), y, x, pdims)[:,:,1,1] == [0 0; 0 1]
    # Same with padding, where each of the four windows holds all of `x`
    pdims = PoolDims(x, 3; padding=1, stride=1)
    y = maxpool(x, pdims)
    @test ∇maxpool(one.(y), y, x, pdims)[:,:,1,1] == [0 0; 0 4]

    # Among equal elements, the first one takes the gradient
    x = ones(Float32, 2, 2, 1, 1)
    pdims = PoolDims(x, 2)
    y = maxpool(x, pdims)
    @test ∇maxpool(one.(y), y, x, pdims)[:,:,1,1] == [1 0; 0 0]

    # Match the brute-force reference across spatial ranks and configs, on inputs whose
    # elements are all within `≈` of each other (and a few of them equal).
    for T in (Float32, Float64), nsd in (1, 2, 3)
        x = 1 .+ rand(rng, T, fill(9, nsd)..., 2, 2) .* sqrt(eps(T)) ./ 4
        for (k, pad, stride, dilation) in ((2, 0, 2, 1), (2, 1, 1, 1), (3, 1, 2, 1), (3, 2, 1, 2))
            pdims = PoolDims(x, k; padding=pad, stride=stride, dilation=dilation)
            y = maxpool(x, pdims)
            # Integer-valued, so that the sums in `dx` are exact whatever their order
            dy = T.(rand(rng, 1:8, size(y)))
            @test ∇maxpool(dy, y, x, pdims) == ref_∇maxpool(dy, x, pdims)
        end
    end

    # `y` may have a different eltype than `x` (issue #484), and so not hold the maxima
    # of `x` exactly
    x = randn(rng, Float64, 8, 8, 2, 2)
    pdims = PoolDims(x, 2)
    y = maxpool!(zeros(Float32, 4, 4, 2, 2), x, pdims)
    dy = ones(4, 4, 2, 2)
    @test ∇maxpool!(zero(x), dy, y, x, pdims) == ref_∇maxpool(dy, x, pdims)
end

@testset "meanpool count_include_pad (issue #218)" begin
    # Brute-force reference: average over the window, optionally excluding padding.
    function ref_meanpool(x, k; pad, stride, count_include_pad)
        N = ndims(x) - 2
        pdims = PoolDims(x, k; padding=NNlib.expand(Val(N), pad), stride=NNlib.expand(Val(N), stride))
        out = NNlib.meanpool(x, pdims)  # used only for output sizing/type
        osz = size(out)
        kk = NNlib.kernel_size(pdims)
        P = NNlib.padding(pdims)        # (w_lo, w_hi, h_lo, h_hi, d_lo, d_hi)
        S = NNlib.stride(pdims)
        isz = size(x)[1:N]
        for batch in 1:size(x, ndims(x)), c in 1:size(x, ndims(x)-1)
            for oidx in CartesianIndices(osz[1:N])
                acc = zero(eltype(x)); cnt = 0
                for kidx in CartesianIndices(kk)
                    iin = ntuple(N) do d
                        (Tuple(oidx)[d] - 1) * S[d] - P[2d-1] + Tuple(kidx)[d]
                    end
                    valid = all(d -> 1 <= iin[d] <= isz[d], 1:N)
                    if valid
                        acc += x[iin..., c, batch]; cnt += 1
                    end
                end
                denom = count_include_pad ? prod(kk) : cnt
                out[Tuple(oidx)..., c, batch] = denom == 0 ? zero(eltype(x)) : acc / denom
            end
        end
        return out
    end

    # Known small example (2x2, pad=1, stride=2), verified by hand
    x = reshape(Float64.(1:16), 4, 4, 1, 1)
    ye = meanpool(x, (2,2); pad=1, stride=2, count_include_pad=false)
    @test ye[:,:,1,1] ≈ [1.0 7.0 13.0; 2.5 8.5 14.5; 4.0 10.0 16.0]

    # Default is unchanged (include padding) — non-breaking
    @test meanpool(x, (2,2); pad=1, stride=2) ==
          meanpool(x, (2,2); pad=1, stride=2, count_include_pad=true)

    # With no padding the two modes coincide
    xn = rand(8, 8, 2, 2)
    @test meanpool(xn, (2,2)) == meanpool(xn, (2,2); count_include_pad=false)

    # Match the brute-force reference across spatial ranks and configs
    for T in (Float32, Float64), nsd in (1, 2, 3)
        x = rand(T, fill(7, nsd)..., 2, 2)
        for (k, pad, stride) in ((2, 1, 2), (3, 1, 1), (3, 2, 2))
            kt = ntuple(_ -> k, nsd)
            for cip in (true, false)
                y = meanpool(x, kt; pad=pad, stride=stride, count_include_pad=cip)
                yref = ref_meanpool(x, kt; pad=pad, stride=stride, count_include_pad=cip)
                @test y ≈ yref
            end
        end
    end
end

@testset "complex meanpool (issue #610)" begin
    # Mean pooling is linear, so complex support is implemented once in core by
    # pooling the real and imaginary parts independently (and is shared by every
    # backend). `maxpool` is not supported for complex inputs, since `max` is
    # undefined for complex numbers.
    for nsd in (1, 2, 3)
        x = rand(ComplexF64, fill(7, nsd)..., 3, 2)
        for pdims in (PoolDims(x, 2), PoolDims(x, 2; padding=1, stride=2))
            for cip in (true, false)
                y = meanpool(x, pdims; count_include_pad=cip)
                @test eltype(y) == ComplexF64
                # pooling commutes with taking real/imaginary parts
                @test real(y) ≈ meanpool(real(x), pdims; count_include_pad=cip)
                @test imag(y) ≈ meanpool(imag(x), pdims; count_include_pad=cip)

                # `∇meanpool` also splits over real/imaginary parts
                dy = rand(ComplexF64, size(y))
                dx = ∇meanpool(dy, y, x, pdims; count_include_pad=cip)
                @test eltype(dx) == ComplexF64
                @test real(dx) ≈ ∇meanpool(real(dy), real(y), real(x), pdims; count_include_pad=cip)
                @test imag(dx) ≈ ∇meanpool(imag(dy), imag(y), imag(x), pdims; count_include_pad=cip)

                # AD goes through the same split (the `meanpool` rrule is real-only):
                # check against a finite-difference-free reference built from the
                # real-input rrule on each part.
                g = gradient(z -> abs2(sum(meanpool(z, pdims; count_include_pad=cip))), x)[1]
                gref = gradient(z -> abs2(sum(meanpool(z, pdims; count_include_pad=cip))),
                                real(x))[1] .+
                       im .* gradient(z -> abs2(sum(meanpool(z, pdims; count_include_pad=cip))),
                                      imag(x))[1]
                @test eltype(g) == ComplexF64
                @test g ≈ gref
            end
        end

        # maxpool has no complex extension (max is undefined for complex)
        @test_throws Exception maxpool(x, PoolDims(x, 2))
    end
end

@testset "ceil_mode (issue #839)" begin
    # Brute-force reference, checked against PyTorch: window `o` starts at
    # `(o - 1) * stride - pad_lo`, and its positions past the padded input count neither
    # as input nor as padding. Returns `y` and the gradient of `sum(y .* dy)`.
    function ref_pool(op, x, dy, pdims, ceil_mode; count_include_pad=true, p=2)
        N = ndims(x) - 2
        K, S, D = NNlib.kernel_size(pdims), NNlib.stride(pdims), NNlib.dilation(pdims)
        lo, hi = NNlib.padding(pdims)[1:2:end], NNlib.padding(pdims)[2:2:end]
        I = size(x)[1:N]
        O = ntuple(N) do i
            m = I[i] + lo[i] + hi[i] - D[i] * (K[i] - 1) - 1
            ceil_mode || return fld(m, S[i]) + 1
            o = cld(m, S[i]) + 1
            return (o - 1) * S[i] >= I[i] + lo[i] ? o - 1 : o
        end
        y = zeros(eltype(x), O..., size(x)[N+1:end]...)
        dx = zero(x)
        for b in axes(x, N + 2), c in axes(x, N + 1), o in CartesianIndices(O)
            ps = [ntuple(i -> (o[i] - 1) * S[i] - lo[i] + (kk[i] - 1) * D[i] + 1, N)
                  for kk in CartesianIndices(K)]
            inx = [q for q in ps if all(i -> 1 <= q[i] <= I[i], 1:N)]
            xs = [x[q..., c, b] for q in inx]
            if op == :max
                y[o, c, b], imax = findmax(xs)
                dx[inx[imax]..., c, b] += dy[o, c, b]
            elseif op == :mean
                n = count_include_pad ? count(q -> all(i -> q[i] <= I[i] + hi[i], 1:N), ps) : length(inx)
                y[o, c, b] = sum(xs) / n
                foreach(q -> dx[q..., c, b] += dy[o, c, b] / n, inx)
            else
                y[o, c, b] = sum(xs .^ p)^(1 / p)
                foreach(q -> dx[q..., c, b] += dy[o, c, b] * (x[q..., c, b] / y[o, c, b])^(p - 1), inx)
            end
        end
        return y, dx
    end

    # Values from PyTorch
    x = reshape(Float64.(1:6), 6, 1, 1)
    @test vec(maxpool(x, (3,); pad=1, stride=2, ceil_mode=true)) == [2, 4, 6, 6]
    @test vec(meanpool(x, (3,); pad=1, stride=2, ceil_mode=true)) ≈ [1, 3, 5, 3]
    @test vec(meanpool(x, (3,); pad=1, stride=2, ceil_mode=true, count_include_pad=false)) ≈ [1.5, 3, 5, 6]
    x = reshape(Float64.(1:30), 5, 6, 1, 1)
    @test maxpool(x, (3, 3); pad=1, stride=2, ceil_mode=true)[:, :, 1, 1] ==
          [7 17 27 27; 9 19 29 29; 10 20 30 30]
    pdims = PoolDims(x, 3; padding=1, stride=2, ceil_mode=true)
    y = meanpool(x, pdims)
    @test y[:, :, 1, 1] ≈ [16/9 69/9 129/9 53/6; 33/9 13 23 14; 28/9 87/9 147/9 59/6]
    @test ∇meanpool(one.(y), y, x, pdims)[:, :, 1, 1] ≈
          [1 2 1 2 1 2.5; 2 4 2 4 2 5; 1 2 1 2 1 2.5; 2 4 2 4 2 5; 1 2 1 2 1 2.5] ./ 9
    y = meanpool(x, pdims; count_include_pad=false)
    @test y[:, :, 1, 1] ≈ [4 11.5 21.5 26.5; 5.5 13 23 28; 7 14.5 24.5 29.5]

    # Output sizes: the last window must start inside the input or the low padding
    @test NNlib.output_size(PoolDims((5, 1, 1), 2; ceil_mode=true)) == (3,)
    @test NNlib.output_size(PoolDims((5, 1, 1), 2; padding=1, ceil_mode=true)) == (3,)
    @test NNlib.output_size(PoolDims((5, 1, 1), 3; stride=3, padding=(2, 0), ceil_mode=true)) == (3,)
    @test NNlib.output_size(PoolDims((5, 1, 1), 3; stride=3, padding=(0, 2), ceil_mode=true)) == (2,)
    @test size(maxpool(rand(7, 8, 2, 3), (2, 2); ceil_mode=true)) == (4, 4, 2, 3)

    # `ceil_mode` is kept where it changes the output size, and dropped elsewhere
    pdims = PoolDims((5, 5, 1, 1), 2; ceil_mode=true)
    @test NNlib.ceil_mode(pdims)
    @test pdims != PoolDims((5, 5, 1, 1), 2)
    @test occursin("ceil_mode: true", sprint(show, pdims))
    @test NNlib.ceil_mode(NNlib.insert_singleton_spatial_dimension(pdims))
    @test PoolDims((6, 6, 1, 1), 2; ceil_mode=true) == PoolDims((6, 6, 1, 1), 2)
    @test !NNlib.ceil_mode(PoolDims((5, 5, 1, 1), 2; ceil_mode=false))
    @test !occursin("ceil_mode", sprint(show, PoolDims((5, 5, 1, 1), 2)))
    # The positional constructors keep working, without `ceil_mode`
    @test PoolDims((5, 5), (2, 2), 1, (2, 2), (0, 0, 0, 0), (1, 1)) == PoolDims((5, 5, 1, 1), 2)
    @test PoolDims{2, 2, 2, 4, 2}((5, 5), (2, 2), 1, (2, 2), (0, 0, 0, 0), (1, 1)) ==
          PoolDims((5, 5, 1, 1), 2)

    # Match the brute-force reference, forward and backward, also with asymmetric padding
    # and dilation (which PyTorch doesn't have for mean and Lp pooling)
    for nsd in (1, 2, 3), (k, stride, pad, dil) in (
            (2, 2, 0, 1), (3, 2, 1, 1), (3, 2, (1, 0), 1), (3, 2, (0, 2), 1),
            (2, 3, 1, 2), (3, 3, 1, 1), (1, 2, 0, 1), (3, 1, (2, 1), 2))
        sz = (7, 6, 5)[1:nsd]
        padding = pad isa Tuple ? ntuple(i -> pad[mod1(i, 2)], 2nsd) : pad
        x = Float64.(reshape(Random.randperm(rng, prod(sz) * 4), sz..., 2, 2))
        for ceil_mode in (false, true)
            pdims = PoolDims(x, k; stride, padding, dilation=dil, ceil_mode)
            y = maxpool(x, pdims)
            dy = randn(rng, size(y))
            yr, dxr = ref_pool(:max, x, dy, pdims, ceil_mode)
            @test y == yr
            @test ∇maxpool(dy, y, x, pdims) ≈ dxr
            for cip in (true, false)
                y = meanpool(x, pdims; count_include_pad=cip)
                yr, dxr = ref_pool(:mean, x, dy, pdims, ceil_mode; count_include_pad=cip)
                @test y ≈ yr
                @test ∇meanpool(dy, y, x, pdims; count_include_pad=cip) ≈ dxr
            end
            for p in (1, 2.5)
                y = lpnormpool(x, pdims; p)
                yr, dxr = ref_pool(:lp, x, dy, pdims, ceil_mode; p)
                @test y ≈ yr
                @test ∇lpnormpool(dy, y, x, pdims; p) ≈ dxr
            end
        end
    end

    # Gradients
    for nsd in (1, 2)
        x = Float64.(reshape(Random.randperm(rng, 6^nsd * 4), fill(6, nsd)..., 2, 2))
        for pdims in (PoolDims(x, 3; stride=2, ceil_mode=true),
                      PoolDims(x, 3; padding=1, stride=2, ceil_mode=true))
            @test NNlib.ceil_mode(pdims)
            @test test_gradients(z -> maxpool(z, pdims), x)
            @test test_gradients(z -> meanpool(z, pdims), x)
            @test test_gradients(z -> meanpool(z, pdims; count_include_pad=false), x)
            @test test_gradients(z -> lpnormpool(z, pdims; p=2), x)
        end
        k = ntuple(_ -> 3, nsd)
        @test test_gradients(z -> meanpool(z, k; pad=1, stride=2, ceil_mode=true), x)
    end
end

# The low and high padded regions of `calc_padding_regions` used to overlap when one
# window spills over both ends, and the gradient went twice into it
@testset "∇pool with a window wider than the padded input" begin
    x = reshape([5.0], 1, 1, 1)
    pdims = PoolDims(x, 3; padding=1, stride=1)
    @test ∇maxpool(ones(1, 1, 1), maxpool(x, pdims), x, pdims) == ones(1, 1, 1)
    @test ∇meanpool(ones(1, 1, 1), meanpool(x, pdims), x, pdims) ≈ fill(1/3, 1, 1, 1)
end

@testset "AutoDiff: spatial_rank=$spatial_rank" for spatial_rank in (1, 2)
  # Use an input with distinct, well-separated values so that every pooling window has
  # a unique maximum. `maxpool` is non-differentiable at ties, where finite differences
  # and AD disagree; with random input this makes the `maxpool` gradtests flaky (this is
  # what the now-closed #188 was about). Distinct integers (gaps >= 1 >> the FD step)
  # keep the argmax stable, so the comparison is well-posed for all spatial ranks.
  x = Float64.(reshape(Random.randperm(rng, 10^spatial_rank * 3 * 2), repeat([10], spatial_rank)..., 3, 2))
  pdims = PoolDims(x, 2)
  @test test_gradients(x -> maxpool(x, pdims), x)
  @test test_gradients(x -> meanpool(x, pdims), x)
  @test test_gradients(x -> sum(maxpool(x, pdims)), x)
  @test test_gradients(x -> sum(meanpool(x, pdims)), x)

  k = ntuple(_ -> 2, spatial_rank)  # Kernel size of pool in ntuple format
  @test test_gradients(x -> maxpool(x, k), x)
  @test test_gradients(x -> meanpool(x, k), x)
  @test test_gradients(x -> sum(maxpool(x, k)), x)
  @test test_gradients(x -> sum(meanpool(x, k)), x)

  # count_include_pad=false (issue #218), with padding so the modes differ
  k1 = ntuple(_ -> 2, spatial_rank)
  xp = rand(rng, repeat([7], spatial_rank)..., 3, 2)
  pdims_p = PoolDims(xp, 2; padding=1, stride=2)
  @test test_gradients(z -> meanpool(z, pdims_p; count_include_pad=false), xp)
  @test test_gradients(z -> sum(meanpool(z, k1; pad=1, stride=2, count_include_pad=false)), xp)
end

@static if NNLIB_TEST_ENZYME

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