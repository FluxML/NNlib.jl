FiniteDifferences.to_vec(stats::NNlib.RunningStats) = [], _ -> stats

randn_sample(shape, μ, σ) = randn(rng, shape) .* σ .+ μ
f32_arange(shape...) = Float32.(reshape(1:prod(shape), shape))

function make_bn(ch; training = true)
    stats, bias, scale = NNlib.RunningStats(zeros(ch), ones(ch), 0.1), zeros(ch), ones(ch)
    return x -> NNlib.batchnorm(x, stats, scale, bias; training)
end
function make_in(ch; training = true)
    stats, bias, scale = NNlib.RunningStats(zeros(ch), ones(ch), 0.1), zeros(ch), ones(ch)
    return x -> NNlib.instancenorm(x, stats, scale, bias; training)
end
function make_gn(ch, groups)
    bias, scale = zeros(ch), ones(ch)
    return x -> NNlib.groupnorm(x, groups, scale, bias)
end

@testset "Helpers" begin
    # BatchNorm dimensions
    let W = 128, C = 4, N = 64
        x = cat([randn_sample((W, W, 1, N), i, i) for i in 1:C]...; dims = 3)
        μ, σ² = NNlib.norm_stats(x, (1, 2, 4))
        @test vec(μ)≈1:C rtol=0.05
        @test vec(σ²)≈abs2.(1:C) rtol=0.05
    end

    # LayerNorm dimensions
    let W = 128, C = 64, N = 4
        x = cat([randn_sample((W, W, C, 1), i, i) for i in 1:N]...; dims = 4)
        μ, σ² = NNlib.norm_stats(x, (1, 2, 3))
        @test vec(μ)≈1:N rtol=0.05
        @test vec(σ²)≈abs2.(1:N) rtol=0.05
    end

    # Group/InstanceNorm dimensions
    let W = 128, C = 2, N = 2, shape = (W, W, 1, 1)
        x = [randn_sample(shape, 1, 1);;; randn_sample(shape, 2, 2);;;;
             randn_sample(shape, 3, 3);;; randn_sample(shape, 4, 4)]
        μ, σ² = NNlib.norm_stats(x, (1, 2))
        @test vec(μ)≈1:(C * N) rtol=0.05
        @test vec(σ²)≈abs2.(1:(C * N)) rtol=0.05
    end

    x = rand(rng, 16, 16, 3, 4)
    @testset "dims = $dims" for (dims, tsize) in [
        (1, 2, 4) => (1, 1, size(x, 3), 1),
        (1, 2, 3) => (1, 1, 1, size(x, 4)),
        (1, 2) => (1, 1, size(x, 3), size(x, 4)),
    ]
        meanvar = (ones(tsize), ones(tsize))
        test_rrule(NNlib.norm_stats, x, dims ⊢ NoTangent(); output_tangent = meanvar)

        running_stats = NNlib.RunningStats(meanvar..., 0.1)
        y_ns, back_ns = rrule(NNlib.norm_stats, x, dims)
        dx_ns = back_ns(meanvar)[2]
        for (stats, training, y, y_ad, dx) in [
            (nothing, true, y_ns, y_ns, dx_ns),
            (nothing, false, y_ns, y_ns, dx_ns),
            (running_stats, true, y_ns, y_ns, dx_ns),
            (running_stats, false, meanvar, meanvar, NoTangent()),
        ]
            @test NNlib.maybe_norm_stats(stats, x, dims, !training) == y
            ŷ, back = rrule(NNlib.maybe_norm_stats, stats, x, dims, !training)
            @test ŷ == y_ad
            @test back(meanvar) == (NoTangent(), NoTangent(), dx, NoTangent(), NoTangent())

            test_rrule(NNlib.maybe_norm_stats, stats ⊢ NoTangent(), x, dims ⊢ NoTangent(),
                       !training; output_tangent = meanvar, check_inferred = false)
        end

        ps = ntuple(_ -> rand(rng, tsize...), 4)
        gradtest((args...) -> NNlib.norm_helper(args..., size(ps[1])), x, ps..., 1e-5)
    end

    p = ones(16, 16)
    @test_throws ErrorException NNlib.norm_helper(x, p, p, nothing, p, 1e-5)
    @test_throws ErrorException NNlib.norm_helper(x, p, p, p, nothing, 1e-5)
end

@testset "Layer Norm" begin
    full_size = (2, 3, 4, 5)
    @testset for xdims in 2:4, kdims in 1:(xdims - 1)
        x = rand(rng, full_size[1:xdims]...)
        bias, scale = ntuple(_ -> rand(rng, full_size[1:kdims]...), 2)
        dims = Val(kdims)

        y = @inferred NNlib.layernorm(x, dims)
        @test size(y) == size(x)
        y = @inferred NNlib.layernorm(x, dims, scale, bias)
        @test size(y) == size(x)

        # FiniteDifferences gives incorrect results on some but not all args, why?
        gradtest(x -> NNlib.layernorm(x, dims), x; broken = true)
        gradtest((x, s, b) -> NNlib.layernorm(x, dims, s, b), x, scale, bias; skip = true)
    end
end

@testset "Batch Norm" begin
    let x = [1.0 3.0 5.0; 2.0 4.0 6.0], bias = zeros(2), scale = ones(2)
        @testset for use_stats in (true, false)
            stats = use_stats ? NNlib.RunningStats(zeros(2), ones(2), 0.1) : nothing
            y, back = Zygote.pullback(NNlib.batchnorm, x, stats, scale, bias, 1e-5)
            @test y≈[-1.22474 0 1.22474; -1.22474 0 1.22474] atol=1e-5

            expected_mean, expected_var = [0.3, 0.4], [1.3, 1.3]
            if use_stats
                # μ of batch will be
                #  (1. + 3. + 5.) / 3 = 3
                #  (2. + 4. + 6.) / 3 = 4
                #
                # ∴ update rule with momentum:
                #  .1 * 3 + 0 = .3
                #  .1 * 4 + 0 = .4
                @test stats.mean ≈ expected_mean
                # σ² of batch will be
                #  sum(abs2, [1., 3., 5.] .- 3) / 2 = 2.6...
                #  sum(abs2, [2., 4., 6.] .- 4) / 2 = 2.6...
                #
                # ∴ update rule with momentum:
                #  .1 * (3 / (3 - 1)) * 2.6 + (1 - .1) * 1 = 1.3
                @test stats.variance ≈ expected_var
            end

            dx, dstats, dscale, dbias, _ = back(fill!(similar(y), 1))
            @test dx≈[3.06186 0.612371 -1.83711; 3.06186 0.612371 -1.83711] atol=1e-5
            @test dscale == zeros(2)
            @test dbias == fill(3.0, 2)
            @test dstats === nothing

            if use_stats
                tmp_mean, tmp_var = copy(stats.mean), copy(stats.variance)
                x′ = @inferred NNlib.batchnorm(x, stats, scale, bias, 1e-5)
                @test x′[1]≈((1 - expected_mean[1]) / sqrt(expected_var[1])) atol=1e-5
                # Stats should be unchanged
                @test stats.mean == tmp_mean
                @test stats.variance == tmp_var
            end
        end
    end

    let x = f32_arange(3, 2, 1), m = make_bn(2)
        y = reshape(permutedims(x, [2, 1, 3]), 2, :)
        y = permutedims(reshape(m(y), 2, 3, 1), [2, 1, 3])
        @test m(x) == y
        @inferred m(x)
    end

    let x = f32_arange(2, 3, 2, 1), m = make_bn(2)
        y = reshape(permutedims(x, [3, 1, 2, 4]), 2, :)
        y = permutedims(reshape(m(y), 2, 2, 3, 1), [2, 3, 1, 4])
        @test m(x) == y
        @inferred m(x)
    end

    let x = f32_arange(2, 2, 3, 2, 1), m = make_bn(2)
        y = reshape(permutedims(x, [4, 1, 2, 3, 5]), 2, :)
        y = permutedims(reshape(m(y), 2, 2, 2, 3, 1), [2, 3, 4, 1, 5])
        @test m(x) == y
        @inferred m(x)
    end

    let x = randn(Float32, 416, 416, 32, 1), m = make_bn(32; training = false)
        @test (@allocated m(x)) < 100_000_000
    end
end

@testset "Instance Norm" begin
    let x = reshape(1.0:12.0, 3, 2, 2), bias = zeros(2), scale = ones(2)
        @testset for use_stats in (true, false)
            stats = use_stats ? NNlib.RunningStats(zeros(2), ones(2), 0.1) : nothing
            y, back = Zygote.pullback(NNlib.instancenorm, x, stats, scale, bias, 1e-5)
            @test y≈[-1.22474 -1.22474; 0.0 0.0; 1.22474 1.22474;;;
                     -1.22474 -1.22474; 0.0 0.0; 1.22474 1.22474] rtol=1e-5

            expected_mean, expected_var = [0.5, 0.8], [1.0, 1.0]
            if use_stats
                # μ will be
                #  (1. + 2. + 3.) / 3 = 2.
                #  (4. + 5. + 6.) / 3 = 5.
                #  (7. + 8. + 9.) / 3 = 8.
                #  (10. + 11. + 12.) / 3 = 11.
                #
                # ∴ update rule with momentum:
                #  .1 * (2. + 8.) / 2 + 0 = .5
                #  .1 * (5. + 11.) / 2 + 0 = .8
                @test stats.mean ≈ expected_mean
                # σ² will be
                #  sum(abs2, [1. + 2. + 3.] .- 2) / 3 = 2.6...
                #  sum(abs2, [4. + 5. + 6.] .- 5) / 3 = 2.6...
                #  sum(abs2, [7. + 8. + 9.] .- 8) / 3 = 2.6...
                #  sum(abs2, [10. + 11. + 12.] .- 11) / 3 = 2.6...
                #
                # ∴ update rule with momentum:
                #  .1 * (3 / (3 - 1)) * 2.6... + (1 - .1) * 1 = 1.
                @test stats.variance ≈ expected_var
            end

            dx, dstats, dscale, dbias, _ = back(fill!(similar(y), 1))
            @test dx≈[3.6742 3.6742; 1.22474 1.22474; -1.22474 -1.22474;;;
                      3.6742 3.6742; 1.22474 1.22474; -1.22474 -1.22474] rtol=1e-5
            @test dscale == zeros(2)
            @test dbias == fill(6.0, 2)
            @test dstats === nothing

            if use_stats
                tmp_mean, tmp_var = copy(stats.mean), copy(stats.variance)
                x′ = @inferred NNlib.instancenorm(x, stats, scale, bias, 1e-5)
                @test x′[1]≈((1 - expected_mean[1]) / sqrt(expected_var[1])) atol=1e-5
                # Stats should be unchanged
                @test stats.mean == tmp_mean
                @test stats.variance == tmp_var
            end
        end
    end

    let m = make_in(2), shape = (2, 4, 1, 2, 3), x = f32_arange(shape...)
        y = reshape(permutedims(x, [3, 1, 2, 4, 5]), :, 2, 3)
        y = reshape(m(y), shape...)
        @test m(x) == y
        @inferred m(x)
    end

    # Instance norm == batch norm when channel and batch dims are squashed
    let m_inorm = make_in(2; training = true), m_bnorm = make_bn(12; training = true),
        shape = (5, 5, 3, 4, 2, 6), x = f32_arange(shape...)

        x′ = reshape(x, (shape[1:(end - 2)]..., :, 1))
        @test m_inorm(x) == reshape(m_bnorm(x′), shape)
    end

    let m = make_in(32), x = randn(Float32, 416, 416, 32, 1)
        @test (@allocated m(x)) < 100_000_000
    end
end

@testset "Group Norm" begin
    full_size = (2, 3, 6, 5)
    @testset for xdims in 1:3, groups in (1, 2, 3, 6)
        x = rand(rng, full_size[(end - xdims):end]...)
        bias, scale = ntuple(_ -> rand(rng, full_size[end - 1]), 2)

        y = @inferred NNlib.groupnorm(x, groups)
        @test size(y) == size(x)
        y = @inferred NNlib.groupnorm(x, groups, scale, bias)
        @test size(y) == size(x)

        # FiniteDifferences gives incorrect results on some but not all args, why?
        gradtest(x -> NNlib.groupnorm(x, groups), x; broken = true)
        gradtest((x, s, b) -> NNlib.groupnorm(x, groups, s, b), x, scale, bias; skip = true)
    end

    let m = make_gn(4, 2), shape = (5, 5, 3, 4, 4, 6)
        y = Zygote.pullback(m, f32_arange(shape...))[1]
        @test size(y) == shape
    end

    let m = make_gn(2, 2), shape = (2, 4, 1, 2, 3), x = f32_arange(shape...)
        y = reshape(permutedims(x, [3, 1, 2, 4, 5]), :, 2, 3)
        y = reshape(m(y), shape...)
        @test m(x) == y
    end

    # Group norm == instance norm when the group size == number of channels
    let m_inorm = make_in(4), m_gnorm = make_gn(4, 4), x = f32_arange(2, 2, 3, 4, 5)
        @test m_inorm(x) ≈ m_gnorm(x)
    end

    # Group norm == batch norm for a group of size 1 and batch of size 1
    let m_bnorm = make_bn(4), m_gnorm = make_gn(4, 4), x = f32_arange(2, 2, 3, 4, 1)
        @test m_bnorm(x) ≈ m_gnorm(x)
    end
end