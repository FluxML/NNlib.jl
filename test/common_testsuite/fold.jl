function fold_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    # Metal supports only Float32 (and needs a looser round-trip tolerance below);
    # other backends keep Float64.
    is_metal = nameof(Backend) === :MetalBackend
    T = is_metal ? Float32 : Float64
    rtol = is_metal ? 1.0e-5 : 1.0e-7

    @testset "unfold wrapper" begin
        x = device(rand(rng, T, 16, 16, 3, 10))
        w = device(rand(rng, T, 5, 5, 3, 2))
        @test size(NNlib.unfold(x, size(w))) == (144, 75, 10)
        @test size(NNlib.unfold(x, size(w); pad=2)) == (256, 75, 10)
        @test size(NNlib.unfold(x, size(w); stride=2)) == (36, 75, 10)
        @test size(NNlib.unfold(x, size(w); dilation=2)) == (64, 75, 10)
    end

    @testset "Inverses: spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
        x = device(rand(rng, T, repeat([8], spatial_rank)..., 3, 2))
        w = device(rand(rng, T, repeat([3], spatial_rank)..., 3, 3))

        cdims = DenseConvDims(x, w; padding=1)
        y = NNlib.unfold(x, cdims)
        z = NNlib.fold(y, size(x), cdims)

        o = device(ones(eltype(x), size(x)...))
        divisor = NNlib.fold(NNlib.unfold(o, cdims), size(x), cdims)
        @test isapprox(z ./ divisor, x; rtol)

        # introduce stride
        cdims = DenseConvDims(x, w; padding=1, stride=2)
        y = NNlib.unfold(x, cdims)
        z = NNlib.fold(y, size(x), cdims)
        divisor = NNlib.fold(NNlib.unfold(o, cdims), size(x), cdims)
        @test isapprox(z ./ divisor, x; rtol)
    end

    @testset "AutoDiff: spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
        x = rand(rng, repeat([5], spatial_rank)..., 3, 2)
        w = rand(rng, repeat([3], spatial_rank)..., 3, 3)
        cdims = DenseConvDims(x, w)

        @test test_gradients(x -> NNlib.unfold(x, cdims), x; test_gpu = Backend != CPU)
        Backend == CPU && ChainRulesTestUtils.test_rrule(NNlib.unfold, x, cdims)

        y = NNlib.unfold(x, cdims)
        @test test_gradients(y -> NNlib.fold(y, size(x), cdims), y; test_gpu = Backend != CPU)
        Backend == CPU && ChainRulesTestUtils.test_rrule(NNlib.fold, y, size(x), cdims)
    end
end
