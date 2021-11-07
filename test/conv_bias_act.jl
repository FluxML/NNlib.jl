using NNlib, ForwardDiff, Zygote
using NNlib: conv_bias_act, conv_bias_act!

@testset "conv_bias_act" begin
    x1 = rand(6,3,7)
    w1 = rand(2,3,10) # Conv((2,), 3 => 10)
    b1 = rand(1,10,1)
    cdims1 = DenseConvDims(x1, w1)
    for act in [identity, relu, tanh, cbrt]
        @test conv_bias_act(x1, w1, cdims1, b1, act) ≈ act.(conv(x1, w1, cdims1) .+ b1) atol=1e-5
        @test conv_bias_act(x1, w1, cdims1, false, act) ≈ act.(conv(x1, w1, cdims1) .+ 0) atol=1e-5

        g1x = ForwardDiff.gradient(x -> sum(sin, conv_bias_act(x, w1, cdims1, b1, act)), x1)
        @test g1x ≈ Zygote.gradient(x -> sum(sin, conv_bias_act(x, w1, cdims1, b1, act)), x1)[1]
        @test g1x ≈ Zygote.gradient(x -> sum(sin, conv_bias_act(x, w1, cdims1, false, act)), x1)[1]

        g1w = ForwardDiff.gradient(w -> sum(sin, conv_bias_act(x1, w, cdims1, b1, act)), w1)
        @test g1w ≈ Zygote.gradient(w -> sum(sin, conv_bias_act(x1, w, cdims1, b1, act)), w1)[1]

        g1b = ForwardDiff.gradient(b -> sum(sin, conv_bias_act(x1, w1, cdims1, b, act)), b1)
        @test g1b ≈ Zygote.gradient(b -> sum(sin, conv_bias_act(x1, w1, cdims1, b, act)), b1)[1]

        @test nothing == Zygote.gradient(b -> sum(sin, conv_bias_act(x1, w1, cdims1, b, act)), false)[1]
    end

    x = randn(9,9,2,5)
    w = randn(3,3,2,4) # Conv((3,3), 2 => 4, stride=2) #, pad=1)
    b = randn(1,1,5,1)
    cdims = DenseConvDims(x, w; stride=2)#, pad=1)
    for act in [identity, relu, tanh, cbrt]
        @test conv_bias_act(x, w, cdims, b, act) ≈ act.(conv(x, w, cdims) .+ b) atol=1e-5

        g2x = ForwardDiff.gradient(x -> sum(sin, conv_bias_act(x, w, cdims, b, act)), x)
        @test g2x ≈ Zygote.gradient(x -> sum(sin, conv_bias_act(x, w, cdims, b, act)), x)[1]
    end
end
