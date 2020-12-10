@testset "conv_bias_act" begin
    x = rand(4,4,3,3)
    w = rand(2,2,3,3)
    b = rand(1,1,1,3)
    cdims = DenseConvDims(x, w; stride=2)
    @test NNlib.conv_bias_act(x, w, cdims, b, relu) ≈ relu.(conv(x, w, cdims) .+ b) atol=1e-5
end
