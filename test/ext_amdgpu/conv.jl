@testset "Compare CPU & GPU" begin
    channels, batch = 3, 2
    for T in (Float16, Float32), nd in (1, 2, 3)
        x = rand(Float32, fill(4, nd)..., 3, 1)
        w = rand(Float32, fill(2, nd)..., channels, 4)

        cdims = DenseConvDims(x, w, flipkernel=true)
        gputest((x, w) -> NNlib.conv(x, w, cdims), x, w; atol=1e-4)

        # This one flips manually kernel for AMDGPU.
        cdims = DenseConvDims(x, w)
        gputest((x, w) -> NNlib.conv(x, w, cdims), x, w; atol=1e-4)
    end
end
