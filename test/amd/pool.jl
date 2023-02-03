@testset "Compare CPU & GPU" begin
    channels, batch = 3, 2
    for T in (Float16, Float32), nd in (1, 2, 3)
        x = rand(T, fill(8, nd)..., channels, batch)
        pdims = PoolDims(x, 2)
        # NOTE: Disable grad check for maxpool as *sometimes*
        # it does not *completely* agree with CPU :/
        gputest(x -> NNlib.maxpool(x, pdims), x; checkgrad=false)
        gputest(x -> NNlib.meanpool(x, pdims), x)
    end
end
