
@testset "fold" begin
    # Test for agreement between CPU/GPU versions, across a variety of kwargs
    options = Dict{Any, Any}.((
        (), (:dilation => 2), (:flipkernel => true), (:stride => 2),
        (:padding => 1),
        (:padding => (1,0)),
        (:padding => (0,1)),
        (:padding => (2,3)),
    ))

    C_in = 3
    C_out = 4
    batch_size = 1

    @testset "spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
        for opts in options
            if :padding in keys(opts)
                padding = opts[:padding]
                if 1 < length(padding) && length(padding) != 2spatial_rank
                    opts[:padding] = ntuple(i -> padding[mod1(i,2)] .+ 2div(i-1,2), 2spatial_rank)   
                end
            end

            x = rand(Float64, fill(8, spatial_rank)..., C_in, batch_size)
            w = rand(Float64, fill(2, spatial_rank)..., C_in, C_out)
            cdims = DenseConvDims(x, w; opts...)
            y = NNlib.unfold(x, cdims)

            # test equivalence of fold/unfold across GPU/CPU
            gputest(x -> NNlib.unfold(x, cdims), x) 
            gputest(y -> NNlib.fold(y, size(x), cdims), y) 
        end
    end
end

