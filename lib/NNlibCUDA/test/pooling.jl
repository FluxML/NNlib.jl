@testset "pooling" begin

    # Test for agreement between CPU NNlib and CuDNN versions, across a variety of kwargs
    for num_spatial_dims in (1, 2, 3)
        # Initialize data we'll run our tests over
        C_in = 3
        batch_size = 1
        x = rand(Float64, fill(8, num_spatial_dims)..., C_in, batch_size)
       
        # Test that pooling is equivalent across GPU/CPU
        pdims = PoolDims(x, 2)
        y = maxpool(x, pdims)
        dy = ones(size(y))
        gputest(x -> maxpool(x, pdims), x)
        gputest((dy, y, x) -> ∇maxpool(dy, y, x, pdims), dy, y, x, checkgrad=false)
        gputest(x -> maxpool(x, pdims), x)
        gputest((dy, y, x) -> ∇maxpool(dy, y, x, pdims), dy, y, x, checkgrad=false)

        # Test the compatibility shims for pooling
        cx,cy,cdy = CuArray{Float32}.((x,y,dy))
        win,pad=2,1
        maxpool!(similar(cy), cx, win; pad=pad, stride=win) ≈ maxpool!(similar(cy), cx, PoolDims(cx, win; padding=pad, stride=win))
        meanpool!(similar(cy), cx, win; pad=pad, stride=win) ≈ meanpool!(similar(cy), cx, PoolDims(cx, win; padding=pad, stride=win))
    end
end
