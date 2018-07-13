using NNlib: conv, ∇conv_filter, ∇conv_data, ∇maxpool, maxpool, depthwiseconv, ∇depthwiseconv_filter, ∇depthwiseconv_data

@testset "conv2d" begin
    x = reshape(Float64[1:20;], 5, 4, 1, 1)
    w = reshape(Float64[1:4;], 2, 2, 1, 1)

    @test squeeze(conv(x, w),(3,4)) == [
        51 101 151;
        61 111 161;
        71 121 171;
        81 131 181]

    @test squeeze(conv(x, w; stride=2),(3,4)) == [
        51 151;
        71 171]

    @test squeeze(conv(x, w; pad=1),(3,4)) == [
        4.0 26.0 56.0 86.0 32.0; 
        11.0 51.0 101.0 151.0 50.0; 
        18.0 61.0 111.0 161.0 53.0; 
        25.0 71.0 121.0 171.0 56.0; 
        32.0 81.0 131.0 181.0 59.0; 
        15.0 35.0 55.0 75.0 20.0]

    @test squeeze(conv(x, w; dilation=2),(3,4)) == [
        92.0 142.0; 
        102.0 152.0;
        112.0 162.0]

	# NaN tests for dilation forward pass

	ys = []
	for idx in 1:1000 		
    	push!(ys, conv(x, w; dilation=2))
	end
	@test !any([any(isnan.(ys[idx])) for idx in 1:1000])

    # for gradients, check only size
    # correctness of gradients is cross-checked with CUDNN.jl
    # (it's assumed convolution code won't change often)

    @test size(∇conv_filter(reshape(rand(4,3), 4, 3, 1, 1), x, w)) == size(w)
    @test size(∇conv_data(reshape(rand(4,3), 4, 3, 1, 1), x, w)) == size(x)

    # Test that stride/pad work backward as well
    y = conv(x, w; stride=2, pad=1, dilation=2)
    @test size(y) == (3, 2, 1, 1)
    @test size(∇conv_filter(y, x, w; stride=2, pad=1, dilation=2)) == size(w)
    @test size(∇conv_data(y, x, w; stride=2, pad=1, dilation=2)) == size(x)

	# NaN tests for dilation backward pass: filters
	dy = randn(size(ys[1]))
	dws = []
	for idx in 1:1000
	    push!(dws, ∇conv_filter(dy, x, w; dilation=2))
	end

	# NaN tests for dilation backward pass: input
	dxs = []
	for idx in 1:1000
	    push!(dxs, ∇conv_data(dy, x, w; dilation=2))
	end

	@test !any([any(isnan.(dws[idx])) for idx in 1:1000])
	@test !any([any(isnan.(dxs[idx])) for idx in 1:1000])

end

@testset "depthwiseconv2d" begin
    x = reshape(Float64[1:18;], 3, 3, 2, 1)
    w = reshape(Float64[1:8;], 2, 2, 1, 2)

    @test depthwiseconv(x, w)[:] == [37.0, 47.0, 67.0, 77.0, 319.0, 345.0, 397.0, 423.0]

    @test depthwiseconv(x, w, stride = 2, pad = 1)[:] == [4.0, 18.0, 36.0, 77.0, 80.0, 173.0, 206.0, 423.0]

    @test depthwiseconv(x, w, stride = 2)[:] == [37.0, 319.0]

    @test depthwiseconv(x, w, pad = 1)[:] == [4.0, 11.0, 18.0, 9.0, 18.0, 37.0, 47.0, 21.0, 36.0, 67.0, 77.0, 33.0, 14.0, 23.0, 26.0, 9.0, 80.0, 158.0, 173.0, 84.0, 164.0, 319.0, 345.0, 165.0, 206.0, 397.0, 423.0, 201.0, 96.0, 182.0, 193.0, 90.0]

    # the correctness of the gradients are being verified by calling
    # the corresponding counvolution gradients

    dy = reshape(Float64[1:8;], 2,2,2,1)
    local z = ∇depthwiseconv_data(dy,x,w)
    for i in 1:2
        X = copy(x[:,:,i:i,:]);
        W = copy(permutedims(w[:,:,:,i:i],[1,2,4,3]));
        DY = copy(dy[:,:,i:i,:]);
        res = ∇conv_data(DY,X,W)
        @test squeeze(z[:,:,i:i,:], (3,4)) == squeeze(res, (3,4))
    end

    z = ∇depthwiseconv_filter(dy, x, w)
    for i in 1:2
        X = copy(x[:,:,i:i,:]);
        W = copy(permutedims(w[:,:,:,i:i],[1,2,4,3]))
        DY = copy(dy[:,:,i:i,:])
        res = ∇conv_filter(DY,X,W)
        @test squeeze(z[:,:,:,i:i], (3,4)) == squeeze(res, (3,4))
    end

    @test size(∇depthwiseconv_filter(rand(2,2,2,1), x, w)) == size(w)
    @test size(∇depthwiseconv_data(rand(2,2,2,1), x, w)) == size(x)

    # Test for the stride/pad for backward pass
    y = depthwiseconv(x,w,stride=2,pad=1)
    @test size(y) == (2,2,2,1)
    @test size(∇depthwiseconv_filter(rand(size(y)), x, w, stride=2, pad=1)) == size(w)
    @test size(∇depthwiseconv_data(rand(size(y)), x, w, stride=2, pad=1)) == size(x)
end

@testset "maxpool2d" begin

    x = reshape(Float64[1:20;], 5, 4, 1, 1)

    @test squeeze(maxpool(x, (2,2)), (3,4)) == [7 17; 9 19]
    @test squeeze(maxpool(x, (2,2); stride=(2,2)), (3,4)) == [7 17; 9 19]
    @test squeeze(maxpool(x, (2,2); pad=(1,1)), (3,4)) == [
        1.0  11.0  16.0;
        3.0  13.0  18.0;
        5.0  15.0  20.0;
    ]

    # for gradients, check only size
    # correctness of gradients is cross-checked with CUDNN.jl
    # (it's assumed maxpooling code won't change often)

    y = maxpool(x, (2,2))
    dy = reshape(rand(2,2), 2, 2, 1, 1)
    @test size(∇maxpool(dy, y, x, (2,2))) == size(x)

end


@testset "conv3d" begin

    x = reshape(Float64[1:60;], 5, 4, 3, 1, 1)
    w = reshape(Float64[1:8;], 2, 2, 2, 1, 1)
    res = zeros(4,3,2)
    res[:, :, 1] = [
        686.0 866.0 1046.0;
        722.0 902.0 1082.0;
        758.0 938.0 1118.0;
        794.0 974.0 1154.0]
    res[:, :, 2] = [
    1406.0 1586.0 1766.0;
    1442.0 1622.0 1802.0;
    1478.0 1658.0 1838.0;
    1514.0 1694.0 1874.0]
    
    @test squeeze(conv(x, w),(4,5)) == res

    @test squeeze(conv(x, w; stride=2),(3,4,5)) == [
        686.0 1046.0; 
        758.0 1118.0]

    res = zeros(6,5,4)
    res[:, :, 1] = [
        8.0 54.0 124.0 194.0 96.0;
        23.0 115.0 245.0 375.0 182.0;
        38.0 141.0 271.0 401.0 193.0;
        53.0 167.0 297.0 427.0 204.0;
        68.0 193.0 323.0 453.0 215.0;
        35.0 95.0 155.0 215.0 100.0]
    res[:, :, 2] = [
        172.0 360.0 460.0 560.0 248.0;
        334.0 686.0 866.0 1046.0 452.0;
        356.0 722.0 902.0 1082.0 466.0;
        378.0 758.0 938.0 1118.0 480.0;
        400.0 794.0 974.0 1154.0 494.0;
        190.0 370.0 450.0 530.0 220.0]
    res[:, :, 3] = [
        412.0 760.0 860.0 960.0 408.0;
        774.0 1406.0 1586.0 1766.0 732.0;
        796.0 1442.0 1622.0 1802.0 746.0;
        818.0 1478.0 1658.0 1838.0 760.0;
        840.0 1514.0 1694.0 1874.0 774.0;
        390.0 690.0 770.0 850.0 340.0]
    res[:, :, 4] = [
        164.0 266.0 296.0 326.0 112.0;
        291.0 451.0 501.0 551.0 170.0;
        298.0 461.0 511.0 561.0 173.0;
        305.0 471.0 521.0 571.0 176.0;
        312.0 481.0 531.0 581.0 179.0;
        135.0 195.0 215.0 235.0 60.0]
    @test squeeze(conv(x, w; pad=1),(4,5)) == res

    @test squeeze(conv(x, w; dilation=2),(3,4,5)) == [
    1336.0 1516.0;
    1372.0 1552.0;
    1408.0 1588.0]

	# NaN tests for dilation forward pass

	ys = []
	for idx in 1:1000 		
    	push!(ys, conv(x, w; dilation=2))
	end
	@test !any([any(isnan.(ys[idx])) for idx in 1:1000])

    # for gradients, check only size
    # correctness of gradients is cross-checked with CUDNN.jl
    # (it's assumed convolution code won't change often)

    @test size(∇conv_filter(reshape(rand(4,3,2), 4, 3, 2, 1, 1), x, w)) == size(w)
    @test size(∇conv_data(reshape(rand(4,3,2), 4, 3, 2, 1, 1), x, w)) == size(x)

	# NaN tests for dilation backward pass: filters
	dy = randn(size(ys[1]))
	dws = []
	for idx in 1:1000
	    push!(dws, ∇conv_filter(dy, x, w; dilation=2))
	end

	# NaN tests for dilation backward pass: input
	dxs = []
	for idx in 1:1000
	    push!(dxs, ∇conv_data(dy, x, w; dilation=2))
	end

	@test !any([any(isnan.(dws[idx])) for idx in 1:1000])
	@test !any([any(isnan.(dxs[idx])) for idx in 1:1000])

end


@testset "maxpool3d" begin

    x = reshape(Float64[1:60;], 5, 4, 3, 1, 1)

    @test squeeze(maxpool(x, (2,2,2)), (3,4,5)) == [27 37; 29 39.]
    @test squeeze(maxpool(x, (2,2,2); stride=(2,2,2)), (3,4,5)) == [27 37; 29 39.]
    res = zeros(3,3,2)
    res[:, :, 1] = [
        1.0  11.0  16.0;
        3.0  13.0  18.0;
        5.0  15.0  20.0]
    res[:, :, 2] = [
        41.0  51.0  56.0;
        43.0  53.0  58.0;
        45.0  55.0  60.0]
    @test squeeze(maxpool(x, (2,2,2), pad=(1,1,1)), (4,5)) == res

    # for gradients, check only size
    # correctness of gradients is cross-checked with CUDNN.jl
    # (it's assumed maxpooling code won't change often)

    y = maxpool(x, (2,2,2))
    dy = reshape(rand(2,2), 2, 2, 1, 1, 1)
    @test size(∇maxpool(dy, y, x, (2,2,2))) == size(x)

end
