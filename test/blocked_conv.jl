using BenchmarkTools
using Test
using NNlib


BLAS.set_num_threads(4)

function test_blocked_conv(im_size,
                           k_size,
                           rank,
                           pad,
                           stride,
                           dilation;
                           benchmark = false)
    X_shape = vcat([im_size for i in 1:rank], [32, 128])
    W_shape = vcat([k_size for i in 1:rank], [32, 16])

    X = rand(Float32, X_shape...)
    W = rand(Float32, W_shape...)

    bX = block(X, rank + 1)
    bW = block(block(W, rank + 1), rank + 3)


    if benchmark
        println("Data Shape: $(size(X))")
        println("Weight Shape: $(size(W))")
        println("pad=$pad, stride=$stride, dilation=$dilation")
        # print("block_data: ")
        # @btime block($X, $(rank + 1))
        # print("block_weights: ")
        # @btime block(block($W, $(rank + 1)), $(rank + 3))



        print("blocked_conv2d: ")
        @btime Out1 = blocked_conv($bX, $bW, pad = $pad, stride = $stride, dilation = $dilation)
        # print("NNlib.conv: ")
        # @btime Out2 = NNlib.conv($X, $W, pad = $pad, stride = $stride, dilation = $dilation)
    end

    Out1 = blocked_conv(bX, bW, pad = pad, stride = stride, dilation = dilation)

    # Out2 = NNlib.conv(X, W, pad = pad, stride = stride, dilation = dilation)
    # @test isapprox(deblock(Out1, rank + 1), Out2)
    println()
end

do_benchmarking = true

for im_size = [32, 64, 128, 192]
    for k_size = [5]
        for pad = [3], stride = [2], dilation = [2]
            # test_blocked_conv(im_size, k_size, 1, pad, stride, dilation, benchmark = do_benchmarking)
            test_blocked_conv(im_size, k_size, 2, pad, stride, dilation, benchmark = do_benchmarking)
            if im_size <= 32
                # test_blocked_conv(im_size, k_size, 3, pad, stride, dilation, benchmark = do_benchmarking)
            end
        end
    end
end
