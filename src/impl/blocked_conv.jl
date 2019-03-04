using BenchmarkTools
using NNlib
using LinearAlgebra
using Primes
using SIMD
using Flux
# using CuArrays

BLAS.set_num_threads(4)



# unoptimized blocking
function block(x, block_axis = 3, block_len = 8)
    @assert size(x)[block_axis]%block_len == 0
    shape = [i for i in size(x)]
    shape[block_axis] /= block_len
    insert!(shape, block_axis, block_len)
    permute = vcat([block_axis], [i for i=1:length(shape) if i != block_axis])
    permutedims(reshape(x,Tuple(shape)), permute)
end

function deblock(x, block_axis = 3)
    permute = [i for i = 2:length(size(x))]
    insert!(permute, block_axis, 1)
    shape = [size(x)[i] for i = 2:length(size(x))]
    shape[block_axis] *= size(x)[1]
    reshape(permutedims(x, permute), Tuple(shape))
end

##Iteration indicies, outer to inner:
# batch - n
# out_channels - j
# in channels - i
# out height - hₒ
# out width - wₒ
# filter height - hₖ
# filter width - wₖ
# filter depth - dₖ
# out depth - dₒ
# in blocked channels - ii
# out blocked channels (simd'd), jj
function blocked_conv2d_inner_loop!(Out::Array{T,5},
                                    X::Array{T,5},
                                    W::Array{T,6},
                                    ol::Int64,
                                    ::Type{Vec{B,T}},
                                    pad = 0,
                                    stride = 1,
                                    dilation = 1) where {B,T}
    cₒ, cᵢ, Wₖ, Hₖ, Cᵢ, Cₒ = size(W)
    cₒ, Wₒ, Hₒ, Cₒ, N = size(Out)
    cᵢ, Wᵢ, Hᵢ, Cᵢ, N = size(X)

    # get fused loop indexes
    ool = ol - 1
    n = div(ool, Cₒ * Cᵢ * Hₒ)
    ool -= (n) *  Cₒ * Cᵢ * Hₒ
    j =  div(ool, Cᵢ * Hₒ)
    ool -= (j) * Cᵢ * Hₒ
    i =  div(ool,  Hₒ)
    ool -= i *  Hₒ
    hₒ =  ool

    n += 1
    j += 1
    i += 1
    hₒ += 1

    @inbounds for hₖ = 1:Hₖ, wₖ = 1:Wₖ, wₒ = 1:Wₒ
        # pre-calculate indexes for the inner loop
        hᵢ = 1 + stride * (hₒ - 1) + dilation * (hₖ - 1) - pad
        wᵢ = 1 + stride * (wₒ - 1) + dilation * (wₖ - 1) - pad
        # Check for over-input TODO(mbrookhart): move to compile step
        if (hᵢ < 1 || wᵢ < 1 || hᵢ > Hᵢ || wᵢ > Wᵢ)
            continue
        end
        F_w = Wₖ - (wₖ - 1)
        F_h = Hₖ - (hₖ - 1)
        @inbounds for ii = 1:B
            tmpI = Vec{8, T}(X[ii, wᵢ, hᵢ, i, n])
            tmpO = vload(Vec{B, T}, view(Out, :, wₒ, hₒ, j, n), 1)
            tmpW = vload(Vec{B, T}, view(W, :, ii, F_w, F_h, i, j), 1)
            tmpOut = fma(tmpI, tmpW, tmpO)
            vstore(tmpOut, view(Out, :, wₒ, hₒ, j, n), 1)
        end
    end
end

function blocked_conv2d!(Out::Array{T,5}, X::Array{T,5}, W::Array{T,6}, pad = 0, stride = 1, dilation = 1) where T<:Number
    @assert size(Out)[1] == size(W)[1]
    @assert size(X)[1] == size(W)[2]
    ## Fuse a few outer loops to make sure we have enough jobs for the threads
    ## Most important if it's a low batch size kernel
    out_loop_size = size(Out)[5] * size(W)[5] * size(W)[6] * size(Out)[3]
    @inbounds Threads.@threads for ol = 1:out_loop_size
        blocked_conv2d_inner_loop!(Out, X, W, ol, Vec{size(X)[1],T}, pad, stride, dilation)
    end
end

function blocked_conv2d(X::Array{T,5}, W::Array{T,6}; pad = 0, stride = 1, dilation = 1) where T<:Number
    @assert size(X)[1] == size(W)[2]
    @assert size(X)[4] == size(W)[5]
    cₒ, cᵢ, Wₖ, Hₖ, Cᵢ, Cₒ = size(W)
    cᵢ, Wᵢ, Hᵢ, Cᵢ, N = size(X)
    Wₒ = 1 + div(Wᵢ - (dilation * (Wₖ - 1) + 1) + 2 * pad, stride)
    Hₒ = 1 + div(Hᵢ - (dilation * (Hₖ - 1) + 1) + 2 * pad, stride)

    Out = zeros(T, cₒ, Wₒ, Hₒ, Cₒ, N)
    blocked_conv2d!(Out, X, W, pad, stride, dilation)
    Out
end


for im_size = [32, 64, 128, 192]
    for k_size = [5]
        for pad = [3], stride = [2], dilation = [2]
            X = rand(Float32, im_size, im_size, 32, 128)
            W = rand(Float32, k_size, k_size, 32, 16)

            println("Data Shape: $(size(X))")
            println("Weight Shape: $(size(W))")
            println("pad=$pad, stride=$stride, dilation=$dilation")

            # print("block_data: ")
            # @btime block($X)
            # print("block_weights: ")
            # @btime block(block($W,3),5)

            bX = block(X)
            bW = block(block(W,3),5)

            print("blocked_conv2d: ")
            @btime Out1 = blocked_conv2d($bX, $bW, pad = $pad, stride = $stride, dilation = $dilation)
            print("NNlib.conv: ")
            @btime Out2 = NNlib.conv($gpu($X), $gpu($W), pad = $pad, stride = $stride, dilation = $dilation)

            Out1 = blocked_conv2d(bX, bW, pad = pad, stride = stride, dilation = dilation)

            Out2 = NNlib.conv(X, W, pad = pad, stride = stride, dilation = dilation)
            @assert isapprox(deblock(Out1), Out2)
            println()
        end
    end
end
