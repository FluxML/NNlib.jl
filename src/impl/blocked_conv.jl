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

function blocked_conv2d_inner_loop!(Out::Array{T,5},
                                    X::Array{T,5},
                                    W::Array{T,6},
                                    ol::Int64,
                                    ::Type{Vec{B,T}},
                                    pad = 0,
                                    stride = 1,
                                    dilation = 1) where {B,T}
    cₒ, cᵢ, Wf, Hf, Cᵢ, Cₒ = size(W)
    cₒ, Wₒ, Hₒ, Cₒ, N = size(Out)
    cᵢ, Wᵢ, Hᵢ, Cᵢ, N = size(X)
    # get fused loop indexes
    ool = ol - 1
    batch = div(ool, Cₒ * Cᵢ * Hₒ)
    ool -= (batch) *  Cₒ * Cᵢ * Hₒ
    j′ =  div(ool, Cᵢ * Hₒ)
    ool -= (j′) * Cᵢ * Hₒ
    i′ =  div(ool,  Hₒ)
    ool -= i′ *  Hₒ
    l =  ool
    batch += 1
    j′ += 1
    i′ += 1
    l += 1
    @inbounds for n = 1:Hf, m = 1:Wf, k′ = 1:Wₒ
        # pre-calculate indexes for the inner loop
        I_w = 1 + stride * (k′ - 1) + dilation * (m - 1) - pad
        I_h = 1 + stride * (l - 1) + dilation * (n - 1) - pad
        if (I_w < 1 || I_h < 1 || I_h > Hᵢ || I_w > Wᵢ)
            continue
        end
        F_w = Wf - (m - 1)
        F_h = Hf - (n - 1)
        @inbounds for ii = 1:B
            tmpI = Vec{8, T}(X[ii, I_w, I_h, i′, batch])
            tmpO = vload(Vec{B, T}, view(Out, :, k′, l, j′, batch), 1)
            tmpW = vload(Vec{B, T}, view(W, :, ii, F_w, F_h, i′, j′), 1)
            tmpOut = fma(tmpI, tmpW, tmpO)
            vstore(tmpOut, view(Out, :, k′, l, j′, batch), 1)
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
    cₒ, cᵢ, Wf, Hf, Cᵢ, Cₒ = size(W)
    cᵢ, Wᵢ, Hᵢ, Cᵢ, N = size(X)
    Wₒ = 1 + div(Wᵢ - (dilation * (Wf - 1) + 1) + 2 * pad, stride)
    Hₒ = 1 + div(Hᵢ - (dilation * (Hf - 1) + 1) + 2 * pad, stride)

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
