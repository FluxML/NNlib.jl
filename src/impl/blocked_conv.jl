export blocked_conv, block, deblock
using SIMD

@inline function remove_singleton_spatial_dimension(x::AbstractArray)
    return reshape(x, size(x)[1:end-3]..., size(x)[end-1:end]...)
end

@inline function remove_singleton_spatial_dimension(x, reps::Int)
    for r in 1:reps
        x = remove_singleton_spatial_dimension(x)
    end
    return x
end

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
function blocked_conv_inner_loop!(Out::Array{T,6},
                                  X::Array{T,6},
                                  W::Array{T,7},
                                  ol::Int64,
                                  ::Type{Vec{B,T}},
                                  pad::NTuple{6,Int64},
                                  stride::NTuple{3,Int64},
                                  dilation::NTuple{3,Int64}) where {B,T}
    cₒ, cᵢ, Dₖ, Wₖ, Hₖ, Cᵢ, Cₒ = size(W)
    cₒ, Dₒ, Wₒ, Hₒ, Cₒ, N = size(Out)
    cᵢ, Dᵢ, Wᵢ, Hᵢ, Cᵢ, N = size(X)

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

    @inbounds for wₒ = 1:Wₒ, hₖ = 1:Hₖ, wₖ = 1:Wₖ, dₖ = 1:Dₖ, dₒ = 1:Dₒ
        # pre-calculate indexes for the inner loop
        hᵢ = 1 + stride[3] * (hₒ - 1) + dilation[3] * (hₖ - 1) - pad[5]
        wᵢ = 1 + stride[2] * (wₒ - 1) + dilation[2] * (wₖ - 1) - pad[3]
        dᵢ = 1 + stride[1] * (dₒ - 1) + dilation[1] * (dₖ - 1) - pad[1]
        # Check for over-input TODO(mbrookhart): move to compile step?
        if (hᵢ < 1 || wᵢ < 1 || dᵢ < 1 || hᵢ > Hᵢ || wᵢ > Wᵢ || dᵢ > Dᵢ)
            continue
        end
        F_d = Dₖ - (dₖ - 1)
        F_w = Wₖ - (wₖ - 1)
        F_h = Hₖ - (hₖ - 1)
        @inbounds for ii = 1:B
            tmpI = Vec{8, T}(X[ii, dᵢ, wᵢ, hᵢ, i, n])
            tmpO = vload(Vec{B, T}, view(Out, :, dₒ, wₒ, hₒ, j, n), 1)
            tmpW = vload(Vec{B, T}, view(W, :, ii, F_d, F_w, F_h, i, j), 1)
            tmpOut = fma(tmpI, tmpW, tmpO)
            vstore(tmpOut, view(Out, :, dₒ, wₒ, hₒ, j, n), 1)
        end
    end
end

function blocked_conv!(Out::Array{T,6},
                       X::Array{T,6},
                       W::Array{T,7},
                       cdims::DenseConvDims) where T<:Number
    @assert size(Out)[1] == size(W)[1]
    @assert size(X)[1] == size(W)[2]
    ## Fuse a few outer loops to make sure we have enough jobs for the threads
    ## Most important if it's a low batch size kernel
    out_loop_size = size(Out)[6] * size(W)[6] * size(W)[7] * size(Out)[4]
    @inbounds Threads.@threads for ol = 1:out_loop_size
        blocked_conv_inner_loop!(Out, X, W, ol, Vec{size(X)[1],T},
                                 padding(cdims), stride(cdims), dilation(cdims))
    end
end


function blocked_conv(X::Array{T,6}, W::Array{T,7}, cdims::DenseConvDims) where T<:Number
    Out = zeros(T, size(W,1), output_size(cdims)...,
                   div(channels_out(cdims),size(W, 1)), size(X, 6))
    blocked_conv!(Out, X, W, cdims)
    Out
end

for N in (3, 4)
    @eval begin
        function $(Symbol("blocked_conv!"))(
                        y::AbstractArray{yT,$(N+1)}, x::AbstractArray{xT,$(N+1)},
                        w::AbstractArray{wT,$(N+2)}, cdims::ConvDims) where {yT, xT, wT}
            $(Symbol("blocked_conv!"))(
                insert_singleton_spatial_dimension(y, $(5 - N)),
                insert_singleton_spatial_dimension(x, $(5 - N)),
                insert_singleton_spatial_dimension(w, $(5 - N)),
                insert_singleton_spatial_dimension(cdims, $(5 - N))
            )

            # We explicitly return `y` here, because the backend call
            # itself may return a reshaped view, which we don't want.
            return y
        end
    end
    @eval begin
        function $(Symbol("blocked_conv"))(
                        x::AbstractArray{xT,$(N+1)},
                        w::AbstractArray{wT,$(N+2)}, cdims::ConvDims) where {yT, xT, wT}
            remove_singleton_spatial_dimension(
                $(Symbol("blocked_conv"))(
                    insert_singleton_spatial_dimension(x, $(5 - N)),
                    insert_singleton_spatial_dimension(w, $(5 - N)),
                    insert_singleton_spatial_dimension(cdims, $(5 - N))
                ),
                $(5 - N)
            )
        end
    end
end
