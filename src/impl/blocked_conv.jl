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
# out depth - dₒ
# in channels - i
# filter depth - dₖ
# filter height - hₖ
# filter width - wₖ
# out height - hₒ
# out width - wₒ
# in blocked channels - ii
# out blocked channels (simd'd), jj
function blocked_conv_inner_loop!(Out::Array{T,6},
                                  X::Array{T,6},
                                  W::Array{T,7},
                                  ol::Int64,
                                  ::Type{Vec{B,T}},
                                  cdims::DenseConvDims) where {B,T}
    cₒ, cᵢ, Wₖ, Hₖ, Dₖ, Cᵢ, Cₒ = size(W)
    cₒ, Wₒ, Hₒ, Dₒ, Cₒ, N = size(Out)
    cᵢ, Wᵢ, Hᵢ, Dᵢ, Cᵢ, N = size(X)
    p = padding(cdims)
    s = stride(cdims)
    d = dilation(cdims)
    padded_regions, central_region = calc_padding_regions(cdims)
    # get fused loop indexes
    ool = ol - 1
    n = div(ool, Cₒ)
    ool -= (n) * Cₒ
    j =  ool

    n += 1
    j += 1

    #calculate the central region without conditionals
    w_region, h_region, d_region = central_region
    @inbounds for i = 1:Cᵢ, dₒ = d_region, hₒ = h_region, dₖ = 1:Dₖ, hₖ = 1:Hₖ, wₖ = 1:Wₖ, wₒ = w_region
        # pre-calculate indexes for the inner loop
        dᵢ = 1 + s[3] * (dₒ - 1) + d[3] * (dₖ - 1) - p[5]
        hᵢ = 1 + s[2] * (hₒ - 1) + d[2] * (hₖ - 1) - p[3]
        wᵢ = 1 + s[1] * (wₒ - 1) + d[1] * (wₖ - 1) - p[1]

        F_w = Wₖ - (wₖ - 1)
        F_h = Hₖ - (hₖ - 1)
        F_d = Dₖ - (dₖ - 1)
        @inbounds for ii = 1:B
            tmpI = Vec{8, T}(X[ii, wᵢ, hᵢ, dᵢ, i, n])
            tmpO = vload(Vec{B, T}, view(Out, :, wₒ, hₒ, dₒ, j, n), 1)
            tmpW = vload(Vec{B, T}, view(W, :, ii, F_w, F_h, F_d, i, j), 1)
            tmpOut = fma(tmpI, tmpW, tmpO)
            vstore(tmpOut, view(Out, :, wₒ, hₒ, dₒ, j, n), 1)
        end
    end

    #calculate the regions with conditionals
    @inbounds for (w_region, h_region, d_region) in padded_regions
        @inbounds for i =1:Cᵢ, dₒ = d_region, hₒ = h_region, dₖ = 1:Dₖ, hₖ = 1:Hₖ, wₖ = 1:Wₖ, wₒ = w_region
            # pre-calculate indexes for the inner loop
            dᵢ = 1 + s[3] * (dₒ - 1) + d[3] * (dₖ - 1) - p[5]
            hᵢ = 1 + s[2] * (hₒ - 1) + d[2] * (hₖ - 1) - p[3]
            wᵢ = 1 + s[1] * (wₒ - 1) + d[1] * (wₖ - 1) - p[1]
            # Check for over-input
            if (hᵢ < 1 || wᵢ < 1 || dᵢ < 1 || hᵢ > Hᵢ || wᵢ > Wᵢ || dᵢ > Dᵢ)
                continue
            end
            F_w = Wₖ - (wₖ - 1)
            F_h = Hₖ - (hₖ - 1)
            F_d = Dₖ - (dₖ - 1)
            @inbounds for ii = 1:B
                tmpI = Vec{8, T}(X[ii, wᵢ, hᵢ, dᵢ, i, n])
                tmpO = vload(Vec{B, T}, view(Out, :, wₒ, hₒ, dₒ, j, n), 1)
                tmpW = vload(Vec{B, T}, view(W, :, ii, F_w, F_h, F_d, i, j), 1)
                tmpOut = fma(tmpI, tmpW, tmpO)
                vstore(tmpOut, view(Out, :, wₒ, hₒ, dₒ, j, n), 1)
            end
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
    out_loop_size = size(Out)[6] * size(Out)[5]
    @inbounds Threads.@threads for ol = 1:out_loop_size
        blocked_conv_inner_loop!(Out, X, W, ol, Vec{size(X)[1],T}, cdims)
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
