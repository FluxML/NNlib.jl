# Defines operations on convolutions
export pixel_shuffle

using Base.Iterators:partition

"""
	pixel_shuffle(x,r)

Pixel shuffling operation. `r` is the scale factor for shuffling.

The operation converts an input of size [W,H,r²C,N] to [rW,rH,C,N]
Used extensively in super-resolution networks to upsample towrads high reolution feature ma

Reference : https://arxiv.org/pdf/1609.05158.pdf
"""
function split_channels(x::AbstractArray,val::Int) # Split chaannels into `val` partitions
    indices = collect(1:size(x)[end-1])
    channels_par = partition(indices,div(size(x)[end-1],val))
    out = []
    for c in channels_par
       push!(out,x[:,:,c,:])
    end
    return out
end

"""
phaseShift cyclically reshapes and permutes the channels
"""
function phaseShift(x,scale,shape_1,shape_2)
    x_ = reshape(x,shape_1...)
    x_ = permutedims(x_,[1,2,4,3,5])
    return reshape(x_,shape_2...)
end

function pixel_shuffle(x,r)
	ndims(x) == 4 || error("PixelShuffle defined only for arrays of dimension 4")
	(size(x)[end-1])%(r*r) == 0 || error("Number of channels($(size(x)[end-1])) must be divisible by $(r*r)")
    scale = r
    W,H,C,N = size(x)
    
    C_out = div(C,scale*scale)
    shape_1 = (W,H,scale,scale,N)
    shape_2 = (W*scale,H*scale,1,N)
    
    C_split = split_channels(x,C_out)
    output = [phaseShift(x_c,scale,shape_1,shape_2) for x_c in C_split]
    return cat(output...,dims=3)
end
