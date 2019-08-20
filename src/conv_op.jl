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
       c = [c_ for c_ in c]
       push!(out,x[:,:,c,:])
    end
    return out
end

"""
phaseShift cyclically reshapes and permutes the channels
"""
function phase_shift(x,r)
    W,H,C,N = size(x)
    x = reshape(x,W,H,r,r,N)
    x = [x[i,:,:,:,:] for i in 1:W]
    x = cat([t for t in x]...,dims=2)
    x = [x[i,:,:,:] for i in 1:size(x)[1]]
    x = cat([t for t in x]...,dims=2)
    x
end

function pixel_shuffle(x,r=3)
	ndims(x) == 4 || error("PixelShuffle defined only for arrays of dimension 4")
	(size(x)[end-1])%(r*r) == 0 || error("Number of channels($(size(x)[end-1])) must be divisible by $(r*r)")
    
    C_out = div(size(x)[end-1],r*r)
    sch = split_channels(x,C_out)
    out = cat([phase_shift(c,r) for c in split_channels(x,C_out)]...,dims=3)
    reshape(out,size(out)[1],size(out)[2],C_out,div(size(out)[end],C_out))
end
