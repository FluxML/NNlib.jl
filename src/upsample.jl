export bilinear_upsample, ∇bilinear_upsample, pixel_shuffle

"""
    bilinear_upsample(x::AbstractArray{<:Number,4}, k::NTuple{2,Int})

Upsamples the first 2 dimensions of the array `x` by the upsample factors stored in `k`,
using bilinear interpolation. 

The size of the output is equal to 
`(k[1]*S1, k[2]*S2, S3, S4)`, where `S1, S2, S3, S4 = size(x)`.

The interpolation grid is identical to the one used by `imresize` from `Images.jl`.

Currently only 2d upsampling is supported.
"""
function bilinear_upsample(x::AbstractArray{<:Number,4}, k::NTuple{2,Int})
    # This function is gpu friendly
    
    imgsize = size(x)
    newsize = get_newsize(imgsize, k)

    # Get linear interpolation lower- and upper index, and weights
    ilow1, ihigh1, wdiff1 = get_inds_and_ws(x, imgsize[1], newsize[1], 1)
    ilow2, ihigh2, wdiff2 = get_inds_and_ws(x, imgsize[2], newsize[2], 2)

    # Adjust the upper interpolation indices of the second dimension
    ihigh2_r = adjoint_of_idx(ilow2)[ihigh2]

    @inbounds y = @view(x[ilow1,ilow2,:,:]) .* (1 .- wdiff1) .+ @view(x[ihigh1,ilow2,:,:]) .* wdiff1
    @inbounds y .= y .* (1 .- wdiff2) .+ y[:,ihigh2_r,:,:] .* wdiff2
    # @inbounds y = y .* (1 .- wdiff2) .+ @view(y[:,ihigh2_r,:,:]) .* wdiff2 # equivalent to line above
    return y
end

function get_inds_and_ws(x::T, n::Int, m::Int, dim::Int) where T <: AbstractArray
    # Creates interpolation grid for resampling. 
    # Creates the same grid as used in Image.jl `imresize`.
    step = n // m
    offset = (n + 1)//2 - step//2 - step * (m//2 - 1)
    xq = clamp.(range(offset, step=step, length=m), 1, n)

    # Creates interpolation lower and upper indices, and broadcastable weights
    ilow = floor.(Int, xq)
    ihigh = ceil.(Int, xq)
    sizew = ntuple(i-> i == dim ? length(xq) : 1, ndims(x))
    wdiff = convert(T, reshape(xq .- ilow, sizew)) # wdiff possibly lives on gpu
    return ilow, ihigh, wdiff
end

"""
    adjoint_of_idx(idx::Vector{<:Integer})

# Arguments
- `idx`: a vector of indices from which you want the adjoint.

# Outputs
-`idx_adjoint`: index that inverses the operation `x[idx]`.

# Explanation
Determines the adjoint of the vector of indices `idx`, based on the following assumptions:
* `idx[1] == 1`
* `all(d in [0,1] for d in diff(idx))`
The adjoint of `idx` can be seen as an inverse operation such that:

```julia
x = [1, 2, 3, 4, 5]
idx = [1, 2, 2, 3, 4, 4, 5]
idx_adjoint = adjoint_of_idx(idx)
@assert x[idx][idx_adjoint] == x
```
The above holds as long as `idx` contains every index in `x`.
"""
function adjoint_of_idx(idx::Vector{Int})
    d = trues(length(idx))
    d[2:end] .= diff(idx)
    idx_adjoint = findall(d)
    return idx_adjoint
end

function get_newsize(sz, k)
    return ntuple(i -> i <= length(k) ? sz[i]*k[i] : sz[i], length(sz))
end


"""
    ∇bilinear_upsample(Δ::AbstractArray{<:Number,4}, k::NTuple{2,Int})
    
# Arguments
- `Δ`: array that has been upsampled using the upsample factors in `k`

# Outputs
- `dx`: downsampled version of `Δ`

# Explanation

Custom adjoint for [`bilinear_upsample`](@ref). 
The adjoint of upsampling is a downsampling operation, which
in this implementation is performed using `NNlib.conv` in combination with a downsampling kernel based on the
upsampling factors. Because of the zero-padding during convolution, the values at the boundary are polluted by edge-effects,
which have been corrected for manually.
"""
function ∇bilinear_upsample(Δ::AbstractArray{<:Number, 4}, k::NTuple{2,Int})
    # This function is gpu friendly
    
    # Be more efficient on some corner cases
    if size(Δ, 1) == k[1]
        Δ = sum(Δ, dims=1)
        k = (1, k[2])
    end
    if size(Δ, 2) == k[2]
        Δ = sum(Δ, dims=2)
        k = (k[1], 1)
    end
    if (size(Δ, 1) == 1) && (size(Δ, 2) == 1)
        dx = Δ
        return dx
    end

    n_chan, n_batch = size(Δ, 3), size(Δ, 4)

    kern1 = get_downsamplekernel(Δ, k[1])
    kern2 = get_downsamplekernel(Δ, k[2])
    kern = kern1 .* kern2'
    
    pad = (floor(Int, k[1]//2), floor(Int, k[2]//2))
    stride = k
    weight = similar(Δ, eltype(Δ), (size(kern)..., n_chan, n_chan))
    weight .= 0
    
    for i in 1:n_chan
        weight[:,:,i,i] .= kern
    end
    
    dx = conv(Δ, weight, pad=pad, stride=stride)

    # Still have to fix edge effects due to zero-padding of convolution,
    # TODO: Could be circumvented by having padding that just extrapolates the value at the first/last index
    # nextras = tuple((Int.(floor(factor//2)) for factor in k)...)
    nextras = (floor(Int, k[1]//2), floor(Int, k[2]//2))

    # First dimension edge-effect correction
    if nextras[1] > 0
        kern1 = kern[1:nextras[1],:]
        pad1 = (0, pad[2])
        stride1 = (1, stride[2])
        weight1 = similar(Δ, eltype(Δ), (size(kern1)..., n_chan, n_chan))
        weight1 .= 0
        for i in 1:n_chan
            weight1[:,:,i,i] .= kern1
        end
    
        dx[[1],:,:,:] .+= conv(Δ[1:nextras[1],:,:,:], weight1, pad=pad1, stride=stride1)
        weight1 .= weight1[end:-1:1,:,:,:]
        dx[[end],:,:,:] .+= conv(Δ[end-nextras[1]+1:end,:,:,:], weight1, pad=pad1, stride=stride1)
    
        ## Conv with views is not dispatched to CUDA.conv
        # dx[[1],:,:,:] .+= conv(@view(Δ[1:nextras[1],:,:,:]), weight1, pad=pad1, stride=stride1)
        # weight1 .= @view(weight1[end:-1:1,:,:,:])
        # dx[[end],:,:,:] .+= conv(@view(Δ[end-nextras[1]+1:end,:,:,:]), weight1, pad=pad1, stride=stride1)
    end

    # Second dimension edge-effect correction
    if nextras[2] > 0
        kern2 = kern[:,1:nextras[2]]
        pad2 = (pad[1], 0)
        stride2 = (stride[1], 1)
        weight2 = similar(Δ, eltype(Δ), (size(kern2)..., n_chan, n_chan))
        weight2 .= 0
        for i in 1:n_chan
            weight2[:,:,i,i] .= kern2
        end

        yy = conv(Δ[:,1:nextras[2],:,:], weight2, pad=pad2, stride=stride2)
        dx[:,[1],:,:] .+= conv(Δ[:,1:nextras[2],:,:], weight2, pad=pad2, stride=stride2)
        weight2 .= weight2[:,end:-1:1,:,:]
        dx[:,[end],:,:] .+= conv(Δ[:,end-nextras[2]+1:end,:,:], weight2, pad=pad2, stride=stride2)

        ## Conv with views is not dispatched to CUDA.conv
        # yy = conv(@view(Δ[:,1:nextras[2],:,:]), weight2, pad=pad2, stride=stride2)
        # dx[:,[1],:,:] .+= conv(@view(Δ[:,1:nextras[2],:,:]), weight2, pad=pad2, stride=stride2)
        # weight2 .= @view(weight2[:,end:-1:1,:,:])
        # dx[:,[end],:,:] .+= conv(@view(Δ[:,end-nextras[2]+1:end,:,:]), weight2, pad=pad2, stride=stride2)
    end

    ## Finally fix four corners if needed
    n1, n2 = nextras
    if (n1 > 0) & (n2 > 0)
        dx[1,1,:,:] .+= sum(kern[1:n1,1:n2] .* @view(Δ[1:n1,1:n2,:,:]), dims=(1,2))[1,1,:,:]
        dx[1,end,:,:] .+= sum(kern[1:n1,end-n2+1:end] .* @view(Δ[1:n1,end-n2+1:end,:,:]), dims=(1,2))[1,1,:,:]
        dx[end,end,:,:] .+= sum(kern[end-n1+1:end,end-n2+1:end] .* @view(Δ[end-n1+1:end,end-n2+1:end,:,:]), dims=(1,2))[1,1,:,:]
        dx[end,1,:,:] .+= sum(kern[end-n1+1:end,1:n2] .* @view(Δ[end-n1+1:end,1:n2,:,:]), dims=(1,2))[1,1,:,:]
    end

    return dx
end

# `n` upsample factor for which a downsample kernel will be determined.
# Δ is given in case of necessity of gpu conversion 
function get_downsamplekernel(Δ, n::Int)
    step = 1//n
    if n % 2 == 0
        start = step//2
        upward = collect(start:step:1//1)
        kernel = [upward; reverse(upward)]
    else
        start = step
        upward = collect(start:step:1//1)
        kernel = [upward; reverse(upward[1:end-1])]
    end
    # TODO there must be a more convenient way to send to gpu 
    kernel = convert(typeof(Δ), reshape(kernel, length(kernel), 1, 1, 1))
    kernel = dropdims(kernel, dims=(2,3,4))
    return kernel
end

function ChainRulesCore.rrule(::typeof(bilinear_upsample), x, k)
    Ω = bilinear_upsample(x, k)
    function bilinear_upsample_pullback(Δ)
        (NO_FIELDS, ∇bilinear_upsample(Δ, k), DoesNotExist())
    end
    return Ω, bilinear_upsample_pullback
end


"""
    pixel_shuffle(x, r)
    
Pixel shuffling operation. `r` is the upscale factor for shuffling.
The operation converts an input of size [W,H,r²C,N] to size [rW,rH,C,N]
Used extensively in super-resolution networks to upsample 
towards high resolution features.

Reference : https://arxiv.org/pdf/1609.05158.pdf
"""
function pixel_shuffle(x::AbstractArray, r::Integer)
    @assert ndims(x) > 2
    d = ndims(x) - 2
    sizein = size(x)[1:d]
    cin, n = size(x, d+1), size(x, d+2) 
    @assert cin % r^d == 0
    cout = cin ÷ r^d
    # x = reshape(x, sizein..., fill(r, d)..., cout, n) # bug https://github.com/FluxML/Zygote.jl/issues/866
    x = reshape(x, sizein..., ntuple(i->r, d)..., cout, n)
    perm = [d+1:2d 1:d]' |> vec  # = [d+1, 1, d+2, 2, ..., 2d, d]
    x = permutedims(x, (perm..., 2d+1, 2d+2))
    return reshape(x, ((r .* sizein)..., cout, n))
end
