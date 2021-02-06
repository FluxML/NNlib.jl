export upsample_nearest, ∇upsample_nearest,
    upsample_bilinear, ∇upsample_bilinear,
    pixel_shuffle

"""
    upsample_nearest(x, scale::NTuple{S,Int})
    upsample_nearest(x; size::NTuple{S,Int})

Upsamples the array `x` by integer multiples along the first `S` dimensions.
Subsequent dimensions of `x` are not altered.

Either the `scale` factors or the final output `size` can be specified. 

See also [`upsample_bilinear`](@ref), for two dimensions of an `N=4` array.

# Example
```jldoctest
julia> upsample_nearest([1 2 3; 4 5 6], (2, 3))
4×9 Array{$Int,2}:
 1  1  1  2  2  2  3  3  3
 1  1  1  2  2  2  3  3  3
 4  4  4  5  5  5  6  6  6
 4  4  4  5  5  5  6  6  6

julia> ans == upsample_nearest([1 2 3; 4 5 6]; size=(4, 9))  # equivalent
true

julia> upsample_nearest([1 2 3; 4 5 6], (2,))
4×3 Array{$Int,1}:
 1  2  3
 1  2  3
 4  5  6
 4  5  6

julia> ans == upsample_nearest([1 2 3; 4 5 6], size=(4,))
true
```
"""
function upsample_nearest(x::AbstractArray; size::NTuple{S,Int}) where S
    xsize = Base.size(x)[1:S]
    all(size .% xsize .== 0) || throw(ArgumentError("expected output size divisible by input size"))
    return upsample_nearest(x, size .÷ xsize)
end

function upsample_nearest(x::AbstractArray{T,N}, scales::NTuple{S, <:Integer}) where {T,N,S}
    S in 1:N || throw(ArgumentError("can't upsample ndims(x)=$N with scale=$scales"))
    outsize = ntuple(d -> d<=S ? scales[d] * size(x,d) : size(x,d), N)
    out = similar(x, T, outsize)
    writesize = ntuple(N+S) do d
        d > 2S && return size(x, d-S)
        isodd(d) ? scales[cld(d,2)] : size(x, cld(d,2))
    end
    readsize = ntuple(N+S) do d
        d > 2S && return size(x, d-S)
        isodd(d) ? 1 : size(x, cld(d,2))
    end
    reshape(out, writesize) .= reshape(x, readsize)
    out
end

function ∇upsample_nearest(x::AbstractArray{T,N}, scales::NTuple{S, <:Integer}) where {T,N,S}
    outsize = ntuple(N) do d
        d > S && return size(x,d)
        rem(size(x,d), scales[d]) == 0 || throw(ArgumentError("expected input array evenly divisible by scale=$scales, got size(x)=$(size(x))"))
        div(size(x,d), scales[d])
    end
    tempsize = ntuple(N+S) do d
        d > 2S && return size(x, d-S)
        s = scales[cld(d,2)]
        isodd(d) ? s : div(size(x, cld(d,2)),s)
    end
    mid = sum(reshape(x, tempsize), dims=ntuple(d -> 2d-1, S))
    reshape(mid, outsize)
end

function ChainRulesCore.rrule(::typeof(upsample_nearest), x::AbstractArray, s::Tuple)
    Ω = upsample_nearest(x, s)
    upsample_nearest_pullback(Δ) = (NO_FIELDS, ∇upsample_nearest(Δ, s), DoesNotExist())
    return Ω, upsample_nearest_pullback
end

# utility function
@inline function compute_source_index_and_lambda(
    ratio, # 0 < ratio < 1
    output_index,
    input_size,
    output_size
)
    real_input_index = ratio*output_index
    input_index0 = floor(Int, real_input_index) # typecast to int was here in C++
    offset = (input_index0 < input_size - 1) ? 1 : 0
    input_index1 = input_index0 + offset
    lambda1 = real_input_index - input_index0
    lambda0 = 1 - lambda1
    return input_index0, input_index1, lambda0, lambda1
end

"""
    upsample_bilinear(x::AbstractArray{T,4}, scale::NTuple{2,Real})
    upsample_bilinear(x::AbstractArray{T,4}; size::NTuple{2,Integer})

Upsamples the first 2 dimensions of the array `x` by the upsample factors stored in `scale`,
using bilinear interpolation. As an alternative to using `scale`, the resulting image `size`
can be directly specified with a keyword argument.

The size of the output is equal to
`(scale[1]*S1, scale[2]*S2, S3, S4)`, where `S1, S2, S3, S4 = size(x)`.

Examples:
```julia
upsample_bilinear(x, (2, pi)) # real scaling factors are allowed
upsample_bilinear(x; size=(64,64)) # specify ouput size
```
"""
function upsample_bilinear(x::AbstractArray{<:Any,4}, scale::NTuple{2,Real})
    outsize = ntuple(i -> floor(Int, scale[i] * Base.size(x, i)), 2)
    return upsample_bilinear(x; size=outsize)
end

upsample_bilinear(x, scale::Real) = upsample_bilinear(x, (scale,scale))

function upsample_bilinear(x::AbstractArray{T,4}; size::NTuple{2,Integer}) where T
    w,h,c,n = Base.size(x)
    if (w,h) == size
        return x
    end
    y = similar(x, T, size..., c, n)
    return upsample_bilinear_whcn!(y, x)
end

function upsample_bilinear(x::AbstractArray{T,4}; size::NTuple{2,Integer}) where T<:Integer
    y = float.(x)
    res = upsample_bilinear(y; size=size)
    return round.(T, res)
end

# this is the core function which works on arrays of arbitrary size
# the implementation is a translation of https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/UpSampleMoreKernel.cpp
# which implements open-cv style linear interpolation / upsampling
# for simplicity, corners are aligned and all logic for other behaviour has been stripped
# - whcn because there is also a cwhn implementation
# - the function is parallelized using @threads
# - RGB types could be supported via reinterpreting
# - integer types need to be converted to Float and back
# - rationals work, but are slow
function upsample_bilinear_whcn!(output::AbstractArray{T,4}, input::AbstractArray{T,4}) where T
    size(input)[3:4] == size(output)[3:4] || error("Number of input and output channels and batches must match. Got input $(size(input)) and output $(size(output))")
    in_w, in_h, channels, batches = size(input)
    # treat batch and channel dimension as one for better parallelization granularity
    channels *= batches
    out_w, out_h, _, _ = size(output)
    output_slice_size = out_h * out_w

    # T() and // so that we can handle rationals (super slow)
    width_scale  = T((in_w - 1) // (out_w - 1))
    height_scale = T((in_h - 1) // (out_h - 1))

    @inline idx(c, h, w) = c * in_h * in_w + h * in_w + w + 1

    @inbounds Threads.@threads for c in 0:channels-1
        for oh in 0:out_h-1
            ih0, ih1, h0lambda, h1lambda = compute_source_index_and_lambda(height_scale, oh, in_h, out_h)
            for ow in 0:out_w-1
                iw0, iw1, w0lambda, w1lambda = compute_source_index_and_lambda(width_scale, ow, in_w, out_w)
                output_offset = c * output_slice_size + oh * out_w + ow + 1
                output[output_offset] =
                    (h0lambda * w0lambda * input[idx(c, ih0, iw0)] + # h0 * w0 * i00
                     h0lambda * w1lambda * input[idx(c, ih0, iw1)] + # h0 * w1 * i01
                     h1lambda * w0lambda * input[idx(c, ih1, iw0)] + # h1 * w0 * i10
                     h1lambda * w1lambda * input[idx(c, ih1, iw1)])  # h1 * w1 * i11
            end
        end
    end
    return output
end

"""
    ∇upsample_bilinear(Δ::AbstractArray{T,4}; size::NTuple{2,Integer}) where T

# Arguments
- `Δ`: Incoming gradient array, backpropagated from downstream layers
- `size`: Lateral (W,H) size of the image upsampled in the first place

# Outputs
- `dx`: Downsampled version of `Δ`
"""
function ∇upsample_bilinear(Δ::AbstractArray{T,4}; size::NTuple{2,Integer}) where T
    w, h, c, n = Base.size(Δ)
    out_w, out_h = size
    if (w,h) == (out_w, out_h)
        return Δ
    end
    dx = zero(similar(Δ, T, out_w, out_h, c, n))
    return ∇upsample_bilinear_whcn!(dx, Δ)
end

function ∇upsample_bilinear_whcn!(dx::AbstractArray{T,4}, Δ::AbstractArray{T,4}) where T
    size(dx)[3:4] == size(Δ)[3:4] || error("Number of input and output channels and batches must match. Got input $(size(input)) and output $(size(output))")
    in_w, in_h, channels, batches = size(dx)

    # treat batch and channel dimension as one for better parallelization granularity
    channels *= batches
    out_w, out_h, _, _ = size(Δ)
    output_slice_size = out_h * out_w

    width_scale  = T((in_w - 1) // (out_w - 1))
    height_scale = T((in_h - 1) // (out_h - 1))

    @inline idx(c, h, w) = c * in_h * in_w + h * in_w + w + 1

    @inbounds Threads.@threads for c in 0:channels-1
        for oh in 0:out_h-1
            ih0, ih1, h0lambda, h1lambda = compute_source_index_and_lambda(height_scale, oh, in_h, out_h)
            for ow in 0:out_w-1
                iw0, iw1, w0lambda, w1lambda = compute_source_index_and_lambda(width_scale, ow, in_w, out_w)
                output_offset = c * output_slice_size + oh * out_w + ow + 1
                Δ_value = Δ[output_offset]
                dx[idx(c, ih0, iw0)] += h0lambda * w0lambda * Δ_value # i00
                dx[idx(c, ih0, iw1)] += h0lambda * w1lambda * Δ_value # i01
                dx[idx(c, ih1, iw0)] += h1lambda * w0lambda * Δ_value # i10
                dx[idx(c, ih1, iw1)] += h1lambda * w1lambda * Δ_value # i11
            end
        end
    end
    return dx
end

function ChainRulesCore.rrule(::typeof(upsample_bilinear), x; size)
    Ω = upsample_bilinear(x; size=size)
    function upsample_bilinear_pullback(Δ)
        (NO_FIELDS, ∇upsample_bilinear(Δ; size=(Base.size(x,1),Base.size(x,2))))
    end
    return Ω, upsample_bilinear_pullback
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
