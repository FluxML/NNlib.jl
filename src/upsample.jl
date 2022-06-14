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
4×9 Matrix{$Int}:
 1  1  1  2  2  2  3  3  3
 1  1  1  2  2  2  3  3  3
 4  4  4  5  5  5  6  6  6
 4  4  4  5  5  5  6  6  6

julia> ans == upsample_nearest([1 2 3; 4 5 6]; size=(4, 9))  # equivalent
true

julia> upsample_nearest([1 2 3; 4 5 6], (2,))
4×3 Matrix{$Int}:
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

function rrule(::typeof(upsample_nearest), x::AbstractArray, s::Tuple)
    Ω = upsample_nearest(x, s)
    upsample_nearest_pullback(Δ) = (NoTangent(), ∇upsample_nearest(unthunk(Δ), s), NoTangent())
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

###########
# linear
###########
"""
    upsample_linear(x::AbstractArray{T,3}, scale::Real)
    upsample_linear(x::AbstractArray{T,3}; size::Integer)

Upsamples the first dimension of the array `x` by the upsample provided `scale`,
using linear interpolation. As an alternative to using `scale`, the resulting array `size`
can be directly specified with a keyword argument.

The size of the output is equal to
`(scale*S1, S2, S3)`, where `S1, S2, S3 = size(x)`.
"""
# the user facing function
function upsample_linear(x::AbstractArray{<:Any,N}, scale::NTuple{M,Real}) where {N,M}
    M == N-2 || error("The scale argument should be an NTuple with length $(N-2), but it has length $M.")
    outsize = ntuple(i -> floor(Int, scale[i] * Base.size(x, i)), N-2)
    return upsample_linear(x; size=outsize)
end

# convenience for single-number scale
upsample_linear(x::AbstractArray{<:Any,N}, scale::Real) where N = upsample_linear(x, ntuple(_ -> scale, N-2))

# this actually calls the upsamling kernel
function upsample_linear(x::AbstractArray{T,N}; size::Union{Integer, NTuple{<:Any,Integer}}) where {T,N}
    length(size) == N-2 || error("The scale argument should be an NTuple with length $(N-2), but it has length $(length(size)).")

    if Base.size(x)[1:N-2] == size
        return x
    end

    y = similar(x, T, size..., Base.size(x)[end-1:end]...)
    return upsample_linear_kernel!(y, x)
end

# Convenience definition for integers. The algo internally works with floats and then rounds.
function upsample_linear(x::AbstractArray{T,<:Any}; size) where T<:Integer
    y = float.(x)
    res = upsample_linear(y; size=size)
    return round.(T, res)
end

# compatibility layer for old versions of NNlibCUDA
# old versions overload upsample_linear_wcn, new versions overload upsample_linear_kernel
# can be removed from NNlib 0.9, i.e. revert https://github.com/FluxML/NNlib.jl/pull/414
# IF https://github.com/FluxML/NNlibCUDA.jl/pull/49 has been merged
upsample_linear_kernel!(y::AbstractArray{<:Any,3}, x::AbstractArray{<:Any,3}) = upsample_linear_wcn!(y,x)
upsample_linear_kernel!(y::AbstractArray{<:Any,4}, x::AbstractArray{<:Any,4}) = upsample_bilinear_whcn!(y,x)
upsample_linear_kernel!(y::AbstractArray{<:Any,5}, x::AbstractArray{<:Any,5}) = upsample_trilinear_whdcn!(y,x)
∇upsample_linear_kernel!(y::AbstractArray{<:Any,3}, x::AbstractArray{<:Any,3}) = ∇upsample_linear_wcn!(y,x)
∇upsample_linear_kernel!(y::AbstractArray{<:Any,4}, x::AbstractArray{<:Any,4}) = ∇upsample_bilinear_whcn!(y,x)
∇upsample_linear_kernel!(y::AbstractArray{<:Any,5}, x::AbstractArray{<:Any,5}) = ∇upsample_trilinear_whdcn!(y,x)

# linearly upsamples first dim of 3D array
function upsample_linear_wcn!(output::AbstractArray{T,3}, input::AbstractArray{T,3}) where T
    size(input)[2:3] == size(output)[2:3] || error("Number of input and output channels and batches must match. Got input $(size(input)) and output $(size(output))")
    in_w, channels, batches = size(input)
    RT = real(T)
    # treat batch and channel dimension as one for better parallelization granularity
    channels *= batches
    out_w, _, _ = size(output)
    output_slice_size = out_w

    #real(T)() and // so that we can handle rationals (super slow)
    width_scale  = RT((in_w - 1) // (out_w - 1))

    @inline idx(c, w) = c * in_w + w + 1

    Threads.@threads for c in 0:channels-1
        @inbounds for ow in 0:out_w-1
            iw0, iw1, w0lambda, w1lambda = compute_source_index_and_lambda(width_scale, ow, in_w, out_w)
            output_offset = c * output_slice_size + ow + 1
            output[output_offset] = (w0lambda * input[idx(c, iw0)] + # w0 * i00
                                    w1lambda * input[idx(c, iw1)])  # w1 * i01
        end
    end
    return output
end

# bilinear
# linearly upsamples first two dims of 4D array
function upsample_bilinear_whcn!(output::AbstractArray{T,4}, input::AbstractArray{T,4}) where T
    size(input)[3:4] == size(output)[3:4] || error("Number of input and output channels and batches must match. Got input $(size(input)) and output $(size(output))")
    in_w, in_h, channels, batches = size(input)
    RT = real(T)
    # treat batch and channel dimension as one for better parallelization granularity
    channels *= batches
    out_w, out_h, _, _ = size(output)
    output_slice_size = out_h * out_w

    #real(T)() and // so that we can handle rationals (super slow)
    width_scale  = RT((in_w - 1) // (out_w - 1))
    height_scale = RT((in_h - 1) // (out_h - 1))

    @inline idx(c, h, w) = c * in_h * in_w + h * in_w + w + 1

    Threads.@threads for c in 0:channels-1
        @inbounds for oh in 0:out_h-1
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

# trilinear
# linearly upsamples first three dims of 5D array
function upsample_trilinear_whdcn!(output::AbstractArray{T,5}, input::AbstractArray{T,5}) where T
    size(input)[4:5] == size(output)[4:5] || error("Number of input and output channels and batches must match. Got input $(size(input)) and output $(size(output))")
    in_w, in_h, in_d, channels, batches = size(input)
    RT = real(T)
    # treat batch and channel dimension as one for better parallelization granularity
    channels *= batches
    out_w, out_h, out_d, _, _ = size(output)
    output_slice_size = out_h * out_w * out_d

    #real(T)() and // so that we can handle rationals (super slow)
    width_scale  = RT((in_w - 1) // (out_w - 1))
    height_scale = RT((in_h - 1) // (out_h - 1))
    depth_scale  = RT((in_d - 1) // (out_d - 1))

    @inline idx(c, d, h, w) = c * in_d * in_h * in_w + d * in_h * in_w + h * in_w + w + 1

    Threads.@threads for c in 0:channels-1
        @inbounds for od in 0:out_d-1
            id0, id1, d0lambda, d1lambda = compute_source_index_and_lambda(depth_scale, od, in_d, out_d)
            for oh in 0:out_h-1
                ih0, ih1, h0lambda, h1lambda = compute_source_index_and_lambda(height_scale, oh, in_h, out_h)
                for ow in 0:out_w-1
                    iw0, iw1, w0lambda, w1lambda = compute_source_index_and_lambda(width_scale, ow, in_w, out_w)
                    output_offset = c * output_slice_size + od * out_w * out_h + oh * out_w + ow + 1
                    output[output_offset] =
                        d0lambda * h0lambda * w0lambda * input[idx(c, id0, ih0, iw0)] + # d0 * h0 * w0 * i000
                        d0lambda * h0lambda * w1lambda * input[idx(c, id0, ih0, iw1)] + # d0 * h0 * w1 * i001
                        d0lambda * h1lambda * w0lambda * input[idx(c, id0, ih1, iw0)] + # d0 * h1 * w0 * i010
                        d0lambda * h1lambda * w1lambda * input[idx(c, id0, ih1, iw1)] + # d0 * h1 * w1 * i011
                        d1lambda * h0lambda * w0lambda * input[idx(c, id1, ih0, iw0)] + # d1 * h0 * w0 * i100
                        d1lambda * h0lambda * w1lambda * input[idx(c, id1, ih0, iw1)] + # d1 * h0 * w1 * i101
                        d1lambda * h1lambda * w0lambda * input[idx(c, id1, ih1, iw0)] + # d1 * h1 * w0 * i110
                        d1lambda * h1lambda * w1lambda * input[idx(c, id1, ih1, iw1)]   # d1 * h1 * w1 * i111
                end
            end
        end
    end
    return output
end



"""
    ∇upsample_linear(Δ::AbstractArray{T,3}; size::Integer) where T

# Arguments
- `Δ`: Incoming gradient array, backpropagated from downstream layers
- `size`: Size of the image upsampled in the first place

# Outputs
- `dx`: Downsampled version of `Δ`
"""
function ∇upsample_linear(Δ::AbstractArray{T,N}; size::NTuple{<:Any,Integer}) where {T,N}
    if Base.size(Δ)[1:N-2] == size
        return Δ
    end
    dx = zero(similar(Δ, T, size..., Base.size(Δ)[end-1:end]...))
    return ∇upsample_linear_kernel!(dx, Δ)
end

# linear
function ∇upsample_linear_wcn!(dx::AbstractArray{T,3}, Δ::AbstractArray{T,3}) where T
    size(dx)[2:3] == size(Δ)[2:3] || error("Number of input and output channels and batches must match. Got input $(size(input)) and output $(size(output))")
    in_w, channels, batches = size(dx)
    RT = real(T)
    # treat batch and channel dimension as one for better parallelization granularity
    channels *= batches
    out_w, _, _ = size(Δ)
    output_slice_size = out_w

    width_scale = RT((in_w - 1) // (out_w - 1))

    @inline idx(c, w) = c * in_w + w + 1

    Threads.@threads for c in 0:channels-1
        @inbounds for ow in 0:out_w-1
            iw0, iw1, w0lambda, w1lambda = compute_source_index_and_lambda(width_scale, ow, in_w, out_w)
            output_offset = c * output_slice_size + ow + 1
            Δ_value = Δ[output_offset]
            dx[idx(c, iw0)] += w0lambda * Δ_value # i00
            dx[idx(c, iw1)] += w1lambda * Δ_value # i01
        end
    end
    return dx
end

# bilinear
function ∇upsample_bilinear_whcn!(dx::AbstractArray{T,4}, Δ::AbstractArray{T,4}) where T
    size(dx)[3:4] == size(Δ)[3:4] || error("Number of input and output channels and batches must match. Got input $(size(input)) and output $(size(output))")
    in_w, in_h, channels, batches = size(dx)
    RT = real(T)
    # treat batch and channel dimension as one for better parallelization granularity
    channels *= batches
    out_w, out_h, _, _ = size(Δ)
    output_slice_size = out_h * out_w

    width_scale  = RT((in_w - 1) // (out_w - 1))
    height_scale = RT((in_h - 1) // (out_h - 1))

    @inline idx(c, h, w) = c * in_h * in_w + h * in_w + w + 1

    Threads.@threads for c in 0:channels-1
        @inbounds for oh in 0:out_h-1
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

# trilinear
function ∇upsample_trilinear_whdcn!(dx::AbstractArray{T,5}, Δ::AbstractArray{T,5}) where T
    size(dx)[4:5] == size(Δ)[4:5] || error("Number of input and output channels and batches must match. Got dx $(size(dx)) and Δ $(size(Δ))")
    in_w, in_h, in_d, channels, batches = size(dx)
    RT = real(T)
    # treat batch and channel dimension as one for better parallelization granularity
    channels *= batches
    out_w, out_h, out_d, _, _ = size(Δ)
    output_slice_size = out_h * out_w * out_d

    #real(T)() and // so that we can handle rationals (super slow)
    width_scale  = RT((in_w - 1) // (out_w - 1))
    height_scale = RT((in_h - 1) // (out_h - 1))
    depth_scale  = RT((in_d - 1) // (out_d - 1))

    @inline idx(c, d, h, w) = c * in_d * in_h * in_w + d * in_h * in_w + h * in_w + w + 1

    Threads.@threads for c in 0:channels-1
        @inbounds for od in 0:out_d-1
            id0, id1, d0lambda, d1lambda = compute_source_index_and_lambda(depth_scale, od, in_d, out_d)
            for oh in 0:out_h-1
                ih0, ih1, h0lambda, h1lambda = compute_source_index_and_lambda(height_scale, oh, in_h, out_h)
                for ow in 0:out_w-1
                    iw0, iw1, w0lambda, w1lambda = compute_source_index_and_lambda(width_scale, ow, in_w, out_w)
                    output_offset = c * output_slice_size + od * out_w * out_h + oh * out_w + ow + 1
                    Δ_value = Δ[output_offset]
                    dx[idx(c, id0, ih0, iw0)] += d0lambda * h0lambda * w0lambda * Δ_value  # /* i000 */
                    dx[idx(c, id0, ih0, iw1)] += d0lambda * h0lambda * w1lambda * Δ_value  # /* i001 */
                    dx[idx(c, id0, ih1, iw0)] += d0lambda * h1lambda * w0lambda * Δ_value  # /* i010 */
                    dx[idx(c, id0, ih1, iw1)] += d0lambda * h1lambda * w1lambda * Δ_value  # /* i011 */
                    dx[idx(c, id1, ih0, iw0)] += d1lambda * h0lambda * w0lambda * Δ_value  # /* i100 */
                    dx[idx(c, id1, ih0, iw1)] += d1lambda * h0lambda * w1lambda * Δ_value  # /* i101 */
                    dx[idx(c, id1, ih1, iw0)] += d1lambda * h1lambda * w0lambda * Δ_value  # /* i110 */
                    dx[idx(c, id1, ih1, iw1)] += d1lambda * h1lambda * w1lambda * Δ_value  # /* i111 */
                end
            end
        end
    end
    return dx
end

function rrule(::typeof(upsample_linear), x::AbstractArray{<:Any,N}; size) where N
    Ω = upsample_linear(x; size=size)
    function upsample_linear_pullback(Δ)
        (NoTangent(), ∇upsample_linear(unthunk(Δ); size=Base.size(x)[1:N-2]))
    end
    return Ω, upsample_linear_pullback
end

###########
# bilinear
###########
"""
    upsample_bilinear(x::AbstractArray{T,4}, scale::NTuple{2,Real})
    upsample_bilinear(x::AbstractArray{T,4}; size::NTuple{2,Integer})

Upsamples the first 2 dimensions of the array `x` by the upsample factors stored in `scale`,
using bilinear interpolation. As an alternative to using `scale`, the resulting image `size`
can be directly specified with a keyword argument.

The size of the output is equal to
`(scale[1]*S1, scale[2]*S2, S3, S4)`, where `S1, S2, S3, S4 = size(x)`.

# Examples

```jldoctest
julia> x = reshape(Float32[1 2 3; 4 5 6], (2,3,1,1))
2×3×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 1.0  2.0  3.0
 4.0  5.0  6.0

julia> upsample_bilinear(x, (2, 3))
4×9×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 1.0  1.25  1.5  1.75  2.0  2.25  2.5  2.75  3.0
 2.0  2.25  2.5  2.75  3.0  3.25  3.5  3.75  4.0
 3.0  3.25  3.5  3.75  4.0  4.25  4.5  4.75  5.0
 4.0  4.25  4.5  4.75  5.0  5.25  5.5  5.75  6.0

julia> ans == upsample_bilinear(x; size=(4, 9))  # specify ouput size instead
true

julia> upsample_bilinear(x, (2.5, 3.5))  # non-integer scaling factors are allowed
5×10×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 1.0   1.22222  1.44444  1.66667  1.88889  …  2.33333  2.55556  2.77778  3.0
 1.75  1.97222  2.19444  2.41667  2.63889     3.08333  3.30556  3.52778  3.75
 2.5   2.72222  2.94444  3.16667  3.38889     3.83333  4.05556  4.27778  4.5
 3.25  3.47222  3.69444  3.91667  4.13889     4.58333  4.80556  5.02778  5.25
 4.0   4.22222  4.44444  4.66667  4.88889     5.33333  5.55556  5.77778  6.0
```
"""
upsample_bilinear(x, scale) = upsample_linear(x, scale)
upsample_bilinear(x; size)  = upsample_linear(x; size)


"""
    ∇upsample_bilinear(Δ::AbstractArray{T,4}; size::NTuple{2,Integer}) where T

# Arguments
- `Δ`: Incoming gradient array, backpropagated from downstream layers
- `size`: Lateral (W,H) size of the image upsampled in the first place

# Outputs
- `dx`: Downsampled version of `Δ`
"""
∇upsample_bilinear(Δ; size) = ∇upsample_linear(Δ; size)

"""
    upsample_trilinear(x::AbstractArray{T,5}, scale::NTuple{3,Real})
    upsample_trilinear(x::AbstractArray{T,5}; size::NTuple{3,Integer})

Upsamples the first 3 dimensions of the array `x` by the upsample factors stored in `scale`,
using trilinear interpolation. As an alternative to using `scale`, the resulting image `size`
can be directly specified with a keyword argument.

The size of the output is equal to
`(scale[1]*S1, scale[2]*S2, scale[3]*S3, S4, S5)`, where `S1, S2, S3, S4, S5 = size(x)`.

# Examples

```julia
upsample_trilinear(x, (2, 3, 4))
upsample_trilinear(x; size=(4, 9, 11))  # specify ouput size instead
upsample_trilinear(x, (2.5, 3.5, pi))  # non-integer scaling factors are allowed
```
"""
upsample_trilinear(x, scale) = upsample_linear(x, scale)
upsample_trilinear(x; size)  = upsample_linear(x; size)

"""
    ∇upsample_trilinear(Δ::AbstractArray{T,5}; size::NTuple{3,Integer}) where T

# Arguments
- `Δ`: Incoming gradient array, backpropagated from downstream layers
- `size`: Lateral size & depth (W,H,D) of the image upsampled in the first place

# Outputs
- `dx`: Downsampled version of `Δ`
"""
∇upsample_trilinear(Δ; size)= ∇upsample_linear(Δ; size)



"""
    pixel_shuffle(x, r::Integer)

Pixel shuffling operation, upscaling by a factor `r`.

For 4-arrays representing `N` images, the operation converts input `size(x) == (W, H, r^2*C, N)`
to output of size `(r*W, r*H, C, N)`. For `D`-dimensional data, it expects `ndims(x) == D+2`
with channel and batch dimensions, and divides the number of channels by `r^D`.

Used in super-resolution networks to upsample towards high resolution features.
Reference: Shi et. al., "Real-Time Single Image and Video Super-Resolution ...", CVPR 2016, https://arxiv.org/abs/1609.05158

# Examples

```jldoctest
julia> x = [10i + j + channel/10 for i in 1:2, j in 1:3, channel in 1:4, batch in 1:1]
2×3×4×1 Array{Float64, 4}:
[:, :, 1, 1] =
 11.1  12.1  13.1
 21.1  22.1  23.1

[:, :, 2, 1] =
 11.2  12.2  13.2
 21.2  22.2  23.2

[:, :, 3, 1] =
 11.3  12.3  13.3
 21.3  22.3  23.3

[:, :, 4, 1] =
 11.4  12.4  13.4
 21.4  22.4  23.4

julia> pixel_shuffle(x, 2)  # 4 channels used up as 2x upscaling of image dimensions
4×6×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 11.1  11.3  12.1  12.3  13.1  13.3
 11.2  11.4  12.2  12.4  13.2  13.4
 21.1  21.3  22.1  22.3  23.1  23.3
 21.2  21.4  22.2  22.4  23.2  23.4

julia> y = [i + channel/10 for i in 1:3, channel in 1:6, batch in 1:1]
3×6×1 Array{Float64, 3}:
[:, :, 1] =
 1.1  1.2  1.3  1.4  1.5  1.6
 2.1  2.2  2.3  2.4  2.5  2.6
 3.1  3.2  3.3  3.4  3.5  3.6

julia> pixel_shuffle(y, 2)  # 1D image, with 6 channels reduced to 3
6×3×1 Array{Float64, 3}:
[:, :, 1] =
 1.1  1.3  1.5
 1.2  1.4  1.6
 2.1  2.3  2.5
 2.2  2.4  2.6
 3.1  3.3  3.5
 3.2  3.4  3.6
```
"""
function pixel_shuffle(x::AbstractArray, r::Integer)
    ndims(x) > 2 || throw(ArgumentError("expected x with at least 3 dimensions"))
    d = ndims(x) - 2
    sizein = size(x)[1:d]
    cin, n = size(x, d+1), size(x, d+2)
    cin % r^d == 0 || throw(ArgumentError("expected channel dimension to be divisible by r^d = $(
        r^d), where d=$d is the number of spatial dimensions. Given r=$r, input size(x) = $(size(x))"))
    cout = cin ÷ r^d
    x = reshape(x, sizein..., ntuple(i->r, d)..., cout, n)
    perm = hcat(d+1:2d, 1:d) |> transpose |> vec  # = [d+1, 1, d+2, 2, ..., 2d, d]
    x = permutedims(x, (perm..., 2d+1, 2d+2))
    return reshape(x, map(s -> s*r, sizein)..., cout, n)
end
