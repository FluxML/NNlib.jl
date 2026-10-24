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

#
# Upsampling
#
# GPU based bilinear upsampling including its gradient
#
# Based on the Caffe2 implementation at:
# The code is a translation from the following files:
# - https://github.com/pytorch/pytorch/blob/v1.8.0-rc1/caffe2/operators/upsample_op.cu
# - https://github.com/pytorch/pytorch/blob/v1.8.0-rc1/caffe2/core/common_gpu.h
#
# Copyright (c) 2016-2021 Facebook Inc.
# Copyright (c) 2015 Google Inc.
# Copyright (c) 2015 Yangqing Jia
# Copyright 2019-2020 Kakao Brain
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
#    conditions and the following disclaimer in the documentation and/or other materials
#    provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America and
#    IDIAP Research Institute nor the names of its contributors may be used to endorse or
#    promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Forward and backward pass have been tested to produce the same output
# as pytorch with align_corners=True - it works modulo bit noise.
# pytorch's default is align_corners=False, because otherwise the gradients depend on the
# image size, which should be avoided -> this should be considered here as well

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

"""
    ∇upsample_nearest(Δ::AbstractArray{T,3}, scales::NTuple{S, <:Integer}) where T

# Arguments
- `Δ`: Incoming gradient array, backpropagated from downstream layers
- `scales`: scales by which the image was upsampled in the first place

# Outputs
- `dx`: Downsampled version of `Δ`
"""
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

"""
    upsample_linear(x::AbstractArray{T,3}, scale::Real; align_corners::Bool = true)
    upsample_linear(x::AbstractArray{T,3}; size::Integer, align_corners::Bool = true)

Upsamples the first dimension of the array `x` by the upsample provided `scale`,
using linear interpolation. As an alternative to using `scale`, the resulting array `size`
can be directly specified with a keyword argument.

The size of the output is equal to
`(scale*S1, S2, S3)`, where `S1, S2, S3 = size(x)`.
"""  # the user facing function
function upsample_linear(x::AbstractArray{<:Any,N}, scale::NTuple{M,Real}; align_corners::Bool = true) where {N,M}
    M == N-2 || error("The scale argument should be an NTuple with length $(N-2), but it has length $M.")
    outsize = ntuple(i -> floor(Int, scale[i] * Base.size(x, i)), N-2)
    return upsample_linear(x; size=outsize, align_corners)
end

# convenience for single-number scale
upsample_linear(x::AbstractArray{<:Any,N}, scale::Real; align_corners::Bool = true) where N =
    upsample_linear(x, ntuple(_ -> scale, N-2); align_corners)

# this actually calls the upsamling kernel
function upsample_linear(x::AbstractArray{T,N}; size::Union{Integer, NTuple{<:Any,Integer}}, align_corners::Bool = true) where {T,N}
    length(size) == N-2 || error("The scale argument should be an NTuple with length $(N-2), but it has length $(length(size)).")

    if Base.size(x)[1:N-2] == size
        return x
    end

    y = similar(x, T, size..., Base.size(x)[end-1:end]...)
    return upsample_linear_kernel!(y, x; align_corners)
end

# Convenience definition for integers. The algo internally works with floats and then rounds.
function upsample_linear(x::AbstractArray{T,<:Any}; size, align_corners::Bool = true) where T<:Integer
    y = float.(x)
    res = upsample_linear(y; size=size, align_corners)
    return round.(T, res)
end

"""
    ∇upsample_linear(Δ::AbstractArray{T,3}; size::Integer, align_corners::Bool = true) where T

# Arguments
- `Δ`: Incoming gradient array, backpropagated from downstream layers
- `size`: Size of the image upsampled in the first place

# Outputs
- `dx`: Downsampled version of `Δ`
"""
function ∇upsample_linear(Δ::AbstractArray{T,N}; size::Union{Integer, NTuple{<:Any,Integer}}, align_corners::Bool = true) where {T,N}
    size isa Integer && (size = (size,))
    if Base.size(Δ)[1:N-2] == size
        return Δ
    end
    dx = fill!(similar(Δ, T, size..., Base.size(Δ)[end-1:end]...), zero(T))
    return ∇upsample_linear_kernel!(dx, Δ; align_corners)
end


function rrule(::typeof(upsample_linear), x::AbstractArray{<:Any,N}; size, align_corners::Bool = true) where N
    Ω = upsample_linear(x; size, align_corners)
    function upsample_linear_pullback(Δ)
        (NoTangent(), ∇upsample_linear(unthunk(Δ); size=Base.size(x)[1:N-2], align_corners))
    end
    return Ω, upsample_linear_pullback
end

"""
    upsample_bilinear(x::AbstractArray{T,4}, scale::NTuple{2,Real}; align_corners::Bool = true, antialias::Bool = false)
    upsample_bilinear(x::AbstractArray{T,4}; size::NTuple{2,Integer}, align_corners::Bool = true, antialias::Bool = false)

Upsamples the first 2 dimensions of the array `x` by the upsample factors stored in `scale`,
using bilinear interpolation. As an alternative to using `scale`, the resulting image `size`
can be directly specified with a keyword argument.

The size of the output is equal to
`(scale[1]*S1, scale[2]*S2, S3, S4)`, where `S1, S2, S3, S4 = size(x)`.

With `antialias = true`, the output can also be smaller than the input: as in PyTorch's
`F.interpolate(x, size, mode="bilinear", antialias=True)` and PIL, the triangle filter is then
stretched by the downsampling factor and renormalised at the borders. Pass `align_corners = false`
to match them. See also [`upsample_bicubic`](@ref).

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
function upsample_bilinear(x, scale; align_corners::Bool = true, antialias::Bool = false)
    antialias || return upsample_linear(x, scale; align_corners)
    return upsample_bilinear(x; size = _upsample_size(x, scale), align_corners, antialias)
end

function upsample_bilinear(x; size, align_corners::Bool = true, antialias::Bool = false)
    antialias || return upsample_linear(x; size, align_corners)
    return _upsample_separable(x, size, :bilinear, align_corners, antialias)
end


"""
    ∇upsample_bilinear(Δ::AbstractArray{T,4}; size::NTuple{2,Integer}, align_corners::Bool = true, antialias::Bool = false) where T

# Arguments
- `Δ`: Incoming gradient array, backpropagated from downstream layers
- `size`: Lateral (W,H) size of the image upsampled in the first place

# Outputs
- `dx`: Downsampled version of `Δ`
"""
∇upsample_bilinear(Δ; size, align_corners::Bool = true, antialias::Bool = false) =
    antialias ? ∇_upsample_separable(Δ, size, :bilinear, align_corners, antialias) :
                ∇upsample_linear(Δ; size, align_corners)

"""
    upsample_trilinear(x::AbstractArray{T,5}, scale::NTuple{3,Real}; align_corners::Bool = true)
    upsample_trilinear(x::AbstractArray{T,5}; size::NTuple{3,Integer}, align_corners::Bool = true)

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
upsample_trilinear(x, scale; align_corners::Bool = true) = upsample_linear(x, scale; align_corners)
upsample_trilinear(x; size, align_corners::Bool = true)  = upsample_linear(x; size, align_corners)

"""
    ∇upsample_trilinear(Δ::AbstractArray{T,5}; size::NTuple{3,Integer}, align_corners::Bool = true) where T

# Arguments
- `Δ`: Incoming gradient array, backpropagated from downstream layers
- `size`: Lateral size & depth (W,H,D) of the image upsampled in the first place

# Outputs
- `dx`: Downsampled version of `Δ`
"""
∇upsample_trilinear(Δ; size, align_corners::Bool = true) = ∇upsample_linear(Δ; size, align_corners)

# The type of the source positions and of the interpolation weights. In Float16 and
# BFloat16, the positions drift by whole pixels on long axes and can go out of bounds,
# so they are computed in Float32, as in ATen.
_weight_type(T) = real(T) <: Union{Float16, BFloat16} ? Float32 : real(T)

# the weights of 16-bit floats are Float32
@inline _atomic_add!(dx, idx, val) = _atomic_scatter!(dx, idx, +, convert(eltype(dx), val))

function upsample_linear_kernel!(
    y::AbstractArray{T, N}, x::AbstractArray{T, N}; align_corners::Bool = true,
) where {T, N}
    backend = KernelAbstractions.get_backend(x)
    ndrange = backend isa CPU ?
        size(y)[N - 1:end] : # Parallelization along channel x batch.
        size(y)[1:N - 2] # Parallelization along WHD.
    ratios = align_corners ?
        ntuple(i -> _weight_type(T)((size(x, i) - 1) / (size(y, i) - 1)), N - 2) :
        ntuple(i -> _weight_type(T)(size(x, i) / size(y, i)), N - 2)
    _upsample_linear_kernel!(backend)(backend, y, x, ratios..., Val(align_corners); ndrange)
    return y
end

function ∇upsample_linear_kernel!(
    dx::AbstractArray{T, N}, Δ::AbstractArray{T, N}; align_corners::Bool = true,
) where {T, N}
    backend = KernelAbstractions.get_backend(dx)
    ndrange = backend isa CPU ?
        size(Δ)[N - 1:end] : # Parallelization along channel x batch.
        size(Δ)[1:N - 2] # Parallelization along WHD.
    ratios = align_corners ?
        ntuple(i -> _weight_type(T)((size(dx, i) - 1) / (size(Δ, i) - 1)), N - 2) :
        ntuple(i -> _weight_type(T)(size(dx, i) / size(Δ, i)), N - 2)
    _∇upsample_linear_kernel!(backend)(backend, dx, Δ, ratios..., Val(align_corners); ndrange)
    return dx
end

# Linear (CPU): parallelization along channel x batch dimensions.

@kernel function _upsample_linear_kernel!(::CPU, y::T, x::T, rwidth, align::Val{A}) where {
    T <: AbstractArray{<:Any, 3}, A,
}
    @uniform in_width, channels, batch = size(x)
    @uniform out_width = size(y, 1)
    c, n = @index(Global, NTuple)
    yv, xv = @view(y[:, c, n]), @view(x[:, c, n])
    @inbounds for i in 1:out_width
        iw0, iw1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, in_width)
        yv[i] = w0λ * xv[iw0] + w1λ * xv[iw1]
    end
end

@kernel function _∇upsample_linear_kernel!(::CPU, dx::T1, Δ::T2, rwidth, align::Val{A}) where {
    T1 <: AbstractArray{<:Any, 3}, T2 <: AbstractArray{<:Any, 3}, A,
}
    @uniform in_width, channels, batch = size(Δ)
    @uniform out_width = size(dx, 1)
    c, n = @index(Global, NTuple)
    Δv, dxv = @view(Δ[:, c, n]), @view(dx[:, c, n])
    @inbounds for i in 1:in_width
        ow0, ow1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, out_width)
        val = Δv[i]
        dxv[ow0] += w0λ * val
        dxv[ow1] += w1λ * val
    end
end

# Linear (GPU): parallelization along width dimension.

@kernel function _upsample_linear_kernel!(::B, y::T, x::T, rwidth, align::Val{A}) where {
    B <: Backend, T <: AbstractArray{<:Any, 3}, A,
}
    @uniform in_width, channels, batch = size(x)
    i = @index(Global)
    iw0, iw1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, in_width)
    @inbounds for n in 1:batch, c in 1:channels
        y[i, c, n] = w0λ * x[iw0, c, n] + w1λ * x[iw1, c, n]
    end
end

@kernel function _∇upsample_linear_kernel!(::B, dx::T, Δ::T, rwidth, align::Val{A}) where {
    B <: Backend, T <: AbstractArray{<:Any, 3}, A,
}
    @uniform in_width, channels, batch = size(Δ)
    @uniform out_width = size(dx, 1)
    i = @index(Global)
    ow0, ow1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, out_width)
    @inbounds for n in 1:batch, c in 1:channels
        val = Δ[i, c, n]
        _atomic_add!(dx, (ow0, c, n), w0λ * val)
        _atomic_add!(dx, (ow1, c, n), w1λ * val)
    end
end

# Bilinear (CPU): parallelization along channel x batch dimensions.

@kernel function _upsample_linear_kernel!(::CPU, y::T, x::T, rwidth, rheight, align::Val{A}) where {
    T <: AbstractArray{<:Any, 4}, A,
}
    @uniform in_width, in_height, channels, batch = size(x)
    @uniform out_width, out_height = size(y)[1:2]
    c, n = @index(Global, NTuple)
    yv, xv = @view(y[:, :, c, n]), @view(x[:, :, c, n])
    for j in 1:out_height
        ih0, ih1, h0λ, h1λ = source_idx_and_λ(rheight, j - 1, align, in_height)
        for i in 1:out_width
            iw0, iw1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, in_width)
            @inbounds yv[i, j] =
                h0λ * (w0λ * xv[iw0, ih0] + w1λ * xv[iw1, ih0]) +
                h1λ * (w0λ * xv[iw0, ih1] + w1λ * xv[iw1, ih1])
        end
    end
end

@kernel function _∇upsample_linear_kernel!(::CPU, dx::T1, Δ::T2, rwidth, rheight, align::Val{A}) where {
    T1 <: AbstractArray{<:Any, 4}, T2 <: AbstractArray{<:Any, 4}, A,
}
    @uniform in_width, in_height, channels, batch = size(Δ)
    @uniform out_width, out_height = size(dx)[1:2]
    c, n = @index(Global, NTuple)
    Δv, dxv = @view(Δ[:, :, c, n]), @view(dx[:, :, c, n])
    for j in 1:in_height
        oh0, oh1, h0λ, h1λ = source_idx_and_λ(rheight, j - 1, align, out_height)
        @inbounds for i in 1:in_width
            ow0, ow1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, out_width)
            val = Δv[i, j]
            dxv[ow0, oh0] += w0λ * h0λ * val
            dxv[ow1, oh0] += w1λ * h0λ * val
            dxv[ow0, oh1] += w0λ * h1λ * val
            dxv[ow1, oh1] += w1λ * h1λ * val
        end
    end
end

# Bilinear (GPU): parallelization along width, height dimensions.

@kernel function _upsample_linear_kernel!(::B, y::T, x::T, rwidth, rheight, align::Val{A}) where {
    B <: Backend, T <: AbstractArray{<:Any, 4}, A,
}
    @uniform in_width, in_height, channels, batch = size(x)
    i, j = @index(Global, NTuple)
    iw0, iw1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, in_width)
    ih0, ih1, h0λ, h1λ = source_idx_and_λ(rheight, j - 1, align, in_height)
    @inbounds for n in 1:batch, c in 1:channels
        y[i, j, c, n] =
            h0λ * (w0λ * x[iw0, ih0, c, n] + w1λ * x[iw1, ih0, c, n]) +
            h1λ * (w0λ * x[iw0, ih1, c, n] + w1λ * x[iw1, ih1, c, n])
    end
end

@kernel function _∇upsample_linear_kernel!(::B, dx::T, Δ::T, rwidth, rheight, align::Val{A}) where {
    B <: Backend, T <: AbstractArray{<:Any, 4}, A,
}
    @uniform in_width, in_height, channels, batch = size(Δ)
    @uniform out_width, out_height = size(dx)[1:2]
    i, j = @index(Global, NTuple)
    ow0, ow1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, out_width)
    oh0, oh1, h0λ, h1λ = source_idx_and_λ(rheight, j - 1, align, out_height)
    @inbounds for n in 1:batch, c in 1:channels
        val = Δ[i, j, c, n]
        _atomic_add!(dx, (ow0, oh0, c, n), w0λ * h0λ * val)
        _atomic_add!(dx, (ow1, oh0, c, n), w1λ * h0λ * val)
        _atomic_add!(dx, (ow0, oh1, c, n), w0λ * h1λ * val)
        _atomic_add!(dx, (ow1, oh1, c, n), w1λ * h1λ * val)
    end
end

# Trilinear (CPU): parallelization along channel x batch dimensions.

@kernel function _upsample_linear_kernel!(::CPU, y::T, x::T, rwidth, rheight, rdepth, align::Val{A}) where {
    T <: AbstractArray{<:Any, 5}, A,
}
    @uniform in_width, in_height, in_depth = size(x)[1:3]
    @uniform channels, batch = size(x, 4), size(x, 5)
    @uniform out_width, out_height, out_depth = size(y)[1:3]
    c, n = @index(Global, NTuple)
    yv, xv = @view(y[:, :, :, c, n]), @view(x[:, :, :, c, n])
    for k in 1:out_depth
        id0, id1, d0λ, d1λ = source_idx_and_λ(rdepth, k - 1, align, in_depth)
        for j in 1:out_height
            ih0, ih1, h0λ, h1λ = source_idx_and_λ(rheight, j - 1, align, in_height)
            for i in 1:out_width
                iw0, iw1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, in_width)
                @inbounds yv[i, j, k] =
                    d0λ * (
                        h0λ * (w0λ * xv[iw0, ih0, id0] + w1λ * xv[iw1, ih0, id0]) +
                        h1λ * (w0λ * xv[iw0, ih1, id0] + w1λ * xv[iw1, ih1, id0])) +
                    d1λ * (
                        h0λ * (w0λ * xv[iw0, ih0, id1] + w1λ * xv[iw1, ih0, id1]) +
                        h1λ * (w0λ * xv[iw0, ih1, id1] + w1λ * xv[iw1, ih1, id1]))
            end
        end
    end
end

@kernel function _∇upsample_linear_kernel!(::CPU, dx::T1, Δ::T2, rwidth, rheight, rdepth, align::Val{A}) where {
    T1 <: AbstractArray{<:Any, 5}, T2 <: AbstractArray{<:Any, 5}, A,
}
    @uniform in_width, in_height, in_depth = size(Δ)[1:3]
    @uniform channels, batch = size(Δ, 4), size(Δ, 5)
    @uniform out_width, out_height, out_depth = size(dx)[1:3]
    c, n = @index(Global, NTuple)
    Δv, dxv = @view(Δ[:, :, :, c, n]), @view(dx[:, :, :, c, n])
    for k in 1:in_depth
        od0, od1, d0λ, d1λ = source_idx_and_λ(rdepth, k - 1, align, out_depth)
        for j in 1:in_height
            oh0, oh1, h0λ, h1λ = source_idx_and_λ(rheight, j - 1, align, out_height)
            @inbounds for i in 1:in_width
                ow0, ow1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, out_width)
                val = Δv[i, j, k]
                dxv[ow0, oh0, od0] += w0λ * h0λ * d0λ * val
                dxv[ow1, oh0, od0] += w1λ * h0λ * d0λ * val
                dxv[ow0, oh1, od0] += w0λ * h1λ * d0λ * val
                dxv[ow1, oh1, od0] += w1λ * h1λ * d0λ * val

                dxv[ow0, oh0, od1] += w0λ * h0λ * d1λ * val
                dxv[ow1, oh0, od1] += w1λ * h0λ * d1λ * val
                dxv[ow0, oh1, od1] += w0λ * h1λ * d1λ * val
                dxv[ow1, oh1, od1] += w1λ * h1λ * d1λ * val
            end
        end
    end
end

# Trilinear (GPU): parallelization along width x height x depth dimensions.

@kernel function _upsample_linear_kernel!(::B, y::T, x::T, rwidth, rheight, rdepth, align::Val{A}) where {
    B <: Backend, T <: AbstractArray{<:Any, 5}, A,
}
    @uniform in_width, in_height, in_depth = size(x)[1:3]
    @uniform channels, batch = size(x, 4), size(x, 5)
    i, j, k = @index(Global, NTuple)
    iw0, iw1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, in_width)
    ih0, ih1, h0λ, h1λ = source_idx_and_λ(rheight, j - 1, align, in_height)
    id0, id1, d0λ, d1λ = source_idx_and_λ(rdepth, k - 1, align, in_depth)
    @inbounds for n in 1:batch, c in 1:channels
        y[i, j, k, c, n] =
            d0λ * (
                h0λ * (w0λ * x[iw0, ih0, id0, c, n] + w1λ * x[iw1, ih0, id0, c, n]) +
                h1λ * (w0λ * x[iw0, ih1, id0, c, n] + w1λ * x[iw1, ih1, id0, c, n])) +
            d1λ * (
                h0λ * (w0λ * x[iw0, ih0, id1, c, n] + w1λ * x[iw1, ih0, id1, c, n]) +
                h1λ * (w0λ * x[iw0, ih1, id1, c, n] + w1λ * x[iw1, ih1, id1, c, n]))
    end
end

@kernel function _∇upsample_linear_kernel!(::B, dx::T, Δ::T, rwidth, rheight, rdepth, align::Val{A}) where {
    B <: Backend, T <: AbstractArray{<:Any, 5}, A,
}
    @uniform in_width, in_height, in_depth = size(Δ)[1:3]
    @uniform channels, batch = size(Δ, 4), size(Δ, 5)
    @uniform out_width, out_height, out_depth = size(dx)[1:3]
    i, j, k = @index(Global, NTuple)
    ow0, ow1, w0λ, w1λ = source_idx_and_λ(rwidth, i - 1, align, out_width)
    oh0, oh1, h0λ, h1λ = source_idx_and_λ(rheight, j - 1, align, out_height)
    od0, od1, d0λ, d1λ = source_idx_and_λ(rdepth, k - 1, align, out_depth)
    @inbounds for n in 1:batch, c in 1:channels
        val = Δ[i, j, k, c, n]
        _atomic_add!(dx, (ow0, oh0, od0, c, n), w0λ * h0λ * d0λ * val)
        _atomic_add!(dx, (ow1, oh0, od0, c, n), w1λ * h0λ * d0λ * val)
        _atomic_add!(dx, (ow0, oh1, od0, c, n), w0λ * h1λ * d0λ * val)
        _atomic_add!(dx, (ow1, oh1, od0, c, n), w1λ * h1λ * d0λ * val)

        _atomic_add!(dx, (ow0, oh0, od1, c, n), w0λ * h0λ * d1λ * val)
        _atomic_add!(dx, (ow1, oh0, od1, c, n), w1λ * h0λ * d1λ * val)
        _atomic_add!(dx, (ow0, oh1, od1, c, n), w0λ * h1λ * d1λ * val)
        _atomic_add!(dx, (ow1, oh1, od1, c, n), w1λ * h1λ * d1λ * val)
    end
end

@inline function source_idx_and_λ(
    ratio::T, out_idx::Int, ::Val{align}, in_width::Int,
) where {T, align}
    real_index = align ?
        ratio * out_idx :
        max(zero(T), ratio * (out_idx + T(0.5)) - T(0.5))

    iw0 = if T <: Rational
        floor(Int, real_index) # Not GPU-friendly, but allows for Rational support.
    else
        unsafe_trunc(Int, floor(real_index))
    end
    offset = ifelse(iw0 < in_width - 1, 1, 0)
    iw1 = iw0 + offset + 1

    w1lambda = real_index - iw0
    w0lambda = one(T) - w1lambda
    return iw0 + 1, iw1, w0lambda, w1lambda
end

#
# Bicubic and antialiased resampling: one 1-D pass per resized dimension, each a banded matrix
# whose rows hold the weights of ATen's `upsample_bicubic2d`, `_upsample_bicubic2d_aa` and
# `_upsample_bilinear2d_aa` (aten/src/ATen/native/UpSample.h and cpu/UpSampleKernel.cpp),
# computed on the host in the precision ATen uses. As ATen does, the first dimension (PyTorch's W)
# is resized first.
#

"""
    upsample_bicubic(x::AbstractArray{T,4}, scale::NTuple{2,Real}; align_corners::Bool = false, antialias::Bool = false)
    upsample_bicubic(x::AbstractArray{T,4}; size::NTuple{2,Integer}, align_corners::Bool = false, antialias::Bool = false)

Resizes the first 2 dimensions of the array `x` with bicubic interpolation, to the image `size`
given as a keyword argument, or to `floor.(Int, scale .* size(x)[1:2])`. The output can be
smaller than the input.

This is PyTorch's `F.interpolate(x, size, mode="bicubic", align_corners=..., antialias=...)` on
the same data in the NCHW layout. Without `antialias`, each output pixel is a weighted sum of 4×4
input pixels (the cubic convolution kernel with `a = -0.75`, as in OpenCV), and the image is
extended by repeating its border pixels. With `antialias = true`, the kernel is the one of PIL and
torchvision's `Resize` (`a = -0.5`): when downsampling, it is stretched by the downsampling factor,
so that each output pixel averages all the input pixels it covers, and at the borders the weights
are renormalised over the pixels inside the image.

Unlike [`upsample_bilinear`](@ref), `align_corners` defaults to `false`, as in PyTorch.

# Examples

```jldoctest
julia> x = reshape(Float32[1 2; 3 4], (2, 2, 1, 1));

julia> upsample_bicubic(x, (2, 2))  # overshoots the range of `x`
4×4×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 0.683594  1.01562  1.5625   1.89453
 1.34766   1.67969  2.22656  2.55859
 2.44141   2.77344  3.32031  3.65234
 3.10547   3.4375   3.98438  4.31641

julia> ans == upsample_bicubic(x; size = (4, 4))
true

julia> x = reshape(Float32.(1:16), (4, 4, 1, 1));

julia> upsample_bicubic(x; size = (2, 2), antialias = true)
2×2×1×1 Array{Float32, 4}:
[:, :, 1, 1] =
 3.93388  11.2397
 5.76033  13.0661
```
"""
upsample_bicubic(x::AbstractArray{<:Any,4}, scale::Union{Real,NTuple{2,Real}};
                 align_corners::Bool = false, antialias::Bool = false) =
    upsample_bicubic(x; size = _upsample_size(x, scale), align_corners, antialias)

upsample_bicubic(x::AbstractArray{<:Any,4}; size::NTuple{2,Integer},
                 align_corners::Bool = false, antialias::Bool = false) =
    _upsample_separable(x, size, :bicubic, align_corners, antialias)

"""
    ∇upsample_bicubic(Δ::AbstractArray{T,4}; size::NTuple{2,Integer}, align_corners::Bool = false, antialias::Bool = false)

# Arguments
- `Δ`: Incoming gradient array, backpropagated from downstream layers
- `size`: Lateral (W,H) size of the image resized in the first place

# Outputs
- `dx`: Gradient with respect to the input of [`upsample_bicubic`](@ref)
"""
∇upsample_bicubic(Δ::AbstractArray{<:Any,4}; size::NTuple{2,Integer},
                  align_corners::Bool = false, antialias::Bool = false) =
    ∇_upsample_separable(Δ, size, :bicubic, align_corners, antialias)

_upsample_size(x::AbstractArray{<:Any,N}, scale::Real) where N = _upsample_size(x, ntuple(_ -> scale, N - 2))
_upsample_size(x::AbstractArray, scale::Tuple) = map((s, n) -> floor(Int, s * n), scale, size(x)[1:length(scale)])

function _upsample_separable(x::AbstractArray{T,N}, size::NTuple{M,Integer}, mode::Symbol,
                             align_corners::Bool, antialias::Bool) where {T,N,M}
    M == N - 2 || throw(ArgumentError("expected a size with $(N - 2) entries, got $size"))
    all(>(0), size) || throw(ArgumentError("expected a positive output size, got $size"))
    y = x
    for d in 1:M
        Base.size(x, d) == size[d] && continue
        weights = _upsample_weights(float(_weight_type(T)), Base.size(x, d), size[d], mode, align_corners, antialias)
        y = _resample(y, d, weights...)
    end
    return y === x ? copy(x) : y
end

function ∇_upsample_separable(Δ::AbstractArray{T,N}, size::NTuple{M,Integer}, mode::Symbol,
                              align_corners::Bool, antialias::Bool) where {T,N,M}
    dx = Δ
    for d in M:-1:1
        Base.size(Δ, d) == size[d] && continue
        weights = _upsample_weights(float(_weight_type(T)), size[d], Base.size(Δ, d), mode, align_corners, antialias)
        dx = _resample(dx, d, _transpose_weights(weights..., size[d])...)
    end
    return dx
end

function rrule(::typeof(_upsample_separable), x::AbstractArray{<:Any,N}, size, mode, align_corners, antialias) where N
    y = _upsample_separable(x, size, mode, align_corners, antialias)
    insize = Base.size(x)[1:N-2]
    function upsample_separable_pullback(Δ)
        dx = ∇_upsample_separable(unthunk(Δ), insize, mode, align_corners, antialias)
        return (NoTangent(), dx, NoTangent(), NoTangent(), NoTangent(), NoTangent())
    end
    return y, upsample_separable_pullback
end

# Resizes dimension `d` of `x`: output `i` is the sum of `weight[t, i] * x[.., index[t, i], ..]`
# over `t in 1:len[i]`, accumulated in this order.
function _resample(x::AbstractArray{T,N}, d::Int, index, len, weight) where {T,N}
    sz = size(x)
    pre, post = prod(sz[1:d-1]), prod(sz[d+1:N])
    m = length(len)
    y = similar(x, T, ntuple(i -> i == d ? m : sz[i], N))
    isempty(y) && return y
    on_device(a) = copyto!(similar(y, eltype(a), Base.size(a)), a)
    backend = get_backend(y)
    _resample_kernel!(backend)(reshape(y, pre, m, post), reshape(x, pre, sz[d], post),
        on_device(index), on_device(len), on_device(weight); ndrange = (pre, m, post))
    return y
end

@kernel function _resample_kernel!(y, @Const(x), @Const(index), @Const(len), @Const(weight))
    p, i, q = @index(Global, NTuple)
    acc = zero(promote_type(eltype(weight), eltype(x)))
    @inbounds for t in 1:len[i]
        acc = muladd(weight[t, i], x[p, index[t, i], q], acc)
    end
    @inbounds y[p, i, q] = acc
end

# ATen's `cubic_convolution1/2`. Its CPU kernels are compiled with FMA contraction, which fuses
# other operations in the antialiased filter than in `get_cubic_upsample_coefficients`: these
# `fma` calls reproduce both sets of weights bit for bit.
_cubic1(x, a) = fma(fma(a + 2, x, -(a + 3)) * x, x, one(x))
_cubic2(x, a) = fma(fma(fma(a, x, -5a), x, 8a), x, -4a)
_cubic1_coeff(x, a) = fma(a + 2, x, -(a + 3)) * x * x + 1
_cubic2_coeff(x, a) = fma(fma(a, x, -5a), x, 8a) * x - 4a

function _cubic_filter(x, a)
    x = abs(x)
    return x < 1 ? _cubic1(x, a) : x < 2 ? _cubic2(x, a) : zero(x)
end

_linear_filter(x) = (x = abs(x); x < 1 ? 1 - x : zero(x))

# The cubic kernel of the antialiased mode is Keys' (`a = -0.5`), as PIL's.
_aa_filter(mode, x) = mode === :bicubic ? _cubic_filter(x, oftype(x, -0.5)) : _linear_filter(x)

# The weights of a dimension resized from `n_in` to `n_out` pixels, in the form `_resample` takes.
# Mixed `W`/Float64 arithmetic follows ATen's (its `double` literals and casts) for `W = Float32`.
function _upsample_weights(::Type{W}, n_in::Int, n_out::Int, mode::Symbol,
                           align_corners::Bool, antialias::Bool) where W
    scale = align_corners ? (n_out > 1 ? W(n_in - 1) / W(n_out - 1) : zero(W)) : W(n_in) / W(n_out)
    if !antialias
        mode === :bicubic || throw(ArgumentError("unsupported mode $mode without antialias"))
        A = W(-0.75)
        index = Matrix{Int}(undef, 4, n_out)
        weight = Matrix{W}(undef, 4, n_out)
        for i in 1:n_out
            src = align_corners ? scale * W(i - 1) : fma(scale, W(i - 1) + W(0.5), W(-0.5))
            k = min(floor(Int, Float32(src)), n_in - 1)  # ATen floors in Float32, `floorf`
            t = clamp(src - W(k), zero(W), one(W))
            # Inputs k-1:k+2 (from 0), with the second first: ATen's compiled sum starts there.
            index[:, i] .= clamp.((k + 1, k, k + 2, k + 3), 1, n_in)
            weight[:, i] .= (_cubic1_coeff(t, A), _cubic2_coeff(t + 1, A),
                             _cubic1_coeff(1 - t, A), _cubic2_coeff((1 - t) + 1, A))
        end
        return index, fill(4, n_out), weight
    end
    mode in (:bicubic, :bilinear) || throw(ArgumentError("unsupported mode $mode"))
    interp_size = mode === :bicubic ? 4 : 2
    support = scale >= 1 ? W(interp_size * 0.5 * scale) : W(interp_size * 0.5)
    invscale = scale >= 1 ? W(1 / Float64(scale)) : one(W)
    taps = ceil(Int, support) * 2 + 1
    index, len = ones(Int, taps, n_out), Vector{Int}(undef, n_out)
    weight = zeros(W, taps, n_out)
    for i in 1:n_out
        center = W(scale * (i - 0.5))
        xmin = max(trunc(Int, center - support + 0.5), 0)
        xsize = clamp(min(trunc(Int, center + support + 0.5), n_in) - xmin, 0, taps)
        total = zero(W)
        for j in 1:xsize
            w = _aa_filter(mode, W((j - 1 + xmin - center + 0.5) * invscale))
            index[j, i], weight[j, i] = xmin + j, w
            total += w
        end
        total == 0 || (weight[1:xsize, i] ./= total)
        len[i] = xsize
    end
    return index, len, weight
end

# The weights of the transposed pass, from the outputs back to the `n_in` inputs.
function _transpose_weights(index, len, weight::AbstractMatrix{W}, n_in) where W
    n_out = length(len)
    lo, hi = fill(n_out + 1, n_in), fill(0, n_in)
    for i in 1:n_out, t in 1:len[i]
        k = index[t, i]
        lo[k], hi[k] = min(lo[k], i), max(hi[k], i)
    end
    tlen = max.(hi .- lo .+ 1, 0)
    tindex = [lo[k] + s - 1 for s in 1:maximum(tlen), k in 1:n_in]
    tweight = zeros(W, size(tindex))
    for i in 1:n_out, t in 1:len[i]
        k = index[t, i]
        tweight[i - lo[k] + 1, k] += weight[t, i]
    end
    return tindex, tlen, tweight
end
