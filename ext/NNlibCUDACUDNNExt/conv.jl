
using NNlib: DenseConvDims
import NNlib: conv!, ∇conv_filter!, ∇conv_data!, conv_bias_act!

using cuDNN: CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION, convolution!,
             convolution_data_gradient!, convolution_filter_gradient!

const CUDNNFloat = Union{Float16,BFloat16,Float32,Float64}
const CUDNNComplexFloat = Union{ComplexF16,ComplexF32,ComplexF64}

# cuDNN takes dense arrays, so views and other wrappers of a `CuArray` are copied into one
# (#863). Not with `copy`, whose `getindex` is up to 17× slower (JuliaGPU/GPUArrays.jl#837).
dense(x::DenseCuArray) = x
dense(x::AnyCuArray) = copyto!(similar(x), x)

# Compute (accumulation) type for the convolution. cuDNN's TRUE_HALF_CONFIG (compute
# in Float16) is only supported for a narrow set of 2D shapes; in particular it is
# unsupported for 3D convolutions and for many backward-filter shapes, surfacing as
# CUDNN_STATUS_NOT_SUPPORTED/BAD_PARAM. PSEUDO_HALF_CONFIG (Float16 data, Float32
# compute) is broadly supported and is the standard choice (matching e.g. PyTorch),
# so Float16 convolutions always compute in Float32.
# Fixes https://github.com/FluxML/NNlib.jl/issues/505 and #515.
# BFloat16 has no native cuDNN accumulation config at all, so it likewise computes
# in Float32.
conv_compute_type(::Type{Float16}) = Float32
conv_compute_type(::Type{BFloat16}) = Float32
conv_compute_type(::Type{T}) where T = T

conv_mode(cdims::DenseConvDims) =
    NNlib.flipkernel(cdims) ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION

conv_kwargs(cdims::DenseConvDims, ::Type{T}) where {T} =
    (padding=NNlib.padding(cdims),
     stride=NNlib.stride(cdims),
     dilation=NNlib.dilation(cdims),
     groups=NNlib.groupcount(cdims),
     mode=conv_mode(cdims),
     compute_type=conv_compute_type(real(T)))

conv_bias_activation(::typeof(NNlib.relu)) = :relu
conv_bias_activation(::Any) = nothing

# Zero-pad `x` by the asymmetric part of the padding of `cdims`. Returns the padded array
# and the symmetric padding that remains, or `x` itself if the padding is symmetric.
function symmetric_padded_input(x::DenseCuArray, cdims::DenseConvDims)
    pad = NNlib.padding(cdims)
    nd = NNlib.spatial_dims(cdims)
    lo = ntuple(i -> pad[2i-1], nd)
    hi = ntuple(i -> pad[2i], nd)
    lo == hi && return x, lo
    sym = min.(lo, hi)
    sz = ntuple(i -> i <= nd ? size(x, i) + abs(lo[i] - hi[i]) : size(x, i), ndims(x))
    inner = ntuple(i -> i <= nd ? (1:size(x, i)) .+ (lo[i] - sym[i]) : axes(x, i), ndims(x))
    xp = fill!(similar(x, sz), 0)
    xp[inner...] = x
    return xp, sym
end

@inline function combine_complex!(y::DenseCuArray{T1}, yr::DenseCuArray{T2},
                                  yi::DenseCuArray{T2}; bias=zero(T1), alpha=one(T1),
                                  beta=zero(T1), σ=identity,
) where {T1<:CUDNNComplexFloat,T2<:CUDNNFloat}
    # if y is from similar(), it may have NaNs, and beta*NaN will propagate.
    if beta != 0
        @. y = σ(alpha*(yr + im*yi) + bias + beta*y)
    else
        @. y = σ(alpha*(yr + im*yi) + bias)
    end
    return y
end

function conv!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, cdims::DenseConvDims;
               alpha=1, beta=0, algo=-1) where T<:CUDNNFloat
    if algo != -1
        @warn "algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end
    convolution!(y, x, w; conv_kwargs(cdims, T)..., alpha, beta)
end

# Complex convolution with Gauss's trick (1 complex mul === 3 real mul):
# Consider x = xr + im*xi, y = yr + im*yi,
# so x*y = (xr*yr - xi*yi) + im*(xr*yi + xi*yr).
# Let a = xr*yr,
#     b = xi*yi,
#     c = (xr + xi)*(yr + yi) = xr*yr + xr*yi + xi*yr + xi*yi.
# Then,
# x*y = (a - b) + im*(c - a - b).
# Convolution is linear so this multiplication trick translates to convolution.
function conv!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, cdims::DenseConvDims;
               alpha=1, beta=0, algo=-1) where T<:CUDNNComplexFloat
    xr, xi = reim(x)
    wr, wi = reim(w)
    a = conv!(similar(real(y)), xr, wr, cdims; algo=algo)
    b = conv!(similar(a), xi, wi, cdims; algo=algo)
    c = conv!(similar(a), xr + xi, wr + wi, cdims; algo=algo)
    return combine_complex!(y, a - b, c - a - b; alpha, beta)
end

# (xr + im*xi) * w = xr*w + im*(xi*w)
function conv!(y::DenseCuArray{T1}, x::DenseCuArray{T1}, w::DenseCuArray{T2}, cdims::DenseConvDims;
               alpha=1, beta=0, algo=-1) where {T1<:CUDNNComplexFloat, T2<:CUDNNFloat}
    xr, xi = reim(x)
    yr = conv!(similar(real(y)), xr, w, cdims; algo=algo)
    yi = conv!(similar(yr), xi, w, cdims; algo=algo)
    return combine_complex!(y, yr, yi; alpha, beta)
end

# x * (wr + im*wi) = x*wr + im*(x*wi)
function conv!(y::DenseCuArray{T1}, x::DenseCuArray{T2}, w::DenseCuArray{T1}, cdims::DenseConvDims;
               alpha=1, beta=0, algo=-1) where {T1<:CUDNNComplexFloat, T2<:CUDNNFloat}
    wr, wi = reim(w)
    yr = conv!(similar(real(y)), x, wr, cdims; algo=algo)
    yi = conv!(similar(yr), x, wi, cdims; algo=algo)
    return combine_complex!(y, yr, yi; alpha, beta)
end

function conv_bias_act!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T},
                        cdims::DenseConvDims, bias::DenseCuArray{T}, σ=identity;
                        z::DenseCuArray{T}=y, alpha=1, beta=0, algo=-1) where T<:CUDNNFloat
    if algo != -1
        @warn "The algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end
    act = conv_bias_activation(σ)
    convolution!(y, x, w; conv_kwargs(cdims, T)..., alpha, beta, z, bias,
                 activation=act)
    if act === nothing && σ ∉ (nothing, identity)
        @. y = σ(y)
    end
    return y
end

function conv_bias_act!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T},
                        cdims::DenseConvDims, bias::DenseCuArray{T}, σ=identity;
                        z::DenseCuArray{T}=y, alpha=1, beta=0, algo=-1) where T<:CUDNNComplexFloat
    xr, xi = reim(x)
    wr, wi = reim(w)
    a = conv!(similar(real(y)), xr, wr, cdims; alpha=1, beta=0, algo=algo)
    b = conv!(similar(a), xi, wi, cdims; alpha=1, beta=0, algo=algo)
    c = conv!(similar(a), xr + xi, wr + wi, cdims; alpha=1, beta=0, algo=algo)
    return combine_complex!(y, a - b, c - a - b; bias, alpha, beta, σ)
end

function ∇conv_data!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, w::DenseCuArray{T},
                     cdims::DenseConvDims; alpha=1, beta=0, algo=-1) where T<:CUDNNFloat
    if algo != -1
        @warn "The algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end
    convolution_data_gradient!(dx, dy, w; conv_kwargs(cdims, T)..., alpha, beta)
end

function ∇conv_data!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, w::DenseCuArray{T},
                     cdims::DenseConvDims; alpha=1, beta=0, algo=-1) where T<:CUDNNComplexFloat
    dyr, dyi = reim(dy)
    wr, wi = reim(w)
    # note: w is conjugated, i.e. wi is negated below
    a = ∇conv_data!(similar(real(dx)), dyr, wr, cdims; alpha=1, beta=0, algo=algo)
    b = ∇conv_data!(similar(a), dyi, -wi, cdims; alpha=1, beta=0, algo=algo)
    c = ∇conv_data!(similar(a), dyr + dyi, wr - wi, cdims; alpha=1, beta=0, algo=algo)
    return combine_complex!(dx, a - b, c - a - b; alpha, beta)
end

# dx = (dyr + im*dyi)*w = dyr*w + im*(dyi*w)
function ∇conv_data!(dx::DenseCuArray{T1}, dy::DenseCuArray{T1}, w::DenseCuArray{T2},
                     cdims::DenseConvDims; alpha=1, beta=0, algo=-1) where {T1<:CUDNNComplexFloat, T2<:CUDNNFloat}
    dyr, dyi = reim(dy)
    dxr = ∇conv_data!(similar(real(dx)), dyr, w, cdims; alpha=1, beta=0, algo=algo)
    dxi = ∇conv_data!(similar(dxr), dyi, w, cdims; alpha=1, beta=0, algo=algo)
    return combine_complex!(dx, dxr, dxi; alpha, beta)
end

function ∇conv_filter!(dw::DenseCuArray{T}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                       cdims::DenseConvDims; alpha=1, beta=0, algo=-1) where T<:CUDNNFloat
    if algo != -1
        @warn "The algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end
    # With asymmetric padding, cuDNN's filter gradient silently returns wrong values for
    # some engines (3D convolutions on an A100 with cuDNN 9.26; cuDNN.jl works around
    # the same in 1D), unlike the forward pass and the data gradient: pad the input.
    x, padding = symmetric_padded_input(x, cdims)
    convolution_filter_gradient!(dw, x, dy; conv_kwargs(cdims, T)..., padding, alpha, beta)
end

function ∇conv_filter!(dw::DenseCuArray{T}, x::DenseCuArray{T}, dy::DenseCuArray{T},
                       cdims::DenseConvDims; alpha=1, beta=0, algo=-1) where T<:CUDNNComplexFloat
    xr, xi = reim(x)
    dyr, dyi = reim(dy)
    # note: x is conjugated, i.e. xi is negated below
    a = ∇conv_filter!(similar(real(dw)), xr, dyr, cdims; alpha=1, beta=0, algo=algo)
    b = ∇conv_filter!(similar(a), -xi, dyi, cdims; alpha=1, beta=0, algo=algo)
    c = ∇conv_filter!(similar(a), xr - xi, dyr + dyi, cdims; alpha=1, beta=0, algo=algo)
    return combine_complex!(dw, a - b, c - a - b; alpha, beta)
end

# dw = x*(dyr + im*dyi) = x*dyr + im*(x*dyi)
function ∇conv_filter!(dw::DenseCuArray{T1}, x::DenseCuArray{T2}, dy::DenseCuArray{T1},
                       cdims::DenseConvDims; alpha=1, beta=0, algo=-1) where {T1<:CUDNNComplexFloat, T2<:CUDNNFloat}
    dyr, dyi = reim(dy)
    dwr = ∇conv_filter!(similar(real(dw)), x, dyr, cdims; alpha=1, beta=0, algo=algo)
    dwi = ∇conv_filter!(similar(dwr), x, dyi, cdims; alpha=1, beta=0, algo=algo)
    return combine_complex!(dw, dwr, dwi; alpha, beta)
end

# Non-dense inputs. The dense methods above are more specific, so dense arrays never land here.
const CUDNNEltype = Union{CUDNNFloat,CUDNNComplexFloat}

conv!(y::DenseCuArray{T}, x::AnyCuArray{T}, w::AnyCuArray{T}, cdims::DenseConvDims;
      kwargs...) where T<:CUDNNEltype =
    conv!(y, dense(x), dense(w), cdims; kwargs...)

conv_bias_act!(y::DenseCuArray{T}, x::AnyCuArray{T}, w::AnyCuArray{T}, cdims::DenseConvDims,
               bias::AnyCuArray{T}, σ=identity; z::AnyCuArray{T}=y,
               kwargs...) where T<:CUDNNEltype =
    conv_bias_act!(y, dense(x), dense(w), cdims, dense(bias), σ; z=dense(z), kwargs...)

∇conv_data!(dx::DenseCuArray{T}, dy::AnyCuArray{T}, w::AnyCuArray{T}, cdims::DenseConvDims;
            kwargs...) where T<:CUDNNEltype =
    ∇conv_data!(dx, dense(dy), dense(w), cdims; kwargs...)

∇conv_filter!(dw::DenseCuArray{T}, x::AnyCuArray{T}, dy::AnyCuArray{T}, cdims::DenseConvDims;
              kwargs...) where T<:CUDNNEltype =
    ∇conv_filter!(dw, dense(x), dense(dy), cdims; kwargs...)
