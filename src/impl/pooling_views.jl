# Pooling as broadcasts over one strided view of the padded input per kernel position, for
# arrays that the direct kernels can't index and a window that the vendor libraries don't
# have (dilation in cuDNN, issue #896). It takes any `PoolDims` and array rank, and scans
# the kernel in the order of the direct kernels, so maxpool picks the same element on ties.

# Padded input extent: the last window and the explicit padding both fit
function padded_size(pdims::PoolDims)
    return ntuple(spatial_dims(pdims)) do i
        reach = (output_size(pdims)[i] - 1) * stride(pdims)[i] +
                (kernel_size(pdims)[i] - 1) * dilation(pdims)[i] + 1
        max(reach, input_size(pdims)[i] + padding(pdims)[2i-1] + padding(pdims)[2i])
    end
end

# Where the input sits in the padded array
input_range(pdims::PoolDims, i) = padding(pdims)[2i-1] .+ (1:input_size(pdims)[i])

function padded_copy(x::AbstractArray, pdims::PoolDims, value)
    xp = fill!(similar(x, padded_size(pdims)..., size(x)[end-1:end]...), value)
    xp[ntuple(i -> input_range(pdims, i), spatial_dims(pdims))..., :, :] .= x
    return xp
end

# The element at kernel position `κ` of every window
function window(xp::AbstractArray, pdims::PoolDims, κ::CartesianIndex)
    ranges = ntuple(spatial_dims(pdims)) do i
        start = 1 + (κ[i] - 1) * dilation(pdims)[i]
        start:stride(pdims)[i]:start + (output_size(pdims)[i] - 1) * stride(pdims)[i]
    end
    return view(xp, ranges..., :, :)
end

kernel_positions(pdims::PoolDims) = CartesianIndices(kernel_size(pdims))

# How many elements each window averages: those of the input, plus those of the explicit
# padding with `count_include_pad`, but not what `ceil_mode` adds past it. The shape is
# that of the output with singleton channel and batch dimensions.
function window_counts(like::AbstractArray, pdims::PoolDims, count_include_pad::Bool)
    T = eltype(like)
    included = fill!(similar(like, T, padded_size(pdims)..., 1, 1), 0)
    ranges = ntuple(spatial_dims(pdims)) do i
        count_include_pad ? (1:input_size(pdims)[i] + padding(pdims)[2i-1] + padding(pdims)[2i]) :
                            input_range(pdims, i)
    end
    included[ranges..., :, :] .= 1
    counts = fill!(similar(like, T, output_size(pdims)..., 1, 1), 0)
    for κ in kernel_positions(pdims)
        counts .+= window(included, pdims, κ)
    end
    return max.(counts, one(T))
end

# `dx = grad + beta * dx` for the part of the padded gradient that lies on the input
function add_input_gradient!(dx::AbstractArray, dxp::AbstractArray, pdims::PoolDims, beta)
    grad = view(dxp, ntuple(i -> input_range(pdims, i), spatial_dims(pdims))..., :, :)
    return iszero(beta) ? (dx .= grad) : (dx .= grad .+ beta .* dx)
end

function maxpool_views!(y::AbstractArray{T}, x::AbstractArray, pdims::PoolDims;
                        alpha=1, kwargs...) where T
    check_dims(size(x), size(y), pdims)
    init = T <: AbstractFloat ? nextfloat(typemin(T)) : typemin(T)
    xp = padded_copy(x, pdims, init)
    fill!(y, init)
    for κ in kernel_positions(pdims)
        v = window(xp, pdims, κ)
        y .= ifelse.(v .> y, v, y)
    end
    return isone(alpha) ? y : (y .*= alpha)
end

function ∇maxpool_views!(dx::AbstractArray{T}, dy::AbstractArray, y::AbstractArray,
                         x::AbstractArray, pdims::PoolDims; alpha=1, beta=0, kwargs...) where T
    check_dims(size(x), size(dy), pdims)
    init = eltype(y) <: AbstractFloat ? nextfloat(typemin(eltype(y))) : typemin(eltype(y))
    xp = padded_copy(x, pdims, init)
    # which kernel position holds a window's maximum, zero if the window has no element
    m = fill!(similar(dy, eltype(y)), init)
    kmax = fill!(similar(dy, Int), 0)
    for (i, κ) in enumerate(kernel_positions(pdims))
        v = window(xp, pdims, κ)
        kmax .= ifelse.(v .> m, i, kmax)
        m .= ifelse.(v .> m, v, m)
    end
    dxp = fill!(similar(dx, padded_size(pdims)..., size(dx)[end-1:end]...), 0)
    grad = T(alpha) .* dy
    for (i, κ) in enumerate(kernel_positions(pdims))
        w = window(dxp, pdims, κ)
        w .+= ifelse.(kmax .== i, grad, zero(T))
    end
    add_input_gradient!(dx, dxp, pdims, beta)
    return dx
end

function meanpool_views!(y::AbstractArray{T}, x::AbstractArray, pdims::PoolDims;
                         alpha=1, count_include_pad::Bool=true, kwargs...) where T
    check_dims(size(x), size(y), pdims)
    xp = padded_copy(x, pdims, zero(T))
    fill!(y, 0)
    for κ in kernel_positions(pdims)
        y .+= window(xp, pdims, κ)
    end
    y .= T(alpha) .* y ./ window_counts(y, pdims, count_include_pad)
    return y
end

function ∇meanpool_views!(dx::AbstractArray{T}, dy::AbstractArray, y::AbstractArray,
                          x::AbstractArray, pdims::PoolDims; alpha=1, beta=0,
                          count_include_pad::Bool=true, kwargs...) where T
    check_dims(size(x), size(dy), pdims)
    dxp = fill!(similar(dx, padded_size(pdims)..., size(dx)[end-1:end]...), 0)
    grad = T(alpha) .* dy ./ window_counts(dy, pdims, count_include_pad)
    for κ in kernel_positions(pdims)
        w = window(dxp, pdims, κ)
        w .+= grad
    end
    add_input_gradient!(dx, dxp, pdims, beta)
    return dx
end
