"""
    unfold(x, kernel_size; stride = 1, pad = 0, dilation = 0, flipped = true)

Places sliding windows of x into a container tensor of size `(num_windows,
window_size, batchsize)`. The window size is determined by the `prod(spatial dims
of kernel)*input_channels`. The number of sliding windows will match those of
convolution (`conv`) with the same kernel_size and arguments. Note that
by default `conv` flips the spatial dimensions of its kernel (default
`flipped=false`), whereas `unfold` does not (default `flipped=true`).
Uses `NNlib.im2col!` as backend.

See also [`fold`](@ref), the adjoint/transpose operator
and a potential inverse of `unfold`.

# Example
The below example demonstrates that `unfold` uses the same sliding windows as `conv`.
In general [`batched_mul`](@ref) + `unfold` should not be used to achieve convolution.
```jldoctest
julia> x = reshape([100 2 3 40 5 6 700], 7, 1, 1);  # 1D data, 1 channel, batch of 1

julia> w = reshape([1 0 -1], 3, 1, 1);  # 1D conv kernel of length 3

julia> kws = (pad=1, stride=2, flipped=true);  # use same args for conv and unfold

julia> z = NNlib.unfold(x, size(w); kws...)
4×3×1 Array{Int64, 3}:
[:, :, 1] =
  0  100   2
  2    3  40
 40    5   6
  6  700   0

julia> y1 = conv(x, w; kws...)
4×1×1 Array{Int64, 3}:
[:, :, 1] =
  -2
 -38
  34
   6

julia> y2 = z ⊠ w  # ⊠ (\\boxtimes) is NNlib.batched_mul
4×1×1 Array{Int64, 3}:
[:, :, 1] =
  -2
 -38
  34
   6
```
"""
function unfold(x::AbstractArray{T, N}, kernel_size::NTuple{K}; stride = 1, pad = 0, dilation = 1, flipped = true) where {T, K, N}
    stride = expand(Val(N - 2), stride)
    padding = expand(Val(N - 2), pad)
    dilation = expand(Val(N - 2), dilation)
    cdims = DenseConvDims(size(x), kernel_size; stride, padding, dilation, flipkernel=flipped)
    return unfold(x, cdims)
end

"""
    fold(y, output_size, kernel_size; stride = 1, pad = 0, dilation = 0, flipped = true)

The adjoint/transpose operator of `unfold`. It accumulates sliding windows from
the output of `unfold` into a container tensor of size `output_size`. An inverse
to `unfold` may be obtained (in some cases) by using `fold` and accounting for scaling issues
with a divisor (see example). Uses `NNlib.col2im!` as backend.

See also [`unfold`](@ref).

# Example
```jldoctest
julia> x = reshape([100 2 3 40 5 6 700], 7, 1, 1);  # 1D data, 1 channel, batch of 1

julia> y = NNlib.unfold(x, (3,1,1))  # sliding window of size 3
5×3×1 Array{Int64, 3}:
[:, :, 1] =
 100   2    3
   2   3   40
   3  40    5
  40   5    6
   5   6  700

julia> z = NNlib.fold(y, size(x), (3,1,1))  # sum of contributions in y. 100 appears once, 40 three times
7×1×1 Array{Int64, 3}:
[:, :, 1] =
 100
   4
   9
 120
  15
  12
 700

julia> divisor = NNlib.fold(NNlib.unfold(ones(size(x)...), (3,1,1)), size(x), (3,1,1))
7×1×1 Array{Float64, 3}:
[:, :, 1] =
 1.0
 2.0
 3.0
 3.0
 3.0
 2.0
 1.0

julia> z ./ divisor
7×1×1 Array{Float64, 3}:
[:, :, 1] =
 100.0
   2.0
   3.0
  40.0
   5.0
   6.0
 700.0
```
In general, an inverse to `unfold` does not exist if `divisor` contains zeros.
"""
function fold(x::AbstractArray{T, 3}, output_size::NTuple{N}, kernel_size::NTuple{K}; stride = 1, pad = 0, dilation = 1, flipped = true) where {T, K, N}
    stride = expand(Val(N - 2), stride)
    padding = expand(Val(N - 2), pad)
    dilation = expand(Val(N - 2), dilation)
    cdims = DenseConvDims(output_size, kernel_size; stride, padding, dilation, flipkernel=flipped)
    return fold(x, output_size, cdims)
end

# im2col_dims returns (numblocks, blocksize, threadnum) where thread dim is used as thread-local
# workspace for multithreaded conv. Ultimately, we want to threadnum with batchsize.
unfold_dims(cdims::DenseConvDims) = im2col_dims(cdims)[1:2]

# auto-allocating versions
function unfold(x::AbstractArray{T, N}, cdims::DenseConvDims) where {T, N}
    y = similar(x, unfold_dims(cdims)..., size(x, N)) # (numblocks, blocksize, batchsize)
    return unfold!(y, x, cdims)
end

function fold(y::AbstractArray{T, 3}, output_size::NTuple, cdims::DenseConvDims) where {T}
    x = similar(y, output_size)
    return fold!(x, y, cdims)
end

# N < 5 -dimension in-place versions
function unfold!(y::AbstractArray{yT, 3}, x::AbstractArray{xT, N}, cdims::DenseConvDims) where {yT, xT, N}
    unfold!(
        y,
        insert_singleton_spatial_dimension(x, 5-N),
        insert_singleton_spatial_dimension(cdims, 5-N),
    )
    return y
end

function fold!(x::AbstractArray{xT, N}, y::AbstractArray{yT, 3}, cdims::DenseConvDims) where {yT, xT, N}
    fold!(
        insert_singleton_spatial_dimension(x, 5-N),
        y,
        insert_singleton_spatial_dimension(cdims, 5-N),
    )
    return x
end

# 5-dimension in-place versions
function unfold!(y::AbstractArray{yT, 3}, x::AbstractArray{xT, 5}, cdims::DenseConvDims) where {yT, xT}
    @threads for batch_idx in 1:size(x, 5)
        y_slice = view(y, :, :, batch_idx)
        im2col!(y_slice, view(x, :, :, :, :, batch_idx), cdims)
    end
    return y
end

function fold!(x::AbstractArray{xT, 5}, y::AbstractArray{yT, 3}, cdims::DenseConvDims) where {xT, yT}
    @threads for batch_idx in 1:size(x, 5)
        y_slice = view(y, :, :, batch_idx)
        col2im!(view(x, :, :, :, :, batch_idx), y_slice, cdims)
    end
    return x
end

@kernel function unfold_kernel!(
    col::AbstractArray{T}, x, col_size,
    input_size, output_size, kernel_size,
    flipkernel, stride, pad_lo, dilation, max_idx,
) where T
    index = @index(Global)

    @inbounds if index ≤ max_idx
        i, kw, kh, kd, c, b = CartesianIndices(col_size)[index].I # col indices
        w, h, d = CartesianIndices(output_size)[i].I # x indices

        # project
        w, h, d = @. ((w, h, d) - 1) * stride - pad_lo + 1 + ((kw, kh, kd) - 1) * dilation

        if !flipkernel
            kw, kh, kd = kernel_size .- (kw, kh, kd) .+ 1
        end

        # check out of bounds
        if !all(checkindex.(Bool, UnitRange.(1, input_size), (w, h, d)))
            col[i, kw, kh, kd, c, b] = T(0)
        else
            xval::T = x[w, h, d, c, b]
            col[i, kw, kh, kd, c, b] = xval
        end
    end
end

@kernel function fold_kernel!(
    x::AbstractArray{T}, col, col_size,
    input_size, output_size, kernel_size,
    flipkernel, stride, pad_lo, dilation, max_idx,
) where T
    index = @index(Global)

    @inbounds if index ≤ max_idx
        i, kw, kh, kd, c, b = CartesianIndices(col_size)[index].I # col indices
        w, h, d = CartesianIndices(output_size)[i].I # x indices

        # project
        w, h, d = @. ((w, h, d) - 1) * stride - pad_lo + 1 + ((kw, kh, kd) - 1) * dilation

        # check out of bounds
        if all(checkindex.(Bool, UnitRange.(1, input_size), (w, h, d)))
            if !flipkernel
                kw, kh, kd = kernel_size .- (kw, kh, kd) .+ 1
            end

            cval::T = col[i, kw, kh, kd, c, b]
            @atomic x[w, h, d, c, b] += cval
        end
    end
end

function unfold!(
    col::AnyGPUArray{cT,3}, x::AnyGPUArray{xT,5}, cdims::DenseConvDims,
) where {cT, xT}
    spatial_dims(cdims) != 3 && throw(DimensionMismatch(
        "unfold!() only accepts 3d convoluitional inputs"))

    C_in = channels_in(cdims)
    ker_size = kernel_size(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(cdims)
    pad_lo = (pad_w_lo, pad_h_lo, pad_d_lo)

    out_size = output_size(cdims)
    col_reshaped = reshape(col, (prod(out_size), ker_size..., C_in, :))

    max_idx = prod(size(col))
    unfold_kernel!(get_backend(x))(
        col_reshaped, x, size(col_reshaped),
        input_size(cdims), out_size, ker_size,
        flipkernel(cdims), stride(cdims), pad_lo, dilation(cdims), max_idx;
        ndrange=max_idx)
    return col
end

function fold!(
    x::AnyGPUArray{xT,5}, col::AnyGPUArray{cT,3}, cdims::DenseConvDims,
) where {xT, cT}
    spatial_dims(cdims) != 3 && throw(DimensionMismatch(
        "fold!() only accepts 3d convoluitional inputs"))

    # going to accumulate into x
    fill!(x, xT(0))

    C_in = channels_in(cdims)
    ker_size = kernel_size(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(cdims)
    pad_lo = (pad_w_lo, pad_h_lo, pad_d_lo)
    out_size = output_size(cdims)

    col_reshaped = reshape(col, (prod(out_size), ker_size..., C_in, :))

    max_idx = prod(size(col))
    fold_kernel!(get_backend(x))(
        x, col_reshaped, size(col_reshaped),
        input_size(cdims), out_size, ker_size,
        flipkernel(cdims), stride(cdims), pad_lo, dilation(cdims), max_idx;
        ndrange=max_idx)

    return x
end

# reverse diff rules
function rrule(::typeof(unfold), x, cdims::DenseConvDims; kw...)
    function unfold_pullback(Δ)
        return (
            NoTangent(),
            fold(unthunk(Δ), size(x), cdims; kw...),
            NoTangent(),
        )
    end
    return unfold(x, cdims; kw...), unfold_pullback
end

function rrule(::typeof(fold), x, output_size, cdims::DenseConvDims; kw...)
    function fold_pullback(Δ)
        return (
            NoTangent(),
            unfold(unthunk(Δ), cdims; kw...),
            NoTangent(),
            NoTangent(),
        )
    end
    return fold(x, output_size, cdims; kw...), fold_pullback
end

