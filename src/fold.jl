
"""
    unfold(x, kernel_size; stride = 1, pad = 0, dilation = 0)

Places sliding windows of x into a container tensor of size (num_windows, window_size, batchsize).
The window size is determined by the prod(spatial dims of kernel)*input_channels.
The number of sliding windows will match those of convolution (conv) with the same kernel_size and arguments.
"""
function unfold(x::AbstractArray{T, N}, kernel_size::NTuple{K}; stride = 1, pad = 0, dilation = 1) where {T, K, N}
    stride = expand(Val(N - 2), stride)
    padding = expand(Val(N - 2), pad)
    dilation = expand(Val(N - 2), dilation)
    cdims = DenseConvDims(size(x), kernel_size; stride, padding, dilation)
    return unfold(x, cdims)
end

"""
    fold(y, output_size, kernel_size; stride = 1, pad = 0, dilation = 0)

Accumulates sliding windows from the output of unfold into a container tensor of size `output_size`.
An inverse to `unfold` can be obtained by using `fold` and accounting for scaling issues. 
For example,

```jldoctest
julia> kernel_size, pad = (3, 3, 1, 1), 1

julia> x = reshape(1:64, 8, 8, 1, 1) |> collect;

julia> y = unfold(x, kernel_size; pad=pad);

julia> size(y)
(64, 9, 1)

julia> z = fold(y, size(x), kernel_size; pad=pad);

julia> d = fold(unfold(ones(eltype(x), size(x)...), kernel_size; pad=pad), size(x), kernel_size; pad=pad)
8×8×1×1 Array{Int64, 4}:
[:, :, 1, 1] =
 4  6  6  6  6  6  6  4
 6  9  9  9  9  9  9  6
 6  9  9  9  9  9  9  6
 6  9  9  9  9  9  9  6
 6  9  9  9  9  9  9  6
 6  9  9  9  9  9  9  6
 6  9  9  9  9  9  9  6
 4  6  6  6  6  6  6  4

julia> x == z./d
true
```
"""
function fold(x::AbstractArray{T, 3}, output_size::NTuple{N}, kernel_size::NTuple{K}; stride = 1, pad = 0, dilation = 1) where {T, K, N}
    stride = expand(Val(N - 2), stride)
    padding = expand(Val(N - 2), pad)
    dilation = expand(Val(N - 2), dilation)
    cdims = DenseConvDims(output_size, kernel_size; stride, padding, dilation)
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

# reverse diff rules
function rrule(::typeof(unfold), x, cdims)
    function unfold_pullback(Δ)
        return (
            NoTangent(),
            @thunk(fold(unthunk(Δ), size(x), cdims)),
            NoTangent(),
        )
    end
    return unfold(x, cdims), unfold_pullback
end

function rrule(::typeof(fold), x, output_size, cdims)
    function fold_pullback(Δ)
        return (
            NoTangent(),
            @thunk(unfold(unthunk(Δ), cdims)),
            NoTangent(),
            NoTangent(),
        )
    end
    return fold(x, output_size, cdims), fold_pullback
end

