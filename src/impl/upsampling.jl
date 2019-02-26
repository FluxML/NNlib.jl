function upsample!(y::AbstractArray{T, 5}, x::AbstractArray{T, 5},
                   udims::UpsampleDims) where {T}
    check_dims(size(x), size(y), udims)

    width, height, depth = input_size(udims)
    out_c = channels_out(udims)
    stride_w, stride_h, stride_d = stride(udims)
    out_width, out_height, out_depth = output_size(udims)
    sc = scale(udims)

    function get_linear_index_x(b, c, d, h, w)
        dims = size(x)
        return (b - 1) * prod(dims[1:(end - 1)]) + (c - 1) * prod(dims[1:(end - 2)]) +
               ((d - 1) ÷ stride_d) * prod(dims[1:(end - 3)]) + ((h - 1) ÷ stride_h) *
               prod(dims[1:(end - 4)]) + ((w - 1) ÷ stride_w) + 1
    end

    function get_linear_index_y(b, c, d, h, w)
        dims = size(y)
        return (b - 1) * prod(dims[1:(end - 1)]) + (c - 1) * prod(dims[1:(end - 2)]) +
               (d - 1) * prod(dims[1:(end - 3)]) + (h - 1) * prod(dims[1:(end - 4)]) +
               (w - 1) + 1
    end

    @inbounds for batch_idx in 1:size(x)[end],
        c in 1:out_c,
        d in 1:out_depth,
        h in 1:out_height,
        w in 1:out_width

        x_idx = get_linear_index_x(batch_idx, c, d, h, w)
        y_idx = get_linear_index_y(batch_idx, c, d, h, w)
        y[y_idx] = sc * x[x_idx]
    end

    # Return `y`
    return y
end

function ∇upsample!(dx::AbstractArray{T, 5}, dy::AbstractArray{T, 5},
                    x::AbstractArray{T, 5}, udims::UpsampleDims) where {T}
    check_dims(size(x), size(dy), udims)

    width, height, depth = input_size(udims)
    out_c = channels_out(udims)
    stride_w, stride_h, stride_d = stride(udims)
    out_width, out_height, out_depth = output_size(udims)
    sc = scale(udims)

    function get_linear_index_dx(b, c, d, h, w)
        dims = size(dx)
        return (b - 1) * prod(dims[1:(end - 1)]) + (c - 1) * prod(dims[1:(end - 2)]) +
               ((d - 1) ÷ stride_d) * prod(dims[1:(end - 3)]) + ((h - 1) ÷ stride_h) *
               prod(dims[1:(end - 4)]) + ((w - 1) ÷ stride_w) + 1
    end

    function get_linear_index_dy(b, c, d, h, w)
        dims = size(dy)
        return (b - 1) * prod(dims[1:(end - 1)]) + (c - 1) * prod(dims[1:(end - 2)]) +
               (d - 1) * prod(dims[1:(end - 3)]) + (h - 1) * prod(dims[1:(end - 4)]) +
               (w - 1) + 1
    end

    @inbounds for batch_idx in 1:size(x)[end],
        c in 1:out_c,
        d in 1:out_depth,
        h in 1:out_height,
        w in 1:out_width

        dx_idx = get_linear_index_dx(batch_idx, c, d, h, w)
        dy_idx = get_linear_index_dy(batch_idx, c, d, h, w)
        dx[dx_idx] = sc * dy[dy_idx]
    end

    # Return `dx`
    return dx
end
