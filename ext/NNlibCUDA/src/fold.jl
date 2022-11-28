
function unfold_kernel!(T::Type, col, x, cdims, max_idx)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if index > max_idx
        return nothing
    end

    # Extract those nice, compile-time constant type parameters from `cdims`.
    width, height, depth = NNlib.input_size(cdims)
    kernel_w, kernel_h, kernel_d = NNlib.kernel_size(cdims)
    C_in = NNlib.channels_in(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = NNlib.padding(cdims)
    dil_w, dil_h, dil_d = NNlib.dilation(cdims)
    stride_w, stride_h, stride_d = NNlib.stride(cdims)
    output_size = NNlib.output_size(cdims)

    I = CartesianIndices(output_size)
    w, h, d = I[index].I  # ouput spatial index indices

    # A helper function to project from output (w, h) to input (input_w, input_h)
    @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1

    @inbounds for c in 1:C_in, b in 1:size(x,5)
        for kd in 1:kernel_d, 
            kh in 1:kernel_h, 
            kw in 1:kernel_w

        input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
        input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
        input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w

        kidxs = NNlib.kernel_index(kw, kh, kd, cdims)

        out_of_bounds = (
            input_kd <= 0 || input_kd > depth ||
            input_kh <= 0 || input_kh > height ||
            input_kw <= 0 || input_kw > width
        )
        if out_of_bounds
            col[index, kidxs..., c, b] = T(0)
            continue
        end

        # Copy the data over
        xval::T = x[input_kw, input_kh, input_kd, c, b]
        col[index, kidxs..., c, b] = xval
        end
    end

    return nothing
end

function fold_kernel!(T::Type, x, col, cdims, max_idx)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    if index > max_idx
        return nothing
    end

    # Extract those nice, compile-time constant type parameters from `cdims`.
    width, height, depth = NNlib.input_size(cdims)
    kernel_w, kernel_h, kernel_d = NNlib.kernel_size(cdims)
    C_in = NNlib.channels_in(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = NNlib.padding(cdims)
    dil_w, dil_h, dil_d = NNlib.dilation(cdims)
    stride_w, stride_h, stride_d = NNlib.stride(cdims)
    output_size = NNlib.output_size(cdims)

    I = CartesianIndices(output_size)
    w, h, d = I[index].I  # ouput spatial index indices

    # A helper function to project from output (w, h) to input (input_w, input_h)
    @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1

    @inbounds for c in 1:C_in, b in 1:size(x, 5)
        for kd in 1:kernel_d,
            kh in 1:kernel_h,
            kw in 1:kernel_w

        input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
        input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
        input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w

        out_of_bounds = (
            input_kd <= 0 || input_kd > depth ||
            input_kh <= 0 || input_kh > height ||
            input_kw <= 0 || input_kw > width
        )
        if out_of_bounds
            continue
        end

        # Copy the data over
        kidxs = NNlib.kernel_index(kw, kh, kd, cdims)
        cval::T = col[index, kidxs..., c, b]
        CUDA.@atomic x[input_kw, input_kh, input_kd, c, b] += cval
        end
    end

    return nothing
end

function NNlib.unfold!(col::AnyCuArray{cT,3}, x::AnyCuArray{xT,5}, cdims::NNlib.DenseConvDims) where {cT, xT}
    if NNlib.spatial_dims(cdims) != 3
        throw(DimensionMismatch("unfold!() only accepts 3d convoluitional inputs"))
    end

    output_size = NNlib.output_size(cdims)
    kernel_w, kernel_h, kernel_d = NNlib.kernel_size(cdims)
    C_in = NNlib.channels_in(cdims)

    # Reshape col for easy access.
    col_reshaped = reshape(col, (
        prod(output_size),
        # By input patch size
        kernel_w,
        kernel_h,
        kernel_d,
        C_in,
        size(x, 5),
    ))
    
    max_idx = prod(output_size)
    args = cT, col_reshaped, x, cdims, max_idx
    kernel = @cuda launch=false unfold_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(max_idx, config.threads)
    blocks = cld(max_idx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return col
end

function NNlib.fold!(x::AnyCuArray{xT,5}, col::AnyCuArray{cT,3}, cdims::NNlib.DenseConvDims) where {xT, cT}
    if NNlib.spatial_dims(cdims) != 3
        throw(DimensionMismatch("fold!() only accepts 3d convoluitional inputs"))
    end

    # going to accumulate into x
    fill!(x, xT(0))

    output_size = NNlib.output_size(cdims)
    kernel_w, kernel_h, kernel_d = NNlib.kernel_size(cdims)
    C_in = NNlib.channels_in(cdims)

    # Reshape col for easy access.
    col_reshaped = reshape(col, (
        prod(output_size),
        # input patch size
        kernel_w,
        kernel_h,
        kernel_d,
        C_in,
        size(x, 5),
    ))

    max_idx = prod(output_size)
    args = xT, x, col_reshaped, cdims, max_idx
    kernel = @cuda launch=false fold_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(max_idx, config.threads)
    blocks = cld(max_idx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return x
end

