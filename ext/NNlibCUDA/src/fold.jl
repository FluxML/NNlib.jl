
function unfold_kernel!(col::AbstractArray{T}, x, col_size, input_size, output_size, kernel_size, flipkernel, stride, pad_lo, dilation, max_idx) where {T}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        i, kw, kh, kd, c, b = CartesianIndices(col_size)[index].I # col indices
        w, h, d = CartesianIndices(output_size)[i].I # x indices

        # project 
        w, h, d = @. ((w, h, d) - 1)*stride - pad_lo + 1 + ((kw, kh, kd) - 1)*dilation

        if !flipkernel
            kw, kh, kd = kernel_size .- (kw, kh, kd) .+ 1
        end

        # check out of bounds
        if any((w, h, d) .<= 0 .| (w, h, d) .> input_size)
            col[i, kw, kh, kd, c, b] = T(0)
            return nothing
        end
        
        xval::T = x[w, h, d, c, b]
        col[i, kw, kh, kd, c, b] = xval
    end

    return nothing
end

function fold_kernel!(x::AbstractArray{T}, col, col_size, input_size, output_size, kernel_size, flipkernel, stride, pad_lo, dilation, max_idx) where {T}
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        i, kw, kh, kd, c, b = CartesianIndices(col_size)[index].I # col indices
        w, h, d = CartesianIndices(output_size)[i].I # x indices

        # project 
        w, h, d = @. ((w, h, d) - 1)*stride - pad_lo + 1 + ((kw, kh, kd) - 1)*dilation

        # check out of bounds
        if any((w, h, d) .<= 0 .|| (w, h, d) .> input_size)
            return nothing
        end

        if !flipkernel
            kw, kh, kd = kernel_size .- (kw, kh, kd) .+ 1
        end
        
        cval::T = col[i, kw, kh, kd, c, b]
        CUDA.@atomic x[w, h, d, c, b] += cval
    end

    return nothing
end

function NNlib.unfold!(col::AnyCuArray{cT,3}, x::AnyCuArray{xT,5}, cdims::NNlib.DenseConvDims) where {cT, xT}
    if NNlib.spatial_dims(cdims) != 3
        throw(DimensionMismatch("unfold!() only accepts 3d convoluitional inputs"))
    end

    input_size = NNlib.input_size(cdims)
    C_in = NNlib.channels_in(cdims)
    kernel_size = NNlib.kernel_size(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = NNlib.padding(cdims)
    pad_lo = (pad_w_lo, pad_h_lo, pad_d_lo)
    dilation = NNlib.dilation(cdims)
    stride = NNlib.stride(cdims)
    output_size = NNlib.output_size(cdims)
    flipkernel = NNlib.flipkernel(cdims) 

    col_reshaped = reshape(col, (prod(output_size), kernel_size..., C_in, :))

    max_idx = prod(size(col))
    args = col_reshaped, x, size(col_reshaped), input_size, output_size, kernel_size, flipkernel, stride, pad_lo, dilation, max_idx
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

    input_size = NNlib.input_size(cdims)
    C_in = NNlib.channels_in(cdims)
    kernel_size = NNlib.kernel_size(cdims)
    pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = NNlib.padding(cdims)
    pad_lo = (pad_w_lo, pad_h_lo, pad_d_lo)
    dilation = NNlib.dilation(cdims)
    stride = NNlib.stride(cdims)
    output_size = NNlib.output_size(cdims)
    flipkernel = NNlib.flipkernel(cdims) 

    col_reshaped = reshape(col, (prod(output_size), kernel_size..., C_in, :))

    max_idx = prod(size(col))
    args = x, col_reshaped, size(col_reshaped), input_size, output_size, kernel_size, flipkernel, stride, pad_lo, dilation, max_idx
    kernel = @cuda launch=false fold_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(max_idx, config.threads)
    blocks = cld(max_idx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return x
end

