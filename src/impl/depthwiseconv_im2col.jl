## This file contains adapter code for doing depthwise convolutions with im2col.


"""
    depthwiseconv_im2col!(y, x, w, cdims, col=similar(x); alpha=1, beta=0)

Perform a depthwise convolution using im2col and GEMM, store the result in `y`.
See [`conv_im2col!`](@ref) for explanation of optional parameters.
"""
depthwiseconv_im2col!

function depthwiseconv_im2col!(
                y::AbstractArray{T,5}, x::AbstractArray{T,5},
                w::AbstractArray{T,5}, cdims::DepthwiseConvDims;
                col::AbstractArray{T,3} = similar(x, im2col_dims(cdims)),
                alpha::T=T(1), beta::T=T(0),
                ntasks::Int=nthreads()) where T
    check_dims(size(x), size(w), size(y), cdims)

    # This functions exactly the same as conv_im2col!(), except that we shard the
    # incoming data into slices of single channels.  This means that we need to walk
    # each pointer forward individually, as done below, taking a single input channel
    # and combining it with each kernel individually, before walking forward and doing
    # the next input channel.
    M = prod(output_size(cdims))
    N = channel_multiplier(cdims)
    K = prod(kernel_size(cdims))

    parts = Iterators.partition(axes(y)[end], ceil(Int, size(y, 5) / ntasks))

    dcdims = DenseConvDims(cdims)

    function depthwiseconv_part(task_n, part)
        col_slice = col_slice = view(col, :, :, task_n) # col_slice is a task-local workspace
        for batch_idx in part
            im2col!(col_slice, view(x, :, :, :, :, batch_idx), dcdims)

            # We do a separate convolution for each channel in x, as we must
            for c_in in 1:channels_in(cdims)
                # Walk each pointer forward as we process each input channel
                GC.@preserve col_slice w y begin
                    col_ptr = pointer(col_slice, (c_in-1)*M*K+1)
                    w_ptr = pointer(w, (c_in-1)*K*N+1)
                    y_ptr = pointer(y, ((batch_idx - 1)*channels_in(cdims) + c_in - 1)*M*N + 1)
                    gemm!(Val(false), Val(false), M, N, K, alpha, col_ptr, w_ptr, beta, y_ptr)
                end
            end
        end
    end
    if should_use_spawn() && length(parts) > 1
        @sync for (task_n, part) in enumerate(parts)
            Threads.@spawn depthwiseconv_part(task_n, part)
        end
    else
        for (task_n, part) in enumerate(parts)
            depthwiseconv_part(task_n, part)
        end
    end
    return y
end

"""
    ∇depthwiseconv_filter_im2col!(dw, w, dy, cdims, col=similar(dw, ∇filter_im2col_dims(cdims));
                                  alpha=1, beta=0)

Depthwise conv backward pass onto the weights using im2col and GEMM.
See [`conv_im2col!`](@ref) for explanation of optional parameters.
"""
∇depthwiseconv_filter_im2col!

function ∇depthwiseconv_filter_im2col!(
                dw::AbstractArray{T,5}, x::AbstractArray{T,5},
                dy::AbstractArray{T,5}, cdims::DepthwiseConvDims;
                col::AbstractArray{T,3} = similar(dw, ∇filter_im2col_dims(cdims)),
                alpha::T=T(1), beta::T=T(0)) where T
    check_dims(size(x), size(dw), size(dy), cdims)

    M = prod(kernel_size(cdims))
    N = channel_multiplier(cdims)
    K = prod(output_size(cdims))

    for batch_idx in 1:size(x, 5)
        # Because we accumulate over batches in this loop, we must set `beta` equal
        # to `1.0` after the first sample.
        beta′ = batch_idx == 1 ? beta : T(1)

        # col_slice is a thread-local workspace
        col_slice = view(col, :, :, 1)
        im2col!(col_slice, view(x, :, :, :, :, batch_idx), cdims)

        # We do a separate convolution for each channel in x, as we must
        for c_in in 1:channels_in(cdims)
            # Walk each pointer forward as we process each input channel
            GC.@preserve col_slice dw dy begin
                col_ptr = pointer(col_slice, (c_in - 1)*M*K + 1)
                dy_ptr = pointer(dy, (batch_idx - 1)*N*K*channels_in(cdims) + (c_in - 1)*K*N + 1)
                dw_ptr = pointer(dw, (c_in - 1)*M*N + 1)
                gemm!(Val(true), Val(false), M, N, K, alpha, col_ptr, dy_ptr, beta′, dw_ptr)
            end
        end
    end
    return dw
end

"""
    ∇depthwiseconv_data_im2col!(dx, w, dy, cdims, col=similar(dx); alpha=1, beta=0)

Depwthwise conv2d backward pass onto the input using im2col and GEMM.
See [`conv_im2col!`](@ref) for explanation of optional parameters.
"""
∇depthwiseconv_data_im2col!

function ∇depthwiseconv_data_im2col!(
                dx::AbstractArray{T,5}, dy::AbstractArray{T,5},
                w::AbstractArray{T,5}, cdims::DepthwiseConvDims;
                col::AbstractArray{T,3} = similar(dx, im2col_dims(cdims)),
                alpha::T=T(1), beta::T=T(0),
                ntasks::Int=nthreads()) where T
    check_dims(size(dx), size(w), size(dy), cdims)

    M = prod(output_size(cdims))
    N = prod(kernel_size(cdims))
    K = channel_multiplier(cdims)

    parts = Iterators.partition(axes(dx)[end], ceil(Int, size(dx, 5) / ntasks))

    function ∇depthwiseconv_data_part(task_n, part)
        col_slice = col_slice = view(col, :, :, task_n) # col_slice is a task-local workspace
        for batch_idx in part
            # We do a separate convolution for each channel in x, as we must
            for cidx in 1:channels_in(cdims)
                GC.@preserve col_slice w dy begin
                    # Walk each pointer forward as we process each input channel
                    dy_ptr = pointer(dy, (batch_idx - 1)*M*K*channels_in(cdims)+(cidx - 1)*K*M + 1)
                    w_ptr = pointer(w, (cidx - 1)*K*N + 1)
                    col_ptr = pointer(col_slice, (cidx - 1)*M*N + 1)
                    gemm!(Val(false), Val(true), M, N, K, alpha, dy_ptr, w_ptr, T(0), col_ptr)
                end
            end
            col2im!(view(dx, :, :, :, :, batch_idx), col_slice, cdims, beta)
        end
    end
    if should_use_spawn() && length(parts) > 1
        @sync for (task_n, part) in enumerate(parts)
            Threads.@spawn ∇depthwiseconv_data_part(task_n, part)
        end
    else
        for (task_n, part) in enumerate(parts)
            ∇depthwiseconv_data_part(task_n, part)
        end
    end
    return dx
end
