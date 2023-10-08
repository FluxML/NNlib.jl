#=
Accelerated mean pooling for 2d-images using the power of LoopVectorization.
The speed up is usually lower compared to conv but can be approximately up to 2x.

Since the current state of LoopVectorization ∇meanpool! isn't really faster than the 
original implementation in some situations, it is left out for the moment.

Implementation inspired from here (Jonas Steinebach, MIT license):
https://github.com/jonas208/GradValley.jl/blob/main/src/functional/gv_pooling.jl
=#

function NNlib.meanpool!(output::Array{T,4}, input::Array{T,4}, pdims::PoolDims; kw...) where {T<:Real}
    NNlib.check_dims(size(input), size(output), pdims)
    
    # storing all the necessary shapes
    input_width, input_height, channels, batch_size = size(input)
    output_width, output_height, channels, batch_size = size(output)
    kernel_width, kernel_height = pdims.kernel_size

    x_stride, y_stride = pdims.stride
    x_dilation, y_dilation = pdims.dilation
    x_pad1, x_pad2, y_pad1, y_pad2 = pdims.padding

    # A helper function to project from output (w, h) to input (input_w, input_h)
    @inline project(idx, stride, pad) = (idx - 1) * stride - pad + 1

    # We use calc_padding_regions to split outselves up into separate regions that may or
    # may not need to worry about padding:
    pdims_3d = PoolDims((input_width, input_height, 1, channels, batch_size), (kernel_width, kernel_height, 1), stride=(x_stride, y_stride, 1), padding=(x_pad1, x_pad2, y_pad1, y_pad2, 0, 0), dilation=(x_dilation, y_dilation, 1))
    # println(pdims_3d.padding)
    padded_regions, central_region = NNlib.calc_padding_regions(pdims_3d)

    # We represent division by kernel size by rolling it
    # into the `alpha` multiplier. 
    _alpha = T(1 / prod(pdims.kernel_size))

    # Start with the central region
    w_region, h_region, _ = central_region

    if pdims.stride == (1, 1) && pdims.dilation == (1, 1) # specialized case for better performance
        # println("specialized case for better performance")

        @tturbo for index_batch in 1:batch_size
            # compute pooling for each channel separatly
            for channel in 1:channels, y_out in h_region, x_out in w_region
                kernel_sum = zero(T)
                for y_w in 1:kernel_height, x_w in 1:kernel_width
                    kernel_sum += input[x_out + x_w - 1 - x_pad1, y_out + y_w - 1 - y_pad1, channel, index_batch]
                end
                output[x_out, y_out, channel, index_batch] = kernel_sum * _alpha
            end
        end

    else # general case for any meanpooling
        # println("general case for any meanpooling")

        @tturbo for index_batch in 1:batch_size
            # compute pooling for each channel separatly
            for channel in 1:channels, y_out in h_region, x_out in w_region
                m = y_out + (y_stride - 1) * (y_out - 1) - y_pad1
                n = x_out + (x_stride - 1) * (x_out - 1) - x_pad1
                kernel_sum = zero(T)
                for y_w in 1:kernel_height, x_w in 1:kernel_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    kernel_sum += input[x_in, y_in, channel, index_batch]
                end
                output[x_out, y_out, channel, index_batch] = kernel_sum * _alpha
            end
        end

    end

    # Next, the padded regions
    if pdims.padding != (0, 0, 0, 0)
        @inbounds for (w_region, h_region, d_region) in padded_regions
            for index_batch in 1:batch_size, channel in 1:channels
                for z_out in d_region # for skipping the d_regions
                for y_out in h_region
                m = project(y_out, y_stride, y_pad1)
                for x_out in w_region
                n = project(x_out, x_stride, x_pad1)
                kernel_sum = zero(T)

                    for y_w in 1:kernel_height
                        y_in = m + (y_w - 1) * y_dilation
                        if y_in <= 0 || y_in > input_height
                            continue
                        end

                        for x_w in 1:kernel_width
                            x_in = n + (x_w - 1) * x_dilation
                            if x_in <= 0 || x_in > input_width
                                continue
                            end

                            kernel_sum += input[x_in, y_in, channel, index_batch]
                        end
                    end

                output[x_out, y_out, channel, index_batch] = _alpha * kernel_sum
                end
                end
                end
            end
        end
    end

    return output
end