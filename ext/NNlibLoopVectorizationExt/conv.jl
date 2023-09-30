#=
Accelerated convolution for 2d-images using the power of LoopVectorization.
The acceleration is usually greatest when the inputs have a large spatial size and few channels. 
Using stride > 1, dilation > 1 or groups > 1 can slow down things a bit.

Since the current state of LoopVectorization ∇conv_filter! isn't really faster than the 
original implementation in some situations, it is left out for the moment.

Implementation for forward pass mostly copied from here (Jonas Steinebach, MIT license):
https://github.com/jonas208/GradValley.jl/blob/main/src/functional/gv_convolution.jl

Implementation for backward pass mostly copied from here (Chris Elrod, MIT license):
https://github.com/PumasAI/SimpleChains.jl/blob/main/src/conv.jl
=#

function NNlib.conv!(output::Array{T,4}, input::Array{T,4}, weight::Array{T,4}, cdims::ConvDims; kw...) where {T<:Real}

    # fix for groupcount > 1 (NNlib.check_dims would throw an error otherwise)
    size_weight_check_dims = (size(weight)[1:2]..., size(weight)[3]*cdims.groupcount, size(weight)[4])
    cdims_check_dims = DenseConvDims(size(input), size_weight_check_dims, stride=cdims.stride, padding=cdims.padding, dilation=cdims.dilation, groups=1, flipkernel=cdims.flipkernel)
    NNlib.check_dims(size(input), size_weight_check_dims, size(output), cdims_check_dims)

    # padding is done naively at the moment
    if cdims.padding != (0, 0, 0, 0)
        input = NNlib.pad_zeros(input, cdims.padding, dims=(1, 2))
    end

    output_width, output_height, _ = size(output)
    input_width, input_height, in_channels, batches = size(input)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)

    # it's necessary to flip the kernel if real convolution is performed (flipkernel=false)
    if !NNlib.flipkernel(cdims)
        weight = reverse(weight, dims=(1, 2))
    end

    groups = cdims.groupcount
    x_stride, y_stride = cdims.stride
    x_dilation, y_dilation = cdims.dilation
    out_channels_per_group = out_channels ÷ groups

    if cdims.groupcount == 1 && cdims.stride == (1, 1) && cdims.dilation == (1, 1) # very specialized case for maximum performance
        # println("forward: very specialized case for maximum performance")

        @tturbo for index_batch in 1:batches
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                value = zero(T)
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    value += input[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value
            end
        end

    elseif groups == 1 # second specialized case for better performance
        # println("forward: second specialized case for better performance")

        @tturbo for index_batch in 1:batches
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                value = zero(T)
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    value += input[x_in, y_in, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value
            end
        end

    else # general case for any convolution
        # println("forward: general case for any convolution")

        @tturbo for index_batch in 1:batches
            for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                value = zero(T)
                for in_channel_weight in 1:in_channels_weight, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    in_channel_input = in_channel_weight + (group - 1) * in_channels_weight
                    value += input[x_in, y_in, in_channel_input, index_batch] * weight[x_w, y_w, in_channel_weight, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value
            end
        end

    end

    return output
end

function NNlib.∇conv_data!(input_gradient::Array{T,4}, output_gradient::Array{T,4}, weight::Array{T,4}, cdims::ConvDims; kw...) where {T<:Real}
    
    # fix for groupcount > 1 (NNlib.check_dims would throw an error otherwise)
    size_weight_check_dims = (size(weight)[1:2]..., size(weight)[3]*cdims.groupcount, size(weight)[4])
    cdims_check_dims = DenseConvDims(size(input_gradient), size_weight_check_dims, stride=cdims.stride, padding=cdims.padding, dilation=cdims.dilation, groups=1, flipkernel=cdims.flipkernel)
    NNlib.check_dims(size(input_gradient), size_weight_check_dims, size(output_gradient), cdims_check_dims)
    
    # storing all the necessary shapes
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)

    # because in the actual computation section, values are added, it's saver to reset the given input_gradient first
    input_gradient .= zero(T)
    # check if input_gradient must be padded (padding is done naively at the moment)
    if cdims.padding != (0, 0, 0, 0)
        input_gradient_padded = NNlib.pad_zeros(input_gradient, cdims.padding, dims=(1, 2))
    else
        input_gradient_padded = input_gradient
    end

    # store the size of input after padding 
    input_width, input_height, in_channels, current_batch_size = size(input_gradient_padded) # size after padding

    # it's necessary to flip the kernel if real convolution is performed (flipkernel=false)
    if !NNlib.flipkernel(cdims)
        weight = reverse(weight, dims=(1, 2))
    end

    groups = cdims.groupcount
    x_stride, y_stride = cdims.stride
    x_dilation, y_dilation = cdims.dilation
    out_channels_per_group = out_channels ÷ groups

    @inline static_size(x::AbstractArray{T, N}) where {T, N} = static.(size(x))

    # actual computation (using @tturbo instead of Threads.@threads + @turbo may end up in wrong results)
    if groups == 1 && cdims.stride == (1, 1) && cdims.dilation == (1, 1) # very specialized case for maximum performance
        # println("backward: very specialized case for maximum performance")

        output_gradient = OffsetArray(output_gradient, OffsetArrays.Origin(0, 0, 0, 0))
        input_gradient_padded = OffsetArray(input_gradient_padded, OffsetArrays.Origin(0, 0, 0, 0))
        weight = OffsetArray(weight, OffsetArrays.Origin(0, 0, 0, 0))

        input_width, input_height, in_channels, batch_size = static_size(input_gradient_padded)
        weight_width, weight_height, in_channels_weight, out_channels = static_size(weight)

        y_upper_bound = static(output_height) # input_width - weight_width + static(1)
        x_upper_bound = static(output_width) # input_height - weight_height + static(1)

        @tturbo for index_batch in 0:batch_size-1
            for x_in in 0:input_width-1, y_in in 0:input_height-1, in_channel in 0:in_channels-1 # @tturbo unroll = (2, 1) 

                value = zero(T)
                for x_w in 0:weight_width-1, y_w in 0:weight_height-1, out_channel in 0:out_channels-1
                    ib0 = (x_in - x_w >= 0) & (x_in - x_w < x_upper_bound)
                    ib1 = (y_in - y_w >= 0) & (y_in - y_w < y_upper_bound)
                    output_gradient_value = (ib0 & ib1) ? output_gradient[x_in-x_w, y_in-y_w, out_channel, index_batch] : zero(T)
                    value += weight[x_w, y_w, in_channel, out_channel] * output_gradient_value
                    # value += (ib0 & ib1) ? output_gradient[x_in-x_w, y_in-y_w, out_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel] : zero(T)
                end
                input_gradient_padded[x_in, y_in, in_channel, index_batch] = value

            end
        end

        input_gradient_padded = input_gradient_padded.parent

    elseif groups == 1 && cdims.dilation == (1, 1) # second specialized case for better performance
        # println("backward: second specialized case for better performance")

        #=
        Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    input_gradient_padded[x_in, y_in, in_channel, index_batch] += weight[x_w, y_w, in_channel, out_channel] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
            end
        end
        =#

        y_out_indices = Array{Int, 3}(undef, weight_width, weight_height, input_height)
        x_out_indices = Array{Int, 3}(undef, weight_width, weight_height, input_width)
        x_out_indices .= -1
        y_out_indices .= -1

        @turbo for y_out in 1:output_height, x_out in 1:output_width # 
            m = y_out + (y_stride - 1) * (y_out - 1)
            n = x_out + (x_stride - 1) * (x_out - 1)
            for y_w in 1:weight_height, x_w in 1:weight_width
                y_in = m + (y_w - 1) * y_dilation
                x_in = n + (x_w - 1) * x_dilation
                y_out_indices[x_w, y_w, y_in] = y_out 
                x_out_indices[x_w, y_w, x_in] = x_out
            end
        end

        @tturbo for index_batch in 1:current_batch_size
            for x_in in 1:input_width, y_in in 1:input_height, in_channel in 1:in_channels # @tturbo unroll = (2, 1) 

                value = zero(T)
                for x_w in 1:weight_width, y_w in 1:weight_height, out_channel in 1:out_channels

                    x_out = x_out_indices[x_w, y_w, x_in]
                    y_out = y_out_indices[x_w, y_w, y_in]

                    ib0 = x_out > -1 # !=
                    ib1 = y_out > -1 # !=

                    output_gradient_value = (ib0 & ib1) ? output_gradient[x_out, y_out, out_channel, index_batch] : zero(T)
                    # output_gradient_value = T(2.0) # output_gradient[x_out, y_out, out_channel, index_batch]
                    value += weight[x_w, y_w, in_channel, out_channel] * output_gradient_value
                end
                input_gradient[x_in, y_in, in_channel, index_batch] = value

            end
        end

    else # general case for any convolution 
        # println("backward: general case for any convolution")

        Threads.@threads for index_batch in 1:current_batch_size
            for out_channel_per_group in 1:out_channels_per_group # putting @turbo here may end up in wrong results
                @turbo for group in 1:groups, y_out in 1:output_height, x_out in 1:output_width
                    m = y_out + (y_stride - 1) * (y_out - 1)
                    n = x_out + (x_stride - 1) * (x_out - 1)
                    out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                    for in_channel_weight in 1:in_channels_weight, y_w in 1:weight_height, x_w in 1:weight_width
                        y_in = m + (y_w - 1) * y_dilation
                        x_in = n + (x_w - 1) * x_dilation
                        in_channel_input = in_channel_weight + (group - 1) * in_channels_weight
                        input_gradient_padded[x_in, y_in, in_channel_input, index_batch] += weight[x_w, y_w, in_channel_weight, out_channel] * output_gradient[x_out, y_out, out_channel, index_batch]
                    end
                end
            end
        end

    end

    # depad 
    if cdims.padding != (0, 0, 0, 0)
        x_pad1, x_pad2, y_pad1, y_pad2 = cdims.padding
        input_gradient .= input_gradient_padded[x_pad1+1:input_width-x_pad2, y_pad1+1:input_height-y_pad2, :, :]
    end

    return input_gradient
end