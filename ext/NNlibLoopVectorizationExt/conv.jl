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

function NNlib.conv!(output::Array{T,4}, input::Array{T,4}, weight::Array{T,4}, cdims::ConvDims) where {T<:Real}

    # fix for groupcount > 1 (NNlib.check_dims would throw an error otherwise)
    size_weight_check_dims = (size(weight)[1:2]..., size(weight)[3]*cdims.groupcount, size(weight)[4])
    cdims_check_dims = DenseConvDims(size(input), size_weight_check_dims, stride=cdims.stride, padding=cdims.padding, dilation=cdims.dilation, groups=1, flipkernel=cdims.flipkernel)
    NNlib.check_dims(size(input), size_weight_check_dims, size(output), cdims_check_dims)

    # padding is done naively at the moment
    if cdims.padding != (0, 0, 0, 0)
        input = NNlib.pad_zeros(input, cdims.padding, dims=(1, 2))
    end

    output_width, output_height, _ = size(output)
    input_width, input_height, in_channels, batch_size = size(input)
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

        @tturbo for index_batch in 1:batch_size
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                value = zero(T)
                for in_channel in static(1):static(in_channels), y_w in static(1):static(weight_height), x_w in static(1):static(weight_width)
                # for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    value += input[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value
            end
        end

    elseif groups == 1 && cdims.dilation == (1, 1) # second specialized case for better performance
        # println("forward: second specialized case for better performance")

        @tturbo for index_batch in 1:batch_size
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                value = zero(T)
                # for in_channel in static(1):static(in_channels), y_w in static(1):static(weight_height), x_w in static(1):static(weight_width)
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    value += input[n + x_w - 1, m + y_w - 1, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value
            end
        end

    elseif groups == 1 # third specialized case for better performance
        # println("forward: third specialized case for better performance")

        @tturbo for index_batch in 1:batch_size
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                value = zero(T)
                # for in_channel in static(1):static(in_channels), y_w in static(1):static(weight_height), x_w in static(1):static(weight_width)
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    value += input[n + (x_w - 1) * x_dilation, m + (y_w - 1) * y_dilation, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value
            end
        end

    else # general case for any convolution
        # println("forward: general case for any convolution")

        @tturbo for index_batch in 1:batch_size
            for group in 1:groups, out_channel_per_group in 1:out_channels_per_group, y_out in 1:output_height, x_out in 1:output_width
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                value = zero(T)
                # for in_channel_weight in static(1):static(in_channels_weight), y_w in static(1):static(weight_height), x_w in static(1):static(weight_width)
                for in_channel_weight in 1:in_channels_weight, y_w in 1:weight_height, x_w in 1:weight_width
                    value += input[n + (x_w - 1) * x_dilation, m + (y_w - 1) * y_dilation, in_channel_weight + (group - 1) * in_channels_weight, index_batch] * weight[x_w, y_w, in_channel_weight, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value
            end
        end

    end

    return output
end

function ∇conv_data_im2col_grouped!(input_gradient::Array{T,4}, output_gradient::Array{T,4}, weight::Array{T,4}, cdims::ConvDims) where {T<:Real}

    ∇conv_data!(
        NNlib.insert_singleton_spatial_dimension(input_gradient, 1),
        NNlib.insert_singleton_spatial_dimension(output_gradient, 1),
        NNlib.insert_singleton_spatial_dimension(weight, 1),
        NNlib.insert_singleton_spatial_dimension(cdims, 1)
    )

    return input_gradient
end

function NNlib.∇conv_data!(input_gradient::Array{T,4}, output_gradient::Array{T,4}, weight::Array{T,4}, cdims::ConvDims) where {T<:Real}

    # fix for groupcount > 1 (NNlib.check_dims would throw an error otherwise)
    size_weight_check_dims = (size(weight)[1:2]..., size(weight)[3]*cdims.groupcount, size(weight)[4])
    cdims_check_dims = DenseConvDims(size(input_gradient), size_weight_check_dims, stride=cdims.stride, padding=cdims.padding, dilation=cdims.dilation, groups=1, flipkernel=cdims.flipkernel)
    NNlib.check_dims(size(input_gradient), size_weight_check_dims, size(output_gradient), cdims_check_dims)

    if cdims.groupcount == 1 && cdims.stride == (1, 1) && cdims.dilation == (1, 1) # very specialized case for maximum performance
        # println("backward: very specialized case for maximum performance")

        # storing all the necessary shapes
        output_width, output_height, out_channels, batch_size = size(output_gradient)
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
        input_width, input_height, in_channels, batch_size = size(input_gradient_padded) # size after padding

        # it's necessary to flip the kernel if real convolution is performed (flipkernel=false)
        if !NNlib.flipkernel(cdims)
            weight = reverse(weight, dims=(1, 2))
        end

        output_gradient = OffsetArray(output_gradient, OffsetArrays.Origin(0, 0, 0, 0))
        input_gradient_padded = OffsetArray(input_gradient_padded, OffsetArrays.Origin(0, 0, 0, 0))
        weight = OffsetArray(weight, OffsetArrays.Origin(0, 0, 0, 0))

        input_width, input_height, in_channels, batch_size = size(input_gradient_padded)
        weight_width, weight_height, in_channels_weight, out_channels = size(weight)

        @tturbo for index_batch in 0:batch_size-1
            for x_in in 0:input_width-1, y_in in 0:input_height-1, in_channel in 0:in_channels-1

                value = zero(T)
                for x_w in static(0):static(weight_width-1), y_w in static(0):static(weight_height-1), out_channel in static(0):static(out_channels-1)
                # for x_w in 0:weight_width-1, y_w in 0:weight_height-1, out_channel in 0:out_channels-1

                    is_in_bound_x = (x_in - x_w >= 0) & (x_in - x_w < output_width)
                    is_in_bound_y = (y_in - y_w >= 0) & (y_in - y_w < output_height)
                    output_gradient_value = (is_in_bound_x & is_in_bound_y) ? output_gradient[x_in - x_w, y_in - y_w, out_channel, index_batch] : zero(T)
                    value += weight[x_w, y_w, in_channel, out_channel] * output_gradient_value
                    
                end
                input_gradient_padded[x_in, y_in, in_channel, index_batch] = value

            end
        end

        input_gradient_padded = input_gradient_padded.parent

        # depad 
        if cdims.padding != (0, 0, 0, 0)
            x_pad1, x_pad2, y_pad1, y_pad2 = cdims.padding
            input_gradient .= input_gradient_padded[x_pad1+1:input_width-x_pad2, y_pad1+1:input_height-y_pad2, :, :]
        end

    else # general case for any convolution 
        input_gradient = ∇conv_data_im2col_grouped!(input_gradient, output_gradient, weight, cdims)
    end        

    return input_gradient
end