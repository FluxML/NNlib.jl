#=
Implementation copied from here (Jonas Steinebach, MIT):
https://github.com/jonas208/GradValley.jl/blob/main/src/functional/gv_convolution.jl
Could include bias & activation too, hence overload `conv_bias_act`,
at the cost of needing gradient rules for that.
=#

function zero_pad_2d(input::AbstractArray{T, 4}, padding::NTuple{4, Int}) where {T <: Number}
    width, height, channels, current_batch_size = size(input)
    x_pad1, x_pad2, y_pad1, y_pad2 = padding
    output_height, output_width = height + y_pad1 + y_pad2, width + x_pad1 + x_pad2
    output = zeros(T, output_width, output_height, channels, current_batch_size)
    output[x_pad1 + 1:output_width - x_pad2, y_pad1 + 1:output_height - y_pad2, :, :] = input

    return output
end

function NNlib.conv!(output::Array{T,4}, input::Array{T,4}, weight::Array{T,4}, cdims::ConvDims; kw...) where {T<:Real}
    # println("myconv called")

    size_weight_check_dims = (size(weight)[1:2]..., size(weight)[3]*cdims.groupcount, size(weight)[4])
    cdims_check_dims = DenseConvDims(size(input), size_weight_check_dims, stride=cdims.stride, padding=cdims.padding, dilation=cdims.dilation, groups=1, flipkernel=cdims.flipkernel)
    NNlib.check_dims(size(input), size_weight_check_dims, size(output), cdims_check_dims)

    if cdims.padding != (0, 0, 0, 0)
        #=
        invoke(NNlib.conv!, 
            Tuple{AbstractArray{T,4},AbstractArray{T,4},AbstractArray{T,4},ConvDims}, 
            output, input, weight, cdims; kw...)
        =#
        input = zero_pad_2d(input, cdims.padding)
    end

    output_width, output_height, _ = size(output)
    input_width, input_height, in_channels, batches = size(input)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)

    if !NNlib.flipkernel(cdims)
        weight = reverse(weight, dims=(1, 2))
    end

    groups = cdims.groupcount
    x_stride, y_stride = cdims.stride
    x_dilation, y_dilation = cdims.dilation
    out_channels_per_group = out_channels ÷ groups

    if cdims.groupcount == 1 && cdims.stride == (1, 1) && cdims.dilation == (1, 1)
        # println("very specialized case for maximum performance")

        @tturbo for index_batch in 1:batches
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                value = zero(T)
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    value += input[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] * weight[x_w, y_w, in_channel, out_channel]
                end
                output[x_out, y_out, out_channel, index_batch] = value #+ bias[out_channel]
            end
        end

    elseif groups == 1
        # println("second specialized case for better performance")

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
                output[x_out, y_out, out_channel, index_batch] = value #+ bias[out_channel]
            end
        end

    else 
        # println("general case for any convolution")

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
                output[x_out, y_out, out_channel, index_batch] = value #+ bias[out_channel]
            end
        end
    end

    return output
end

function NNlib.∇conv_data!(input_gradient::Array{T,4}, output_gradient::Array{T,4}, weight::Array{T,4}, cdims::ConvDims; kw...) where {T<:Real}
    # println("myconv data back called")
    
    size_weight_check_dims = (size(weight)[1:2]..., size(weight)[3]*cdims.groupcount, size(weight)[4])
    cdims_check_dims = DenseConvDims(size(input_gradient), size_weight_check_dims, stride=cdims.stride, padding=cdims.padding, dilation=cdims.dilation, groups=1, flipkernel=cdims.flipkernel)
    NNlib.check_dims(size(input_gradient), size_weight_check_dims, size(output_gradient), cdims_check_dims)
    
    # storing all the necessary shapes
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)
    # because in the actual computation section, values are added, it's saver to reset the given input_gradient first
    input_gradient .= zero(T)
    # check if input_gradient must be padded 
    if cdims.padding != (0, 0, 0, 0)
        input_gradient_padded = zero_pad_2d(input_gradient, cdims.padding)
    else
        input_gradient_padded = input_gradient
    end
    # store the size of input after padding 
    input_width, input_height, in_channels, current_batch_size = size(input_gradient_padded) # size after padding

    if !NNlib.flipkernel(cdims)
        weight = reverse(weight, dims=(1, 2))
    end

    groups = cdims.groupcount
    x_stride, y_stride = cdims.stride
    x_dilation, y_dilation = cdims.dilation
    out_channels_per_group = out_channels ÷ groups
    # actual computation
    if groups == 1 && cdims.stride == (1, 1) && cdims.dilation == (1, 1) # very specialized case for maximum performance
        # println("very specialized case for maximum performance")
        Threads.@threads for index_batch in 1:current_batch_size
            @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    input_gradient_padded[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] += weight[x_w, y_w, in_channel, out_channel] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
            end
        end
    elseif groups == 1 # second specialized case for better performance
        # println("second specialized case for better performance")
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
    else # general case for any convolution 
        # println("general case for any convolution")
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

#=
function NNlib.∇conv_filter!(weight_gradient::Array{T,4}, input::Array{T,4}, output_gradient::Array{T,4}, cdims::ConvDims; kw...) where {T<:Real}
    # println("myconv filter back called")
    
    size_weight_check_dims = (size(weight_gradient)[1:2]..., size(weight_gradient)[3]*cdims.groupcount, size(weight_gradient)[4])
    cdims_check_dims = DenseConvDims(size(input), size_weight_check_dims, stride=cdims.stride, padding=cdims.padding, dilation=cdims.dilation, groups=1, flipkernel=cdims.flipkernel)
    NNlib.check_dims(size(input), size_weight_check_dims, size(output_gradient), cdims_check_dims)
    
    # storing all the necessary shapes
    input_width, input_height, in_channels, current_batch_size = size(input)
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight_gradient)

    # check if input must be padded 
    if cdims.padding != (0, 0, 0, 0)
        input_padded = zero_pad_2d(input, cdims.padding)
    else
        input_padded = input
    end

    groups = cdims.groupcount
    x_stride, y_stride = cdims.stride
    x_dilation, y_dilation = cdims.dilation
    out_channels_per_group = out_channels ÷ groups
    # actual computation 
    if groups == 1 && cdims.stride == (1, 1) && cdims.dilation == (1, 1) # very specialized case for maximum performance
        # println("very specialized case for maximum performance")
        #=
        @tturbo for out_channel in 1:out_channels
            for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                value = zero(T)
                for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                    value += input_padded[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
                weight_gradient[x_w, y_w, in_channel, out_channel] = value
            end
        end
        =#
        weight_gradient_batched = zeros(T, weight_width, weight_height, in_channels_weight, out_channels, current_batch_size)
        @tturbo for index_batch in 1:current_batch_size # Threads.@threads
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width # @turbo
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    weight_gradient_batched[x_w, y_w, in_channel, out_channel, index_batch] += input_padded[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
            end
        end
        # weight_gradient .= @time dropdims(sum(weight_gradient_batched, dims=5), dims=5)
        weight_gradient .= dropdims(sum(weight_gradient_batched, dims=5), dims=5)
    elseif groups == 1 # second specialized case for better performance
        # println("second specialized case for better performance")
        #=
        @tturbo for out_channel in 1:out_channels
            for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                value = zero(T)
                for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                    m = y_out + (y_stride - 1) * (y_out - 1)
                    n = x_out + (x_stride - 1) * (x_out - 1)
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    value += input_padded[x_in, y_in, in_channel, index_batch] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
                weight_gradient[x_w, y_w, in_channel, out_channel] = value
            end
        end
        =#
        weight_gradient_batched = zeros(T, weight_width, weight_height, in_channels_weight, out_channels, current_batch_size)
        @tturbo for index_batch in 1:current_batch_size # Threads.@threads
            for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width # @turbo 
                m = y_out + (y_stride - 1) * (y_out - 1)
                n = x_out + (x_stride - 1) * (x_out - 1)
                for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    weight_gradient_batched[x_w, y_w, in_channel, out_channel, index_batch] += input_padded[x_in, y_in, in_channel, index_batch] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
            end
        end
        # weight_gradient .= @time dropdims(sum(weight_gradient_batched, dims=5), dims=5)
        weight_gradient .= dropdims(sum(weight_gradient_batched, dims=5), dims=5)
    else # general case for any convolution 
        # println("general case for any convolution")
        #=
        @tturbo for out_channel_per_group in 1:out_channels_per_group
            for group in 1:groups, in_channel_weight in 1:in_channels_weight, y_w in 1:weight_height, x_w in 1:weight_width
                value = zero(T)
                for index_batch in 1:current_batch_size, y_out in 1:output_height, x_out in 1:output_width
                    m = y_out + (y_stride - 1) * (y_out - 1)
                    n = x_out + (x_stride - 1) * (x_out - 1)
                    out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                    y_in = m + (y_w - 1) * y_dilation
                    x_in = n + (x_w - 1) * x_dilation
                    in_channel_input = in_channel_weight + (group - 1) * in_channels_weight
                    value += input_padded[x_in, y_in, in_channel_input, index_batch] * output_gradient[x_out, y_out, out_channel, index_batch]
                end
                weight_gradient[x_w, y_w, in_channel_weight, out_channel] = value
            end
        end
        =#
        weight_gradient_batched = zeros(T, weight_width, weight_height, in_channels_weight, out_channels, current_batch_size)
        @tturbo for index_batch in 1:current_batch_size # Threads.@threads
            for out_channel_per_group in 1:out_channels_per_group # @turbo 
                for group in 1:groups, y_out in 1:output_height, x_out in 1:output_width
                    m = y_out + (y_stride - 1) * (y_out - 1)
                    n = x_out + (x_stride - 1) * (x_out - 1)
                    out_channel = (group * out_channels_per_group + 1) - out_channel_per_group
                    for in_channel_weight in 1:in_channels_weight, y_w in 1:weight_height, x_w in 1:weight_width
                        y_in = m + (y_w - 1) * y_dilation
                        x_in = n + (x_w - 1) * x_dilation
                        in_channel_input = in_channel_weight + (group - 1) * in_channels_weight
                        weight_gradient_batched[x_w, y_w, in_channel_weight, out_channel, index_batch] += input_padded[x_in, y_in, in_channel_input, index_batch] * output_gradient[x_out, y_out, out_channel, index_batch]
                    end
                end
            end
        end
        # weight_gradient .= @time dropdims(sum(weight_gradient_batched, dims=5), dims=5)
        weight_gradient .= dropdims(sum(weight_gradient_batched, dims=5), dims=5)
    end

    if !NNlib.flipkernel(cdims)
        weight_gradient = reverse(weight_gradient, dims=(1, 2))
    end
    
    return weight_gradient
end
=#