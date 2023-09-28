using NNlib, LoopVectorization, CpuId

function ∇conv_data!_avx(input_gradient::Array{T,4}, output_gradient::Array{T,4}, weight::Array{T,4}, cdims::ConvDims; kw...) where {T<:Real}

    NNlib.check_dims(size(input_gradient), size(weight), size(output_gradient), cdims)
    
    # storing all the necessary shapes
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)
    input_width, input_height, in_channels, current_batch_size = size(input_gradient)

    if cdims.padding != (0, 0, 0, 0) || cdims.groupcount != 1 || cdims.stride != (1, 1) || cdims.dilation != (1, 1)
        throw(ArgumentError("this test function only supports basic conv (or crosscor) bwd with pad=0, stride=1, dilation=1, groups=1"))
    end

    # it's necessary to flip the kernel if real convolution is performed (flipkernel=false)
    if !NNlib.flipkernel(cdims)
        weight = reverse(weight, dims=(1, 2))
    end

    # because in the actual computation section, values are added, it's saver to reset the given input_gradient first
    input_gradient .= zero(T)

    for index_batch in 1:current_batch_size # Threads.@threads 
        @turbo for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width
            for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                input_gradient[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] += weight[x_w, y_w, in_channel, out_channel] * output_gradient[x_out, y_out, out_channel, index_batch]
            end
        end
    end

    return input_gradient
end

function ∇conv_data!_noavx(input_gradient::Array{T,4}, output_gradient::Array{T,4}, weight::Array{T,4}, cdims::ConvDims; kw...) where {T<:Real}

    NNlib.check_dims(size(input_gradient), size(weight), size(output_gradient), cdims)
    
    # storing all the necessary shapes
    output_width, output_height, out_channels, current_batch_size = size(output_gradient)
    weight_width, weight_height, in_channels_weight, out_channels = size(weight)
    input_width, input_height, in_channels, current_batch_size = size(input_gradient)

    if cdims.padding != (0, 0, 0, 0) || cdims.groupcount != 1 || cdims.stride != (1, 1) || cdims.dilation != (1, 1)
        throw(ArgumentError("this test function only supports basic conv (or crosscor) bwd with pad=0, stride=1, dilation=1, groups=1"))
    end

    # it's necessary to flip the kernel if real convolution is performed (flipkernel=false)
    if !NNlib.flipkernel(cdims)
        weight = reverse(weight, dims=(1, 2))
    end

    # because in the actual computation section, values are added, it's saver to reset the given input_gradient first
    input_gradient .= zero(T)

    for index_batch in 1:current_batch_size # NO @threads here
        for out_channel in 1:out_channels, y_out in 1:output_height, x_out in 1:output_width # NO @turbo here!
            for in_channel in 1:in_channels, y_w in 1:weight_height, x_w in 1:weight_width
                input_gradient[x_out + x_w - 1, y_out + y_w - 1, in_channel, index_batch] += weight[x_w, y_w, in_channel, out_channel] * output_gradient[x_out, y_out, out_channel, index_batch]
            end
        end
    end

    return input_gradient
end

println(cpuinfo())

dtype = Float32 # Float64
batch_size = 5
input = rand(dtype, 50, 50, 3, batch_size)
weight = rand(dtype, 5, 5, 3, 9)
cdims = NNlib.DenseConvDims(size(input), size(weight), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
output_gradient = rand(dtype, NNlib.output_size(cdims)..., 9, batch_size)

input_gradient_noavx = zeros(dtype, size(input)...)
input_gradient_noavx = ∇conv_data!_noavx(input_gradient_noavx, output_gradient, weight, cdims)
input_gradient_noavx = @time ∇conv_data!_noavx(input_gradient_noavx, output_gradient, weight, cdims)

input_gradient_avx = zeros(dtype, size(input)...)
input_gradient_avx = ∇conv_data!_avx(input_gradient_avx, output_gradient, weight, cdims)
input_gradient_avx = @time ∇conv_data!_avx(input_gradient_avx, output_gradient, weight, cdims)

@info isapprox(input_gradient_noavx, input_gradient_avx)
@testset "conv bwd minimal" begin
    @test isapprox(input_gradient_noavx, input_gradient_avx)
end
@show sum(input_gradient_noavx)
@show sum(input_gradient_avx)