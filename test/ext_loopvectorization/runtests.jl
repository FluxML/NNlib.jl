using NNlib, Test, BenchmarkTools

function compute_conv_outputs(settings::Vector{<:NNlib.ConvDims}, input::Array{T,4}, weight_ungrouped::Array{T,4}, weight_grouped::Array{T,4}, conv_output_grads::Vector{Array{T,4}}) where {T<:Real}
    conv_outs = Vector{Array{T, 4}}(undef, length(settings))
    conv_grads = Vector{Array{T, 4}}(undef, length(settings))

    for (i, setting) in enumerate(settings)
        if setting.groupcount > 1
            weight = weight_grouped
        else
            weight = weight_ungrouped
        end

        out = @btime NNlib.conv($input, $weight, $setting)
        output_gradient = conv_output_grads[i]

        conv_grads[i] = @btime NNlib.∇conv_data($output_gradient, $weight, $setting)
        conv_outs[i] = out
    end

    return conv_outs, conv_grads
end

function compute_pool_outputs(settings::Vector{<:NNlib.PoolDims}, input::Array{T,4}) where {T<:Real}
    pool_outs = Vector{Array{T, 4}}(undef, length(settings))

    for (i, setting) in enumerate(settings)
        pdims = NNlib.PoolDims(size(input), setting.kernel_size, stride = setting.stride, padding = setting.padding, dilation = setting.dilation)
        pool_outs[i] = @btime NNlib.meanpool($input, $pdims)
    end

    return pool_outs
end

@testset "Convolution & Pooling & Activations" begin
    
    dtype = Float32 # Float64
    batch_size = 64 # 1 # 64 # 32
    input = rand(dtype, 224, 224, 3, batch_size) # for conv & pool
    weight_ungrouped = rand(dtype, 5, 5, 3, 27) # for conv
    weight_grouped = rand(dtype, 5, 5, 1, 27) # for grouped conv
    
    conv_settings_list = [
        NNlib.DenseConvDims(size(input), size(weight_ungrouped), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1), # test 'very specialized case'
        NNlib.DenseConvDims(size(input), size(weight_ungrouped), stride=(2, 1), padding=(0, 0), dilation=(1, 1), groups=1), # test 'second specialized case'
        NNlib.DenseConvDims(size(input), size(weight_ungrouped), stride=(2, 1), padding=(0, 0), dilation=(2, 1), groups=1), # test 'third specialized case'
        NNlib.DenseConvDims(size(input), size(weight_grouped), stride=(2, 1), padding=(2, 0), dilation=(2, 1), groups=3), # test 'general case'
    ]

    conv_output_grads = [rand(dtype, NNlib.output_size(setting)..., 27, batch_size) for setting in conv_settings_list]

    pool_settings_list = [
        NNlib.PoolDims(size(input), (5, 4), stride=(1, 1), padding=(0, 0), dilation=(1, 1)), # test 'specialized case'
        NNlib.PoolDims(size(input), (5, 4), stride=(5, 4), padding=(2, 0), dilation=(2, 1)), # test 'general case'
    ]

    modified_activations = [tanh, softsign, softplus, sigmoid, sigmoid_fast, hardsigmoid, logsigmoid, swish]

    # compute outputs before loading LoopVectorization
    
    println("without LoopVectorization")
    conv_outs_std, conv_grads_std = compute_conv_outputs(conv_settings_list, input, weight_ungrouped, weight_grouped, conv_output_grads)
    pool_outs_std = compute_pool_outputs(pool_settings_list, input)
    act_outs_std = [@btime $act.($input) for act in modified_activations]

    using LoopVectorization # now load the NNlibLoopVectorizationExt

    println("with LoopVectorization")
    conv_outs_lv, conv_grads_lv = compute_conv_outputs(conv_settings_list, input, weight_ungrouped, weight_grouped, conv_output_grads)
    pool_outs_lv = compute_pool_outputs(pool_settings_list, input)
    act_outs_lv = [@btime $act.($input) for act in modified_activations]

    # validate conv
    @testset "Convolution" begin
        @test all(isapprox.(conv_outs_std, conv_outs_lv))
        @test all(isapprox.(conv_grads_std, conv_grads_lv))
    end
    # validate pool
    @testset "Pooling" begin
        @test all(isapprox.(pool_outs_std, pool_outs_lv))
    end
    # validate activations
    @testset "Activations" begin
        @test all(isapprox.(act_outs_std, act_outs_lv))
    end

end