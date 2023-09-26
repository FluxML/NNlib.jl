using NNlib, Test

function compute_conv_outputs(settings::Vector{<:NamedTuple}, input::Array{T,4}, weight_ungrouped::Array{T,4}, weight_grouped::Array{T,4}) where {T<:Real}
    conv_outs = Vector{Array{T, 4}}(undef, length(settings))
    conv_grads = Vector{Array{T, 4}}(undef, length(settings))

    for (i, setting) in enumerate(settings)
        if setting.groups > 1
            weight = weight_grouped
        else
            weight = weight_ungrouped
        end

        cdims = NNlib.DenseConvDims(size(input), size(weight), stride = setting.stride, padding = setting.padding, dilation = setting.dilation, groups = setting.groups)

        out = NNlib.conv(input, weight, cdims)
        output_gradient = ones(T, size(out))

        conv_grads[i] = NNlib.∇conv_data(output_gradient, weight, cdims)
        conv_outs[i] = out
    end

    return conv_outs, conv_grads
end

function compute_pool_outputs(settings::Vector{<:NamedTuple}, input::Array{T,4}) where {T<:Real}
    pool_outs = Vector{Array{T, 4}}(undef, length(settings))

    for (i, setting) in enumerate(settings)
        pdims = NNlib.PoolDims(size(input), setting.kernel_size, stride = setting.stride, padding = setting.padding, dilation = setting.dilation)
        pool_outs[i] = NNlib.meanpool(input, pdims)
    end

    return pool_outs
end

@testset "Convolution & Pooling" begin
    
    dtype = Float32
    input = rand(dtype, 224, 224, 3, 64) # for conv & pool
    weight_ungrouped = rand(dtype, 5, 5, 3, 27) # for conv
    weight_grouped = rand(dtype, 5, 5, 1, 27) # for grouped conv
    
    conv_settings_list = [
        (; stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1), # test 'very specialized case'
        (; stride=(2, 1), padding=(2, 0), dilation=(2, 1), groups=1), # test 'second specialized case'
        (; stride=(2, 1), padding=(2, 0), dilation=(2, 0), groups=3), # test 'general case'
    ]

    pool_settings_list = [
        (; kernel_size=(5, 4), stride=(1, 1), padding=(0, 0), dilation=(1, 1)), # test 'specialized case'
        (; kernel_size=(5, 4), stride=(2, 1), padding=(2, 0), dilation=(2, 1)), # test 'general case'
    ]

    # compute outputs before loading LoopVectorization

    conv_outs_std, conv_grads_std = compute_conv_outputs(conv_settings_list, input, weight_ungrouped, weight_grouped)
    pool_outs_std = compute_pool_outputs(pool_settings_list, input)

    using LoopVectorization # now load the NNlibLoopVectorizationExt

    conv_outs_lv, conv_grads_lv = compute_conv_outputs(conv_settings_list, input, weight_ungrouped, weight_grouped)
    pool_outs_lv = compute_pool_outputs(pool_settings_list, input)

    # validate conv
    @test all(isapprox.(conv_outs_std, conv_outs_lv))
    @test all(isapprox.(conv_grads_std, conv_grads_lv))
    # validate pool
    @test all(isapprox.(pool_outs_std, pool_outs_lv))

end