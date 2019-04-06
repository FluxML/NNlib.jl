using JLD2, NNlib, BenchmarkTools

# We need things to go quickly here
BenchmarkTools.DEFAULT_PARAMETERS.samples = 20
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.5

results = Dict()

function add_result(val, keys...)
    r = results
    for k in keys[1:end-1]
        if !haskey(r, k)
            r[k] = Dict()
        end
        r = r[k]
    end
    r[keys[end]] = val
    return r
end

# Modify these as needed
for rank in (2,),
    N in (20, 40, 80),
    C_in in (1,),
    C_out in (1,),
    K in (3,),
    stride in (1,),
    dilation in (1,),
    padding in (0, 2)

    for (conv!, ∇conv_data!, ∇conv_filter!, cT, backend) in (
            (NNlib.conv_direct!, NNlib.∇conv_data_direct!, NNlib.∇conv_filter_direct!, DenseConvDims, "direct"),
            (NNlib.conv_im2col!, NNlib.∇conv_data_im2col!, NNlib.∇conv_filter_im2col!, DenseConvDims, "im2col"),
            (NNlib.depthwiseconv_direct!, NNlib.∇depthwiseconv_data_direct!, NNlib.∇depthwiseconv_filter_direct!, DepthwiseConvDims, "direct"),
            (NNlib.depthwiseconv_im2col!, NNlib.∇depthwiseconv_data_im2col!, NNlib.∇depthwiseconv_filter_im2col!, DepthwiseConvDims, "im2col"),
        )

        x = zeros(Float32, repeat([N], rank)..., C_in, 1)
        if cT == DenseConvDims
            w = zeros(Float32, repeat([K], rank)..., C_in, C_out)
        else
            w = zeros(Float32, repeat([K], rank)..., C_out, C_in)
        end
        cdims = try
            cT(x, w; stride=stride, dilation=dilation, padding=padding)
        catch
            continue
        end

        if cT == DenseConvDims
            y = zeros(Float32, NNlib.output_size(cdims)..., C_out, 1)
        else
            y = zeros(Float32, NNlib.output_size(cdims)..., C_out*C_in, 1)
        end

        dx = similar(x)
        dw = similar(w)
        dy = similar(y)

        t_fwd = @benchmark $(conv!)($y, $x, $w, $cdims)
        t_dx = @benchmark $(∇conv_data!)($dx, $y, $w, $cdims)
        t_dw = @benchmark $(∇conv_filter!)($dw, $x, $y, $cdims)

        add_result(t_fwd, "conv$(rank)d", backend, cdims)
        add_result(t_dx, "conv$(rank)d_data", backend, cdims)
        add_result(t_dw, "conv$(rank)d_filter", backend, cdims)

        @show(cdims)
        @save "results.jld2" results
    end
end


# Modify these as needed
for rank in (2,),
    N in (20,),
    K in (2, 4),
    stride in (1, 2, 4)

    x = zeros(Float32, repeat([N], rank)..., 1, 1)
    pdims = PoolDims(x, K; stride=stride)
    y = zeros(Float32, NNlib.output_size(pdims)..., 1, 1)
    dx = similar(x)

    for (pool, ∇pool, name) in (
            (NNlib.maxpool!, NNlib.∇maxpool!, "maxpool"),
            (NNlib.meanpool!, NNlib.∇meanpool!, "meanpool"),
        )

        t_fwd  = @benchmark $(pool)( $y, $x, $pdims)
        t_data = @benchmark $(∇pool)($dx, $y, $y, $x, $pdims)

        add_result(t_fwd, "$(name)$(rank)d", "direct", pdims)
        add_result(t_data, "$(name)$(rank)d_data", "direct", pdims)

        @show(pdims)
        @save "results.jld2" results
    end
end
