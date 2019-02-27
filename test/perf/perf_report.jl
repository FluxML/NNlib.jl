using JLD2, NNlib, BenchmarkTools

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

for rank in (2,),
    N in (10, 20, 40, 80),
    C_in in (1, 2, 4),
    C_out in (1, 2, 4),
    K in (3, 6, 12),
    stride in (1, 2, 4),
    padding in (0, 2, 4)

    for (conv!, ∇conv_data!, ∇conv_filter!, backend) in (
            (NNlib.conv2d!, NNlib.conv2d_grad_x!, NNlib.conv2d_grad_w!, "im2col"),
            (NNlib.depthwiseconv2d!, NNlib.depthwiseconv2d_grad_x!, NNlib.depthwiseconv2d_grad_w!, "im2col"),
        )

        x = zeros(Float32, repeat([N], rank)..., C_in, 1)
        if conv! == NNlib.conv2d!
            w = zeros(Float32, repeat([K], rank)..., C_in, C_out)
        else
            w = zeros(Float32, repeat([K], rank)..., C_out, C_in)
        end
        y = zeros(Float32, repeat([div(N + 2*padding - (K-1), stride) + 1], rank)..., C_out, 1)

        dx = similar(x)
        dw = similar(w)
        dy = similar(y)

        t_fwd = @benchmark $(conv!)($y, $x, $w; stride=$stride, padding=$padding)
        t_dx = @benchmark $(∇conv_data!)($dx, $x, $w, $dy; stride=$stride, padding=$padding)
        t_dw = @benchmark $(∇conv_filter!)($dw, $x, $w, $dy; stride=$stride, padding=$padding)

        fake_cdims = (rank, N, K, C_in, C_out, stride, padding)
        add_result(t_fwd, "conv$(rank)d", backend, fake_cdims)
        add_result(t_dx, "conv$(rank)d_data", backend, fake_cdims)
        add_result(t_dw, "conv$(rank)d_filter", backend, fake_cdims)

        @show fake_cdims
        @save "results.jld2" results
    end
end


for rank in (2,),
    N in (10, 20, 40, 80),
    K in (2, 4),
    stride in (1, 2, 4)

    x = zeros(Float32, repeat([N], rank)..., 1, 1)
    y = zeros(Float32, repeat([div(N, stride)], rank)..., 1, 1)
    dx = similar(x)
    dy = similar(y)

    for (pool, ∇pool, name) in (
            (NNlib.maxpool2d!, NNlib.maxpool2d_grad!, "maxpool"),
            (NNlib.meanpool2d!, NNlib.meanpool2d_grad!, "meanpool"),
        )

        t_fwd  = @benchmark $(pool)( $y, $x; window=($K, $K), stride=($stride, $stride))
        t_data = @benchmark $(∇pool)($dx, $dy, $y, $x, window=($K, $K), stride=($stride, $stride))

        fake_pdims = (rank, N, K, stride)
        add_result(t_fwd, "$(name)$(rank)d", "direct", fake_pdims)
        add_result(t_data, "$(name)$(rank)d_data", "direct", fake_pdims)

        @show fake_pdims
        @save "results.jld2" results
    end
end
