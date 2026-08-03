# BFloat16 support on CUDA/cuDNN.
#
# cuDNN's data-type wrapper `CUDNNFloat` includes BFloat16 (cuDNN.jl maps it to
# CUDNN_DATA_BFLOAT16), so conv, pooling, softmax, activations and batchnorm all
# route to cuDNN in BFloat16 exactly like Float16. cuDNN has no native BFloat16
# accumulation config, so convolutions compute in Float32 (as Float16 does with
# PSEUDO_HALF_CONFIG); results stay BFloat16 and match a Float32 reference up to
# BFloat16 rounding.
#
# IMPORTANT (JuliaMath/BFloat16s.jl#107): vectorized *rounding to* BFloat16 on the
# CPU — `Float32 -> BFloat16` broadcast, or `BFloat16 .± BFloat16` — can hang LLVM
# codegen (`X86ISD::VFPROUND`) on some CPUs. Every BFloat16 rounding in these tests
# therefore happens on the GPU; on the host we only *widen* (`Float32.(::BFloat16)`),
# which is safe. Build inputs as host Float32 and round on the device with `tobf`;
# never write `BFloat16.(x)` for a host array `x`.

# host Float32 array -> device BFloat16 (rounding on the GPU, never on the host)
tobf(x) = BFloat16.(CuArray(x))

# `f` is run on host Float32 inputs (the reference) and on their BFloat16 GPU copies.
# The GPU output must stay BFloat16 and agree with the Float32 reference up to bf16
# rounding.
function bf16_matches_f32(f, xs...; rtol=5e-2, atol=5e-2)
    ref = f(xs...)
    out = f(map(tobf, xs)...)
    @test eltype(out) == BFloat16
    @test Float32.(collect(out)) ≈ Float32.(ref) rtol=rtol atol=atol
end

@testset "convolution" begin
    rng = StableRNG(17)
    @testset "groups=$groups, num_spatial_dims=$nsd" for groups in (1, 2), nsd in (1, 2, 3)
        C_in  = groups == 1 ? 3 : 4
        C_out = 4
        x = rand(rng, Float32, fill(8, nsd)..., C_in, 2)
        w = rand(rng, Float32, fill(2, nsd)..., C_in ÷ groups, C_out)
        cdims = DenseConvDims(x, w; groups)
        dy = rand(rng, Float32, size(NNlib.conv(x, w, cdims))...)

        bf16_matches_f32((x, w)  -> NNlib.conv(x, w, cdims), x, w)
        bf16_matches_f32((dy, w) -> NNlib.∇conv_data(dy, w, cdims), dy, w)
        bf16_matches_f32((x, dy) -> NNlib.∇conv_filter(x, dy, cdims), x, dy)
    end

    @testset "conv_bias_act ($act)" for act in (identity, NNlib.relu)
        x    = rand(rng, Float32, 8, 8, 3, 2)
        w    = rand(rng, Float32, 2, 2, 3, 4)
        bias = rand(rng, Float32, 1, 1, 4, 1)
        cdims = DenseConvDims(x, w)
        bf16_matches_f32((x, w, bias) -> NNlib.conv_bias_act(x, w, cdims, bias, act), x, w, bias)
    end
end

@testset "pooling" begin
    rng = StableRNG(23)
    @testset "num_spatial_dims=$nsd" for nsd in (1, 2, 3)
        x = rand(rng, Float32, fill(8, nsd)..., 3, 2)
        pdims = PoolDims(x, 2)

        bf16_matches_f32(x -> maxpool(x, pdims), x)
        bf16_matches_f32(x -> meanpool(x, pdims), x)

        # ∇meanpool is smooth, so compare elementwise.
        dy = rand(rng, Float32, size(meanpool(x, pdims))...)
        bf16_matches_f32((dy, x) -> ∇meanpool(dy, meanpool(x, pdims), x, pdims), dy, x)

        # ∇maxpool routes the gradient to the argmax; in BFloat16 near-tied values
        # can round equal and shift which element is selected, so an elementwise
        # comparison is not meaningful. The routed total is invariant, so we check
        # that the sum matches and the type/shape are preserved.
        y   = maxpool(x, pdims)
        dym = rand(rng, Float32, size(y)...)
        dxf = ∇maxpool(dym, y, x, pdims)
        dxb = ∇maxpool(tobf(dym), maxpool(tobf(x), pdims), tobf(x), pdims)
        @test eltype(dxb) == BFloat16
        @test size(dxb) == size(dxf)
        @test sum(Float32.(collect(dxb))) ≈ sum(dxf) rtol=5e-2 atol=5e-2
    end
end

@testset "softmax" begin
    rng = StableRNG(29)
    # cuDNN routes a leading contiguous softmax axis; other `dims` use the generic
    # kernel. Both stay BFloat16 and match a Float32 reference.
    for (sz, dims) in [((20,), :), ((20,), 1), ((10, 8), 1), ((4, 4, 4, 3), (1, 2))]
        x = rand(rng, Float32, sz...)
        bf16_matches_f32(x -> softmax(x; dims), x)
        bf16_matches_f32(x -> logsoftmax(x; dims), x)
    end
end

@testset "activations" begin
    rng = StableRNG(31)
    # These activations are routed to cuDNN by NNlibCUDACUDNNExt; the rest fall back
    # to the generic broadcast, which also supports BFloat16.
    for f in (tanh, NNlib.σ, NNlib.elu, NNlib.relu)
        x = rand(rng, Float32, 8, 4) .- 0.5f0    # host Float32 arithmetic is safe
        bf16_matches_f32(y -> f.(y), x)
    end
end

@testset "batchnorm" begin
    rng = StableRNG(37)
    # cuDNN requires the scale/bias/statistics tensors to be Float32 when the feature
    # maps are half precision. BFloat16 feature maps with Float32 parameters must
    # therefore match a Float32-feature-map reference (same Float32 parameters).
    @testset "$(nd)D data" for (nd, sz) in ((2, (3, 8)), (4, (4, 4, 3, 8)))
        xf = rand(rng, Float32, sz...)
        g  = rand(rng, Float32, 3);  b  = rand(rng, Float32, 3)
        rm = zeros(Float32, 3);      rv = ones(Float32, 3)
        gg, bg, rmg, rvg = CuArray(g), CuArray(b), CuArray(rm), CuArray(rv)
        xff, xbb = CuArray(xf), tobf(xf)

        @testset "training=$training, track_stats=$track" for training in (true, false), track in (true, false)
            kws = (; training, track_stats = track)
            yf = batchnorm(gg, bg, xff, copy(rmg), copy(rvg), 0.1f0; kws...)
            yb = batchnorm(gg, bg, xbb, copy(rmg), copy(rvg), 0.1f0; kws...)
            @test eltype(yb) == BFloat16
            @test Float32.(collect(yb)) ≈ collect(yf) rtol=5e-2 atol=5e-2

            dyf = CuArray(rand(rng, Float32, size(yf)...))
            dyb = BFloat16.(dyf)
            dgf, dbf, dxf = ∇batchnorm(gg, bg, xff, dyf, copy(rmg), copy(rvg), 0.1f0; kws...)
            dgb, dbb, dxb = ∇batchnorm(gg, bg, xbb, dyb, copy(rmg), copy(rvg), 0.1f0; kws...)
            @test eltype(dxb) == BFloat16       # dx follows the feature-map type
            @test eltype(dgb) == Float32        # parameter grads stay Float32
            @test Float32.(collect(dxb)) ≈ collect(dxf) rtol=5e-2 atol=5e-2
            @test collect(dgb) ≈ collect(dgf) rtol=5e-2 atol=5e-2
            @test collect(dbb) ≈ collect(dbf) rtol=5e-2 atol=5e-2
        end
    end

    @testset "type contract" begin
        xb  = CUDA.randn(BFloat16, 4, 4, 3, 8)
        gb  = CUDA.rand(BFloat16, 3);  bb  = CUDA.rand(BFloat16, 3)
        gf  = CUDA.rand(Float32, 3);   bf  = CUDA.rand(Float32, 3)
        rmb = CUDA.zeros(BFloat16, 3); rvb = CUDA.ones(BFloat16, 3)

        # Half-precision scale/bias with half-precision data is rejected: cuDNN
        # needs Float32 parameters there. We reject rather than silently convert.
        @test_throws ArgumentError batchnorm(gb, bb, xb, nothing, nothing, 0.1f0; training=true)
        @test_throws ArgumentError ∇batchnorm(gb, bb, xb, xb, nothing, nothing, 0.1f0; training=true)
        # Half-precision running statistics with Float32 parameters are rejected too.
        @test_throws ArgumentError batchnorm(gf, bf, xb, rmb, rvb, 0.1f0; training=true)
        # Auto-affine (`nothing`) builds Float32 parameters, so half data works.
        y = batchnorm(nothing, nothing, xb, nothing, nothing, 0.1f0; training=true)
        @test eltype(y) == BFloat16
    end
end
