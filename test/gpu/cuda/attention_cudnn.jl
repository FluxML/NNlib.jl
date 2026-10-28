const cudnn_ext = Base.get_extension(NNlib, :NNlibCUDACUDNNExt)

function attention_options(q, k; is_causal=false, window=nothing, q_offset=0,
                           q_lengths=nothing, kv_lengths=nothing,
                           q_segments=nothing, kv_segments=q_segments, maxlen=nothing)
    is_causal = NNlib.attention_causal_mask(q, k, is_causal, window, q_offset, kv_lengths,
                                          q_segments, kv_segments)
    q, k, _, _, _, q_offset, q_lengths, kv_lengths, segments =
        NNlib.collapse_attention_inputs(q, k, k, nothing, nothing, q_offset, q_lengths, kv_lengths,
                                        q_segments, kv_segments, maxlen)
    options = cudnn_ext.cudnn_sdpa_options(q, k; is_causal, window, q_offset, q_lengths, kv_lengths, segments)
    return q, k, options, segments
end

function uses_cudnn(q, k; kws...)
    q, k, options, segments = attention_options(q, k; kws...)
    out = cudnn_ext.cudnn_sdpa_output(q, k, k, nothing, identity, nothing, options, segments)
    out === nothing && return false
    CUDA.unsafe_free!(out)
    return true
end

@testset "cuDNN 6.5 attention ($T)" for T in (Float16, BFloat16)
    CUDA.capability(CUDA.device()) < v"8.0" && continue
    q = T.(CUDA.randn(Float32, 32, 4, 12, 3) ./ 4)
    k = T.(CUDA.randn(Float32, 32, 2, 16, 3) ./ 4)
    v = T.(CUDA.randn(Float32, size(k)) ./ 4)
    qc, kc, vc = map(x -> Float64.(Array(x)), (q, k, v))
    same(x, y) = isequal(isnan.(x), isnan.(y)) &&
                 isapprox(x[.!isnan.(x)], y[.!isnan.(y)]; rtol=2f-2, atol=1f-4)
    lengths = (; q_lengths=[12, 7, 1], kv_lengths=[16, 9, 8])
    @testset "$(keys(kws))" for kws in
            ((; is_causal=:bottom_right), (; is_causal=true, q_offset=4),
             (; is_causal=true, q_offset=[4, 2, 7], lengths...),
             (; is_causal=:bottom_right, lengths...),
             (; is_causal=true, window=4), (; window=4),
             (; is_causal=:bottom_right, window=4, lengths...))
        @test uses_cudnn(q, k; kws...) == cudnn_ext.CUDNN_6_5
        @test same(Array(scaled_dot_product_attention(q, k, v; kws...)),
                   scaled_dot_product_attention(qc, kc, vc; kws...))
        weights = CUDA.randn(Float32, size(q))
        loss(w, kws) = (q, k, v) -> sum(w .* scaled_dot_product_attention(q, k, v; kws...))
        g = gradient(loss(weights, kws), q, k, v)
        ref = gradient(loss(Array(weights), kws), qc, kc, vc)
        @test all(isapprox.(map(x -> Float64.(Array(x)), g), ref; rtol=5f-2, atol=1f-4))
    end
    @testset "decoding drops an inactive causal mask" begin
        q1 = q[:, :, 1:1, :]
        for kws in ((; is_causal=:bottom_right), (; is_causal=true, q_offset=15))
            _, _, opts, _ = attention_options(q1, k; kws...)
            @test (opts !== nothing && opts.causal === false) ==
                  (kws.is_causal === true || cudnn_ext.CUDNN_6_5)
            @test same(Array(scaled_dot_product_attention(q1, k, v; kws...)),
                       scaled_dot_product_attention(qc[:, :, 1:1, :], kc, vc))
        end
    end
    @testset "inactive causal masks with valid key lengths" begin
        for (nq, offsets, lengths) in
                ((1, [7, 11, 15], [8, 12, 16]),
                 (4, [7, 11, 15], [8, 12, 16]),
                 (4, 7, [8, 7, 6]))
            qi, qci = q[:, :, 1:nq, :], qc[:, :, 1:nq, :]
            kws = (; is_causal=true, q_offset=offsets, kv_lengths=lengths,
                   q_lengths=[nq, nq - 1, 0])
            _, _, opts, _ = attention_options(qi, k; kws...)
            @test opts !== nothing && opts.causal === false
            @test uses_cudnn(qi, k; kws...)
            @test same(Array(scaled_dot_product_attention(qi, k, v; kws...)),
                       scaled_dot_product_attention(qci, kc, vc; kws.q_lengths, kws.kv_lengths))
            g = gradient((q, k, v) -> sum(scaled_dot_product_attention(q, k, v; kws...)), qi, k, v)
            ref = gradient((q, k, v) -> sum(scaled_dot_product_attention(q, k, v;
                kws.q_lengths, kws.kv_lengths)), qci, kc, vc)
            @test all(isapprox.(map(x -> Float64.(Array(x)), g), ref; rtol=5f-2, atol=1f-4))
        end
        for kws in ((; is_causal=true, q_offset=[7, 10, 15], kv_lengths=[8, 12, 16]),
                    (; is_causal=true, q_offset=[7, 11, 15], kv_lengths=[8, 12, 16], window=4))
            q1, qc1 = q[:, :, 1:1, :], qc[:, :, 1:1, :]
            _, _, opts, _ = attention_options(q1, k; kws...)
            @test opts === nothing || opts.causal !== false
            @test same(Array(scaled_dot_product_attention(q1, k, v; kws...)),
                       scaled_dot_product_attention(qc1, kc, vc; kws...))
        end
    end
    @testset "empty and padded queries" begin
        qlong = cat(q, q; dims=3)
        @test same(Array(scaled_dot_product_attention(qlong, k, v; is_causal=:bottom_right)),
                   scaled_dot_product_attention(cat(qc, qc; dims=3), kc, vc; is_causal=:bottom_right))
        @testset "$kws" for kws in
                ((; window=2, q_lengths=[12, 7, 0], kv_lengths=[3, 0, 8]),
                 (; is_causal=:bottom_right, q_lengths=[12, 7, 0], kv_lengths=[3, 0, 8]),
                 (; is_causal=:bottom_right, window=2, q_lengths=[12, 7, 0], kv_lengths=[3, 0, 8]))
            @test same(Array(scaled_dot_product_attention(q, k, v; kws...)),
                       scaled_dot_product_attention(qc, kc, vc; kws...))
            # NNlib's fused gradient excludes queries without a visible key.
            ci = NNlib.collapse_attention_inputs(q, k, v, nothing, nothing, 0, kws.q_lengths, kws.kv_lengths)
            cq, ck, cv, bias, mask, q_offset, q_lengths, kv_lengths, segments = ci
            generic = (q, k, v) -> NNlib.generic_attention(q, k, v, bias;
                fdrop=identity, mask, scale=nothing, is_causal=get(kws, :is_causal, false),
                window=get(kws, :window, nothing), q_offset, q_lengths, kv_lengths, segments)
            g = gradient((q, k, v) -> sum(scaled_dot_product_attention(q, k, v; kws...)), q, k, v)
            ref = gradient((q, k, v) -> sum(generic(q, k, v)), cq, ck, cv)
            @test all(same.(map(Array, g), map(Array, ref)))
        end
    end
    @testset "packed sequences" begin
        qs, ks = [12, 0, 7, 1], [16, 3, 9, 8]
        qp = T.(CUDA.randn(Float32, 32, 4, sum(qs)) ./ 4)
        kp = T.(CUDA.randn(Float32, 32, 2, sum(ks)) ./ 4)
        vp = T.(CUDA.randn(Float32, size(kp)) ./ 4)
        qpc, kpc, vpc = map(x -> Float64.(Array(x)), (qp, kp, vp))
        segs = (; q_segments=qs, kv_segments=ks)
        for kws in ((;), (; is_causal=true), (; is_causal=:bottom_right),
                    (; is_causal=true, window=4), (; window=3))
            @test uses_cudnn(qp, kp; segs..., kws...) == cudnn_ext.CUDNN_6_5
            @test same(Array(scaled_dot_product_attention(qp, kp, vp; segs..., kws...)),
                       scaled_dot_product_attention(qpc, kpc, vpc; segs..., kws...))
            g = gradient((q, k, v) -> sum(scaled_dot_product_attention(q, k, v; segs..., kws...)), qp, kp, vp)
            ref = gradient((q, k, v) -> sum(scaled_dot_product_attention(q, k, v; segs..., kws...)), qpc, kpc, vpc)
            @test all(isapprox.(map(x -> Float64.(Array(x)), g), ref; rtol=5f-2, atol=1f-4))
        end
        inactive = (; is_causal=true, q_offset=ks .- 1)
        _, _, opts, _ = attention_options(qp, kp; segs..., inactive...)
        @test (opts !== nothing && opts.causal === false) == cudnn_ext.CUDNN_6_5
        @test same(Array(scaled_dot_product_attention(qp, kp, vp; segs..., inactive...)),
                   scaled_dot_product_attention(qpc, kpc, vpc; segs...))
        @test !uses_cudnn(qp, kp; segs..., q_lengths=[10, 0, 7, 1])
        @test uses_cudnn(qp, kp; q_segments=CuArray(qs), kv_segments=CuArray(ks), maxlen=16) == cudnn_ext.CUDNN_6_5
        @test same(Array(scaled_dot_product_attention(qp, kp, vp;
                       q_segments=CuArray(qs), kv_segments=CuArray(ks), maxlen=16, is_causal=:bottom_right)),
                   scaled_dot_product_attention(qpc, kpc, vpc; segs..., is_causal=:bottom_right))
    end
end
