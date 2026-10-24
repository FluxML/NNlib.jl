# the attention weights alone
attention_weights(q, k, args...; kws...) =
    last(scaled_dot_product_attention_with_weights(q, k, k, args...; kws...))

# The materialized implementation: an `fdrop` other than `identity` keeps a call on it.
materialized_attention(args...; kws...) = scaled_dot_product_attention(args...; fdrop = x -> x, kws...)

# The gradients of `sum(w .* f(xs...; kws...))` with respect to the `xs`.
attention_gradients(f, w, xs...; kws...) = gradient((xs...) -> sum(w .* f(xs...; kws...)), xs...)

@testset "different batchsizes" begin
    head_dim = 5
    lenq = 3
    lenkv = 4
    for batch_size in [(), 1, 2, (2,1,3)], nheads in [1, 3, 5]
        q = rand(Float32, head_dim, nheads, lenq, batch_size...)
        k = rand(Float32, head_dim, nheads, lenkv, batch_size...)
        v = rand(Float32, head_dim, nheads, lenkv, batch_size...)
        y = scaled_dot_product_attention(q, k, v)
        α = attention_weights(q, k)
        @test y isa Array{Float32}
        @test size(y) == (head_dim, nheads, lenq, batch_size...)
        @test size(α) == (lenkv, lenq, nheads, batch_size...)
        @test sum(α, dims=1) ≈ ones(1, lenq, nheads, batch_size...)
    end
end

@testset "output and weights in one pass" begin
    q = rand(Float32, 4, 2, 3, 1)
    k = rand(Float32, 4, 2, 5, 1)
    v = rand(Float32, 6, 2, 5, 1)
    y, α = @inferred scaled_dot_product_attention_with_weights(q, k, v)
    @test y ≈ scaled_dot_product_attention(q, k, v)
    @test size(α) == (5, 3, 2, 1)

    # the output is computed from the returned weights, also with a stochastic `fdrop`
    fdrop(α) = α .* (rand(Float32, size(α)) .> 0.5f0)
    y, α = scaled_dot_product_attention_with_weights(q, k, v; fdrop)
    @test any(iszero, α)
    @test y ≈ [sum(α[j, i, h, b] * v[d, h, j, b] for j in 1:5) for d in 1:6, h in 1:2, i in 1:3, b in 1:1]

    # batch dimensions are restored in both
    qb, kb, vb = map(x -> repeat(x; outer=(1, 1, 1, 2, 3)), (q, k, v))
    y, α = @inferred scaled_dot_product_attention_with_weights(qb, kb, vb)
    @test size(y) == (6, 2, 3, 2, 3)
    @test size(α) == (5, 3, 2, 2, 3)

    @test sdpa === scaled_dot_product_attention
    @test sdpa_with_weights === scaled_dot_product_attention_with_weights

    qd, kd, vd = map(x -> Float64.(x), (q, k, v))
    # a loss on both the output and the weights
    @test test_gradients((x...) -> scaled_dot_product_attention_with_weights(x...), qd, kd, vd;
                         loss = (f, xs...) -> sum(x -> mean(abs2, x), f(xs...)))
end

@testset "weights ↔ output consistency" begin
    q = rand(Float32, 4, 2, 3, 1)
    k = v = rand(Float32, 4, 2, 5, 1)
    y, α = scaled_dot_product_attention_with_weights(q, k, v)
    # y[:, h, i, b] = Σ_j α[j, i, h, b] * v[:, h, j, b]
    yref = similar(y)
    for b in 1:1, h in 1:2, i in 1:3
        yref[:, h, i, b] = sum(α[j, i, h, b] .* v[:, h, j, b] for j in 1:5)
    end
    @test y ≈ yref
end

@testset "specific results" begin
    # (head_dim, nheads, seq_len, batch) = (2, 2, 3, 1)
    q = k = v = reshape([1:12;] ./ 12, 2, 2, 3, 1)
    y = scaled_dot_product_attention(q, k, v)
    α = attention_weights(q, k)
    ytrue = [0.429754, 0.513087, 0.613791, 0.697125, 0.46431, 0.547644, 0.647876, 0.73121, 0.49773, 0.581064, 0.680455, 0.763788]
    ytrue = reshape(ytrue, 4, 3, 1)
    αtrue = [0.313896, 0.332948, 0.353157, 0.264431, 0.328206, 0.407362, 0.219215, 0.31838, 0.462405, 0.288691, 0.331243, 0.380066, 0.241239, 0.323893, 0.434868, 0.198438, 0.311761, 0.489801]
    αtrue = reshape(αtrue, 3, 3, 2, 1)
    @test reshape(y, 4, 3, 1) ≈ ytrue atol=1e-5   # heads joined back for comparison
    @test α ≈ αtrue atol=1e-5
end

@testset "mask" begin
    q = rand(4, 2, 3, 1)
    k = rand(4, 2, 5, 1)

    mask = rand(Bool, (5, 3))
    α = attention_weights(q, k; mask)
    @test all((α[:,:,1,1].> 0) .== mask)
    @test all((α[:,:,2,1].> 0) .== mask)

    @testset "causal" begin
        x = rand(4, 2, 3, 1)
        mask = make_causal_mask(x, dims=3)
        α = attention_weights(x, x; mask)
        @test all((α[:,:,1,1].> 0) .== mask)
        @test all((α[:,:,2,1].> 0) .== mask)
    end
end

@testset "dropout" begin
    q = k = rand(5, 2, 10)
    fdrop(x, p) = (Random.rand!(similar(x)) .> p) .* x ./ (1-p)
    α = attention_weights(q, k; fdrop = x -> fdrop(x, 0.5))
    @test 0.6 > mean(>(0), α) > 0.4
end

@testset "bias" begin
    q = rand(2, 2, 5, 1)
    k = v = rand(2, 2, 3, 1)
    bias = randn(3, 5)
    y = scaled_dot_product_attention(q, k, v, bias)
    α = attention_weights(q, k, bias)
    @test size(α) == (3, 5, 2, 1)
    @test size(y) == (2, 2, 5, 1)
end

@testset "gradient" begin
    q = rand(2, 2, 5, 1)
    k = v = rand(2, 2, 3, 1)
    bias = randn(3, 5)
    @test test_gradients((x...) -> scaled_dot_product_attention(x...), q, k, v, bias)
end

@testset "scale" begin
    head_dim, nheads = 4, 2
    q = rand(Float32, head_dim, nheads, 5, 1)
    k = v = rand(Float32, head_dim, nheads, 3, 1)
    # passing the default scale explicitly reproduces the default
    α0 = attention_weights(q, k)
    α1 = attention_weights(q, k; scale=inv(sqrt(Float32(head_dim))))
    @test α1 ≈ α0
    # a different scale changes the result
    α2 = attention_weights(q, k; scale=1f0)
    @test !(α2 ≈ α0)
    # the output path accepts `scale` too
    @test size(scaled_dot_product_attention(q, k, v; scale=2f0)) == (head_dim, nheads, 5, 1)
end

@testset "is_causal" begin
    head_dim, nheads = 4, 2
    x = rand(Float32, head_dim, nheads, 4, 1)
    αcausal = attention_weights(x, x; is_causal=true)
    mask = make_causal_mask(x, dims=3)
    αmask = attention_weights(x, x; mask)
    @test αcausal ≈ αmask
    # lower triangle (key index > query index) must be masked out
    @test all(all(αcausal[j, i, :, 1] .== 0) for j in 1:4, i in 1:4 if j > i)
    # output path agrees
    @test scaled_dot_product_attention(x, x, x; is_causal=true) ≈
          scaled_dot_product_attention(x, x, x; mask)
    # `mask` and `is_causal` combine: a key is attended to where both allow it
    mask2 = rand(Bool, 4, 4)
    mask2[1, :] .= true
    αboth = attention_weights(x, x; mask=mask2, is_causal=true)
    @test all((αboth[:, :, h, 1] .> 0) == (mask2 .& mask) for h in 1:nheads)
    @test scaled_dot_product_attention(x, x, x; mask=mask2, is_causal=true) ≈
          scaled_dot_product_attention(x, x, x; mask=mask2 .& mask)
end

@testset "sequence lengths" begin
    head_dim, nheads, lenq, lenkv, bs = 4, 2, 5, 6, 3
    q = rand(Float32, head_dim, nheads, lenq, bs)
    k = rand(Float32, head_dim, nheads, lenkv, bs)
    v = rand(Float32, head_dim, nheads, lenkv, bs)
    q_lengths, kv_lengths = [5, 3, 1], [6, 2, 4]
    kvmask = reshape([j <= n for j in 1:lenkv, n in kv_lengths], lenkv, 1, 1, bs)
    qmask = reshape([i <= n for i in 1:lenq, n in q_lengths], 1, 1, lenq, bs)

    # the keys past `kv_lengths` are not attended to
    @test scaled_dot_product_attention(q, k, v; kv_lengths) ≈
          scaled_dot_product_attention(q, k, v; mask=kvmask)
    α = attention_weights(q, k; kv_lengths)
    @test (α .> 0) == (kvmask .& trues(lenkv, lenq, nheads, bs))

    # the output and the weights are zero at the queries past `q_lengths`
    y = scaled_dot_product_attention(q, k, v; q_lengths)
    @test y ≈ scaled_dot_product_attention(q, k, v) .* qmask
    @test all(iszero, y[:, :, 4:5, 2])
    @test attention_weights(q, k; q_lengths) ≈
          attention_weights(q, k) .* reshape(qmask, 1, lenq, 1, bs)

    # combined with each other, with `is_causal` and with a `mask`
    causal = NNlib.make_causal_mask(q, lenkv, lenq)
    @test scaled_dot_product_attention(q, k, v; q_lengths, kv_lengths, is_causal=true) ≈
          scaled_dot_product_attention(q, k, v; mask=causal .& kvmask) .* qmask
    mask = rand(Bool, lenkv, lenq)
    mask[1, :] .= true
    @test scaled_dot_product_attention(q, k, v; kv_lengths, mask) ≈
          scaled_dot_product_attention(q, k, v; mask=mask .& kvmask)

    # an integer is the length of every sequence, and any integer array is accepted
    @test scaled_dot_product_attention(q, k, v; kv_lengths=4) ≈
          scaled_dot_product_attention(q, k, v; kv_lengths=[4, 4, 4])
    @test scaled_dot_product_attention(q, k, v; kv_lengths=Int32.(kv_lengths)) ==
          scaled_dot_product_attention(q, k, v; kv_lengths)
    # so are integral numbers that are not `Integer`s, as the traced ones of Reactant
    @test scaled_dot_product_attention(q, k, v; q_lengths=Float32.(q_lengths), kv_lengths=4.0) ==
          scaled_dot_product_attention(q, k, v; q_lengths, kv_lengths=4)
    @test_throws InexactError scaled_dot_product_attention(q, k, v; kv_lengths=2.5)
    # lengths reaching the end of the sequences mean no padding
    @test scaled_dot_product_attention(q, k, v; q_lengths=lenq, kv_lengths=fill(lenkv + 2, bs)) ≈
          scaled_dot_product_attention(q, k, v)

    # one length for each batch element, whatever the batch dimensions
    qb, kb, vb = map(x -> reshape(x, size(x)[1:3]..., bs, 1), (q, k, v))
    yb = scaled_dot_product_attention(qb, kb, vb; q_lengths=reshape(q_lengths, bs, 1), kv_lengths)
    @test yb == reshape(scaled_dot_product_attention(q, k, v; q_lengths, kv_lengths), size(yb))
    @test scaled_dot_product_attention(q[:, :, :, 2], k[:, :, :, 2], v[:, :, :, 2]; kv_lengths=2) ≈
          scaled_dot_product_attention(q, k, v; kv_lengths=2)[:, :, :, 2]
    @test_throws ArgumentError scaled_dot_product_attention(q, k, v; kv_lengths=[6, 2])

    @inferred scaled_dot_product_attention(q, k, v; q_lengths, kv_lengths, is_causal=true)

    qd, kd, vd = map(x -> Float64.(x), (q, k, v))
    @test test_gradients((x...) -> scaled_dot_product_attention(x...; q_lengths, kv_lengths, is_causal=true),
                         qd, kd, vd)
end

@testset "query offset" begin
    head_dim, nheads, lenq, lenkv, bs = 4, 2, 3, 7, 3
    q = rand(Float32, head_dim, nheads, lenq, bs)
    k = rand(Float32, head_dim, nheads, lenkv, bs)
    v = rand(Float32, head_dim, nheads, lenkv, bs)
    # the query `i` attends to the keys `j ≤ q_offset + i`
    offset_mask(offsets) = reshape([j <= o + i for j in 1:lenkv, i in 1:lenq, o in offsets], lenkv, lenq, 1, bs)

    for q_offset in (2, [0, 4, 1])
        mask = offset_mask(q_offset isa Integer ? fill(q_offset, bs) : q_offset)
        @test attention_weights(q, k; is_causal=true, q_offset) ≈ attention_weights(q, k; mask)
        @test scaled_dot_product_attention(q, k, v; is_causal=true, q_offset) ≈
              scaled_dot_product_attention(q, k, v; mask)
    end
    # zero is the default, and without `is_causal` an offset has no effect
    @test scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=0) ==
          scaled_dot_product_attention(q, k, v; is_causal=true)
    @test scaled_dot_product_attention(q, k, v; q_offset=[0, 4, 1]) == scaled_dot_product_attention(q, k, v)
    # an offset from the last key on masks nothing, and a negative one leaves the first
    # queries without a key
    @test scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=lenkv) ≈
          scaled_dot_product_attention(q, k, v)
    y = scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=-1)
    @test all(isnan, y[:, :, 1, :])
    @test y[:, :, 2:end, :] ≈ scaled_dot_product_attention(q, k, v; mask=offset_mask(fill(-1, bs)))[:, :, 2:end, :]

    # integral numbers that are not `Integer`s are accepted, as the lengths are
    @test scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=2f0) ==
          scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=2)
    @test scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=Float64[0, 4, 1]) ==
          scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=[0, 4, 1])
    @test_throws InexactError scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=2.5)
    @test_throws ArgumentError scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=[1, 2])

    @inferred scaled_dot_product_attention(q, k, v; is_causal=true, q_offset=[0, 4, 1], kv_lengths=[3, 7, 5])
    @test NNlib.is_fused_attention(q, k, v; is_causal=true, q_offset=[0, 4, 1])

    qd, kd, vd = map(x -> Float64.(x), (q, k, v))
    kws = (; is_causal=true, q_offset=[0, 4, 1], q_lengths=[3, 3, 2], kv_lengths=[3, 7, 5])
    @test test_gradients((x...) -> scaled_dot_product_attention(x...; kws...), qd, kd, vd)
end

@testset "bottom-right causal mask" begin
    head_dim, nheads, lenq, lenkv, bs = 4, 2, 3, 7, 3
    q = rand(Float32, head_dim, nheads, lenq, bs)
    k = rand(Float32, head_dim, nheads, lenkv, bs)
    v = rand(Float32, head_dim, nheads, lenkv, bs)
    q_lengths, kv_lengths = [3, 2, 1], [7, 4, 5]
    with_weights(args...; kws...) = first(scaled_dot_product_attention_with_weights(args...; kws...))
    # the queries are the last positions of the keys: the offsets come from the sizes of
    # the arrays, or from the lengths where given
    for f in (scaled_dot_product_attention, materialized_attention, with_weights)
        @test f(q, k, v; is_causal=:bottom_right) ≈ f(q, k, v; is_causal=true, q_offset=lenkv - lenq)
        @test f(q, k, v; is_causal=:bottom_right, q_lengths, kv_lengths) ≈
              f(q, k, v; is_causal=true, q_offset=kv_lengths .- q_lengths, q_lengths, kv_lengths)
        @test f(q, k, v; is_causal=:bottom_right, kv_lengths) ≈
              f(q, k, v; is_causal=true, q_offset=kv_lengths .- lenq, kv_lengths)
    end
    @test attention_weights(q, k; is_causal=:bottom_right) ≈
          attention_weights(q, k; is_causal=true, q_offset=lenkv - lenq)
    # as many queries as keys: the usual causal mask
    @test scaled_dot_product_attention(k, k, v; is_causal=:bottom_right) ==
          scaled_dot_product_attention(k, k, v; is_causal=true)
    # more queries than keys: the first ones attend to no key
    q9 = rand(Float32, head_dim, nheads, 9, bs)
    y = scaled_dot_product_attention(q9, k, v; is_causal=:bottom_right)
    @test all(isnan, y[:, :, 1:2, :])
    @test y[:, :, 3:end, :] ≈ materialized_attention(q9, k, v; is_causal=true, q_offset=-2)[:, :, 3:end, :]

    @test_throws ArgumentError scaled_dot_product_attention(q, k, v; is_causal=:top_left)
    @test_throws ArgumentError scaled_dot_product_attention(q, k, v; is_causal=:bottom_right, q_offset=1)
    @test_throws ArgumentError scaled_dot_product_attention_with_weights(q, k, v; is_causal=:bottom_right, q_offset=[1, 2, 3])
    @test_throws ArgumentError NNlib.is_fused_attention(q, k, v; is_causal=:upper_left)

    @inferred scaled_dot_product_attention(q, k, v; is_causal=:bottom_right, q_lengths, kv_lengths)
    @test NNlib.is_fused_attention(q, k, v; is_causal=:bottom_right, kv_lengths)

    qd, kd, vd = map(x -> Float64.(x), (q, k, v))
    @test test_gradients((x...) -> scaled_dot_product_attention(x...; is_causal=:bottom_right, q_lengths, kv_lengths),
                         qd, kd, vd)
end

# A chunk of new queries attending to a cache of keys, partly filled, gives the rows of
# the causal attention over the whole sequences that these queries end.
@testset "KV cache" begin
    head_dim, nheads, nkvheads, len, bs = 8, 4, 2, 300, 3
    x = randn(Float32, head_dim, nheads, len, bs) ./ 2
    kx = randn(Float32, head_dim, nkvheads, len, bs) ./ 2
    vx = randn(Float32, head_dim, nkvheads, len, bs)
    y = materialized_attention(x, kx, vx; is_causal=true)

    cached, new = [0, 150, 280], [5, 20, 1]
    kv_lengths = cached .+ new
    q = zeros(Float32, head_dim, nheads, maximum(new), bs)
    k, v = copy(kx), copy(vx)
    for b in 1:bs
        q[:, :, 1:new[b], b] = x[:, :, cached[b] .+ (1:new[b]), b]
        # what the cache holds past the valid keys is not attended to
        k[:, :, (kv_lengths[b] + 1):end, b] .= 100
        v[:, :, (kv_lengths[b] + 1):end, b] .= 100
    end
    for f in (scaled_dot_product_attention, materialized_attention),
            causal in ((; is_causal=true, q_offset=cached), (; is_causal=:bottom_right))
        yc = f(q, k, v; causal..., q_lengths=new, kv_lengths)
        for b in 1:bs
            @test yc[:, :, 1:new[b], b] ≈ y[:, :, cached[b] .+ (1:new[b]), b] rtol=1f-4
            @test all(iszero, yc[:, :, (new[b] + 1):end, b])
        end
    end
    # the offsets of a cache whose valid lengths are those of the arrays
    q1 = x[:, :, (len - 3):len, :]
    @test scaled_dot_product_attention(q1, kx, vx; is_causal=true, q_offset=len - 4) ≈
          y[:, :, (len - 3):len, :] rtol=1f-4
end

# Packed sequences attend as their segments do one by one.
function segment_attention(f, q, k, v, q_segments, kv_segments=q_segments;
                           q_offset=0, q_lengths=nothing, kv_lengths=nothing, kws...)
    qe, ke = cumsum(q_segments), cumsum(kv_segments)
    ys = map(eachindex(q_segments)) do s
        rq, rk = (qe[s] - q_segments[s] + 1):qe[s], (ke[s] - kv_segments[s] + 1):ke[s]
        # an empty array has no batch size for `reshape(x, ..., :)` to find
        isempty(rq) && return similar(v, size(v, 1), size(q, 2), 0)
        at(x) = x isa AbstractArray ? x[s] : x
        f(q[:, :, rq], k[:, :, rk], v[:, :, rk]; q_offset=at(q_offset), q_lengths=at(q_lengths),
          kv_lengths=at(kv_lengths), kws...)
    end
    return cat(ys...; dims=3)
end

# the same, up to rounding, with NaN where the other is
same_attention(y, yref; rtol) =
    isequal(isnan.(y), isnan.(yref)) && isapprox(y[.!isnan.(y)], yref[.!isnan.(yref)]; rtol)

@testset "packed sequences" begin
    head_dim, v_head_dim, nheads, nkvheads = 8, 6, 4, 2
    with_weights(args...; kws...) = first(scaled_dot_product_attention_with_weights(args...; kws...))
    # segments of the queries and of the keys of the same or different lengths, empty
    # ones, a long segment among short ones and longer than a tile of keys
    @testset "segments $q_segments, $kv_segments" for (q_segments, kv_segments) in
            (([5, 12, 3], [5, 12, 3]), ([70, 0, 200, 31], [90, 4, 150, 31]), ([2, 300, 1], [3, 400, 2]))
        n = length(q_segments)
        q = randn(Float32, head_dim, nheads, sum(q_segments)) ./ 2
        k = randn(Float32, head_dim, nkvheads, sum(kv_segments)) ./ 2
        v = randn(Float32, v_head_dim, nkvheads, sum(kv_segments))
        segs = (; q_segments, kv_segments)
        @test NNlib.is_fused_attention(q, k, v; segs...)
        @test !NNlib.is_fused_attention(q, k, v; fdrop = x -> x, segs...)

        for kws in ((;), (; is_causal=true), (; is_causal=:bottom_right), (; scale=0.3f0),
                    (; is_causal=true, q_offset=[1, 0, 2, 5][1:n]),
                    (; is_causal=:bottom_right, kv_lengths=[3, 0, 100, 7][1:n]),
                    (; is_causal=true, q_lengths=[1, 0, 50, 2][1:n]))
            yref = segment_attention(materialized_attention, q, k, v, q_segments, kv_segments; kws...)
            for f in (scaled_dot_product_attention, materialized_attention, with_weights)
                y = f(q, k, v; segs..., kws...)
                @test size(y) == (v_head_dim, nheads, sum(q_segments))
                @test same_attention(y, yref; rtol=1f-4)
            end
        end

        # the weights are those of the segments, and zero elsewhere
        α = attention_weights(q, k; is_causal=true, segs...)
        @test size(α) == (sum(kv_segments), sum(q_segments), nheads)
        qe, ke = cumsum(q_segments), cumsum(kv_segments)
        for s in findall(>(0), q_segments)
            rq, rk = (qe[s] - q_segments[s] + 1):qe[s], (ke[s] - kv_segments[s] + 1):ke[s]
            @test α[rk, rq, :] ≈ attention_weights(q[:, :, rq], k[:, :, rk]; is_causal=true)
            @test all(iszero, α[setdiff(1:end, rk), rq, :])
        end

        # the gradients, of the fused implementation and of the materialized one
        w = randn(Float32, v_head_dim, nheads, sum(q_segments))
        for kws in ((;), (; is_causal=true), (; is_causal=true, q_offset=1, kv_lengths=max.(kv_segments .- 1, 1)))
            gref = attention_gradients((x...; kws...) -> segment_attention(materialized_attention, x...,
                                                                           q_segments, kv_segments; kws...),
                                       w, q, k, v; kws...)
            for f in (scaled_dot_product_attention, materialized_attention)
                @test all(isapprox.(attention_gradients(f, w, q, k, v; segs..., kws...), gref; rtol=1f-3))
            end
        end
    end

    x = randn(Float32, head_dim, nheads, 20)
    q_segments = [5, 12, 3]
    y = scaled_dot_product_attention(x, x, x; is_causal=true, q_segments)
    # the segments of the keys are those of the queries, and a single one is a sequence
    @test y == scaled_dot_product_attention(x, x, x; is_causal=true, q_segments, kv_segments=q_segments)
    @test scaled_dot_product_attention(x, x, x; is_causal=true, q_segments=[20]) ≈
          scaled_dot_product_attention(x, x, x; is_causal=true)
    @inferred scaled_dot_product_attention(x, x, x; is_causal=true, q_segments)

    # The materialized implementation pads the segments into a batch, unless they have
    # lengths so different that the padding outweighs attending over the whole sequence.
    packed(segments) = NNlib.attention_segments(reshape(x, size(x)..., 1), reshape(x, size(x)..., 1),
                                                segments, segments, nothing)
    cheaper(segments) = NNlib.padded_segments_cheaper(reshape(x, size(x)..., 1), reshape(x, size(x)..., 1),
                                                      packed(segments))
    @test cheaper([7, 6, 7]) && !cheaper([1, 18, 1])
    for q_segments in ([7, 6, 7], [1, 18, 1])
        @test materialized_attention(x, x, x; is_causal=true, q_segments) ≈
              segment_attention(materialized_attention, x, x, x, q_segments; is_causal=true)
    end

    # Lengths of segments that are not read on the host, as traced ones: the fused
    # kernels need `maxlen` for their launches, without it the materialized implementation
    # attends over the whole sequence.
    traced = view(Float64.(q_segments), :)
    @test NNlib.is_fused_attention(x, x, x; q_segments=traced, maxlen=12)
    @test !NNlib.is_fused_attention(x, x, x; q_segments=traced)
    @test !cheaper(traced)
    @test scaled_dot_product_attention(x, x, x; is_causal=true, q_segments=traced, maxlen=12) ≈ y
    @test scaled_dot_product_attention(x, x, x; is_causal=true, q_segments=traced) ≈ y
    # `maxlen` is not needed where the lengths are read
    @test scaled_dot_product_attention(x, x, x; is_causal=true, q_segments, maxlen=100) == y

    # inputs with a batch dimension, a bias, a mask, lengths that do not add up to the
    # sequence, keys without queries, as many entries as segments
    @test_throws ArgumentError scaled_dot_product_attention(reshape(x, size(x)..., 1), x, x; q_segments)
    @test_throws ArgumentError scaled_dot_product_attention(x, x, x, randn(Float32, 20, 20); q_segments)
    @test_throws ArgumentError scaled_dot_product_attention(x, x, x; mask=trues(20, 20), q_segments)
    @test_throws ArgumentError scaled_dot_product_attention(x, x, x; q_segments=[5, 12, 4])
    @test_throws ArgumentError scaled_dot_product_attention(x, x, x; q_segments=[5, 16, -1])
    @test_throws ArgumentError scaled_dot_product_attention(x, x, x; kv_segments=q_segments)
    @test_throws ArgumentError scaled_dot_product_attention(x, x, x; q_segments, kv_segments=[20])
    @test_throws ArgumentError scaled_dot_product_attention(x, x, x; q_segments, kv_lengths=[1, 2])
    @test_throws ArgumentError scaled_dot_product_attention_with_weights(x, x, x; q_segments=[5, 12, 4])

    qd, kd = Float64.(x[:, :, 1:9]), Float64.(x[:, :, 1:12])
    @test test_gradients((x...) -> scaled_dot_product_attention(x...; is_causal=true, q_segments=[2, 4, 3]),
                         qd, qd, qd)
    @test test_gradients((x...) -> scaled_dot_product_attention(x...; is_causal=:bottom_right, q_segments=[2, 4, 3],
                                                                kv_segments=[3, 5, 4], kv_lengths=[3, 4, 4]),
                         qd, kd, kd)
end

@testset "sliding window" begin
    head_dim, nheads, lenq, lenkv, bs = 4, 2, 5, 9, 3
    q = rand(Float32, head_dim, nheads, lenq, bs)
    k = rand(Float32, head_dim, nheads, lenkv, bs)
    v = rand(Float32, head_dim, nheads, lenkv, bs)
    with_weights(args...; kws...) = first(scaled_dot_product_attention_with_weights(args...; kws...))
    # the query `i` attends to the keys `j` with `|q_offset + i - j| < window`, those up to
    # `q_offset + i` with a causal mask
    window_mask(offsets, window, causal) =
        reshape([abs(o + i - j) < window && (!causal || j <= o + i) for j in 1:lenkv, i in 1:lenq, o in offsets],
                lenkv, lenq, 1, bs)

    for is_causal in (true, false), window in (1, 3, 20), q_offset in (0, 4, [4, 2, 3])
        mask = window_mask(q_offset isa Integer ? fill(q_offset, bs) : q_offset, window, is_causal)
        kws = (; is_causal, window, q_offset)
        @test attention_weights(q, k; kws...) ≈ attention_weights(q, k; mask)
        for f in (scaled_dot_product_attention, materialized_attention, with_weights)
            @test f(q, k, v; kws...) ≈ scaled_dot_product_attention(q, k, v; mask)
        end
    end
    # a window as long as the keys masks nothing more
    @test scaled_dot_product_attention(q, k, v; is_causal=true, window=lenkv) ≈
          scaled_dot_product_attention(q, k, v; is_causal=true)
    @test scaled_dot_product_attention(q, k, v; window=lenkv) ≈ scaled_dot_product_attention(q, k, v)

    # with the bottom-right alignment of a batch of caches, and the padding
    q_lengths, kv_lengths = [5, 3, 1], [9, 6, 7]
    mask = window_mask(kv_lengths .- q_lengths, 3, true) .& reshape((1:lenkv) .<= kv_lengths', lenkv, 1, 1, bs)
    for f in (scaled_dot_product_attention, materialized_attention, with_weights)
        @test f(q, k, v; is_causal=:bottom_right, window=3, q_lengths, kv_lengths) ≈
              scaled_dot_product_attention(q, k, v; mask, q_lengths)
    end
    # A padding query past the window of every valid key attends to no key, and its output
    # is NaN, unless `q_lengths` makes it zero.
    for f in (scaled_dot_product_attention, materialized_attention)
        y = f(q, k, v; is_causal=true, window=2, kv_lengths=[2, 9, 9])
        @test all(isnan, y[:, :, 4:end, 1]) && !any(isnan, y[:, :, 1:3, :])
        y = f(q, k, v; is_causal=true, window=2, q_lengths=[3, 5, 5], kv_lengths=[2, 9, 9])
        @test all(iszero, y[:, :, 4:end, 1])
    end
    # ... and the padding takes no part in the gradients, which are finite in the
    # materialized implementation too, unpacked or packed
    w = randn(Float32, head_dim, nheads, lenq, bs)
    kws = (; is_causal=true, window=2, q_lengths=[3, 5, 5], kv_lengths=[2, 9, 9])
    gm = attention_gradients(materialized_attention, w, q, k, v; kws...)
    @test all(g -> all(isfinite, g), gm)
    @test all(isapprox.(attention_gradients(scaled_dot_product_attention, w, q, k, v; kws...), gm; rtol=1f-4))
    xs = map(x -> reshape(x, head_dim, nheads, :), (q, k, v))
    kws = (; is_causal=true, window=2, q_segments=[5, 5, 5], kv_segments=[9, 9, 9], q_lengths=[3, 5, 5],
           kv_lengths=[2, 9, 9])
    gm = attention_gradients(materialized_attention, reshape(w, head_dim, nheads, :), xs...; kws...)
    @test all(g -> all(isfinite, g), gm)

    # within each segment of packed sequences
    x = rand(Float32, head_dim, nheads, 12)
    for f in (scaled_dot_product_attention, materialized_attention, with_weights), is_causal in (true, false)
        @test f(x, x, x; is_causal, window=2, q_segments=[5, 4, 3]) ≈
              segment_attention(f, x, x, x, [5, 4, 3]; is_causal, window=2)
    end

    @test_throws ArgumentError scaled_dot_product_attention(q, k, v; window=0)
    @test_throws ArgumentError scaled_dot_product_attention_with_weights(q, k, v; window=-2)
    @test_throws ArgumentError NNlib.is_fused_attention(q, k, v; window=2.0)

    @inferred scaled_dot_product_attention(q, k, v; is_causal=true, window=3, q_offset=[0, 4, 1])
    @test NNlib.is_fused_attention(q, k, v; window=3)

    qd, kd, vd = map(x -> Float64.(x), (q, k, v))
    kws = (; is_causal=:bottom_right, window=3, q_lengths, kv_lengths)
    @test test_gradients((x...) -> scaled_dot_product_attention(x...; kws...), qd, kd, vd)
    @test test_gradients((x...) -> scaled_dot_product_attention(x...; window=2, q_offset=[0, 4, 1]), qd, kd, vd)
end

@testset "bias and mask with several batch dimensions" begin
    head_dim, nheads, lenq, lenkv = 4, 2, 3, 5
    batch_size = (2, 3)
    q = rand(Float32, head_dim, nheads, lenq, batch_size...)
    k = rand(Float32, head_dim, nheads, lenkv, batch_size...)
    v = rand(Float32, head_dim, nheads, lenkv, batch_size...)
    flat(x) = reshape(x, size(x)[1:3]..., :)
    yref(; kws...) = reshape(scaled_dot_product_attention(flat(q), flat(k), flat(v); kws...), size(q))

    # trailing dimensions equal to the batch ones, a prefix of them, with singletons,
    # or already collapsed
    @testset "trailing dimensions $sz" for sz in ((2, 3), (2,), (2, 1), (1, 3), (1, 1), (6,))
        full(x) = sz == (6,) ? x : flat(x .+ zeros(Bool, 1, 1, 1, batch_size...))

        mask = rand(Bool, lenkv, lenq, 1, sz...)
        selectdim(mask, 1, 1) .= true
        @test scaled_dot_product_attention(q, k, v; mask) ≈ yref(mask=Bool.(full(mask)))
        α = attention_weights(q, k; mask)
        @test flat(α .> 0) == (full(mask) .> 0) .& trues(lenkv, lenq, nheads, 6)

        bias = randn(Float32, lenkv, lenq, 1, sz...)
        @test scaled_dot_product_attention(q, k, v, bias) ≈
              reshape(scaled_dot_product_attention(flat(q), flat(k), flat(v), full(bias)), size(q))
    end

    @test_throws DimensionMismatch scaled_dot_product_attention(q, k, v; mask=trues(lenkv, lenq, 1, 3, 2))
    @test_throws DimensionMismatch scaled_dot_product_attention(q, k, v, zeros(Float32, lenkv, lenq, 1, 4))
    @test_throws DimensionMismatch scaled_dot_product_attention(q, k, v; mask=trues(lenkv, lenq, 1, 2, 3, 1, 2))

    @inferred scaled_dot_product_attention(q, k, v, zeros(Float32, lenkv, lenq, 1, 2, 3); mask=trues(lenkv, lenq, 1, 2, 1))

    qd, kd, vd = map(x -> Float64.(x), (q, k, v))
    for sz in ((2, 3), (2, 1))
        @test test_gradients((x...) -> scaled_dot_product_attention(x...), qd, kd, vd, randn(lenkv, lenq, 1, sz...))
    end
end

@testset "input validation" begin
    q = rand(Float32, 4, 2, 3, 2)
    k = v = rand(Float32, 4, 2, 5, 2)
    # arrays with different numbers of dimensions
    @test_throws ArgumentError scaled_dot_product_attention(q, k[:, :, :, 1], v)
    @test_throws ArgumentError scaled_dot_product_attention(q, k, v[:, :, :, 1])
    @test_throws ArgumentError scaled_dot_product_attention_with_weights(q, k[:, :, :, 1], v)
    @test_throws ArgumentError scaled_dot_product_attention(q[:, :, 1, 1], k[:, :, 1, 1], v[:, :, 1, 1])
    # head and batch dimensions
    @test_throws ArgumentError scaled_dot_product_attention(q, k[1:2, :, :, :], v)
    @test_throws ArgumentError scaled_dot_product_attention(q, k, v[:, :, :, 1:1])
    @test_throws ArgumentError scaled_dot_product_attention_with_weights(q, k[:, :, :, 1:1], v)
end

@testset "is_fused_attention" begin
    q = rand(Float32, 8, 2, 3, 2)
    @test NNlib.is_fused_attention(q, q, q)
    @test NNlib.is_fused_attention(q, q, q; is_causal=true, kv_lengths=[3, 2])
    @test NNlib.is_fused_attention(q, q, q, rand(Float32, 3, 3); mask=trues(3, 3))
    @test NNlib.is_fused_attention(Float64.(q), Float64.(q), Float64.(q), rand(3, 3))
    # dropout acts on the weights, which are not formed
    @test !NNlib.is_fused_attention(q, q, q; fdrop = x -> x)
    # mixed element types, a bias that promotes the output, arrays that are not dense
    @test !NNlib.is_fused_attention(q, q, Float64.(q))
    @test !NNlib.is_fused_attention(q, q, q, rand(3, 3))
    @test !NNlib.is_fused_attention(view(q, :, :, :, 1:2), q, q)
    @test !NNlib.is_fused_attention(q, q, q; mask=view(trues(3, 3), :, :))
    @test_throws ArgumentError NNlib.is_fused_attention(q, q[:, :, :, 1], q)
end

@testset "fused implementation" begin
    @testset "$T" for T in (Float32, Float64, Float16)
        # the half-precision reference accumulates in `Float16`
        rtol = T == Float16 ? 2f-2 : sqrt(eps(T))
        # more keys than a tile holds, an odd number of queries
        head_dim, v_head_dim, nheads, lenq, lenkv, bs = 10, 7, 4, 37, 301, 2
        q = randn(T, head_dim, nheads, lenq, bs) ./ 2
        k = randn(T, head_dim, nheads, lenkv, bs) ./ 2
        v = randn(T, v_head_dim, nheads, lenkv, bs)
        @test NNlib.is_fused_attention(q, k, v)

        y = @inferred scaled_dot_product_attention(q, k, v)
        @test y isa Array{T,4}
        @test size(y) == (v_head_dim, nheads, lenq, bs)
        @test y ≈ materialized_attention(q, k, v) rtol=rtol
        @test y ≈ first(scaled_dot_product_attention_with_weights(q, k, v)) rtol=rtol
        @test scaled_dot_product_attention(q, k, v; scale=0.3) ≈ materialized_attention(q, k, v; scale=0.3) rtol=rtol
        @test scaled_dot_product_attention(q, k, v; is_causal=true) ≈
              materialized_attention(q, k, v; is_causal=true) rtol=rtol
        @test NNlib.@disallow_spawns(scaled_dot_product_attention(q, k, v)) ≈ y

        lengths = (; q_lengths=[37, 12], kv_lengths=[150, 301])
        yl = scaled_dot_product_attention(q, k, v; lengths...)
        @test yl ≈ materialized_attention(q, k, v; lengths...) rtol=rtol
        @test all(iszero, yl[:, :, 13:end, 2])

        mask = rand(Bool, lenkv, lenq, 1, bs)
        mask[1, :, :, :] .= true
        bias = randn(T, lenkv, lenq)
        @test scaled_dot_product_attention(q, k, v; mask) ≈ materialized_attention(q, k, v; mask) rtol=rtol
        @test scaled_dot_product_attention(q, k, v; mask=BitArray(mask)) ≈
              materialized_attention(q, k, v; mask) rtol=rtol
        @test scaled_dot_product_attention(q, k, v, bias) ≈ materialized_attention(q, k, v, bias) rtol=rtol
        @test scaled_dot_product_attention(q, k, v, bias; mask, is_causal=true, lengths...) ≈
              materialized_attention(q, k, v, bias; mask, is_causal=true, lengths...) rtol=rtol

        # a causal mask shifted by an offset, which ends inside the tiles of keys
        for q_offset in (lenkv - lenq, [100, 3])
            @test scaled_dot_product_attention(q, k, v; is_causal=true, q_offset) ≈
                  materialized_attention(q, k, v; is_causal=true, q_offset) rtol=rtol
            @test scaled_dot_product_attention(q, k, v, bias; mask, is_causal=true, q_offset, lengths...) ≈
                  materialized_attention(q, k, v, bias; mask, is_causal=true, q_offset, lengths...) rtol=rtol
        end
        @test scaled_dot_product_attention(q, k, v; is_causal=:bottom_right, lengths...) ≈
              materialized_attention(q, k, v; is_causal=:bottom_right, lengths...) rtol=rtol

        # sliding windows, which start and end inside the tiles of keys
        for window in (1, 50, 200), is_causal in (true, false)
            kws = (; is_causal, window, q_offset=lenkv - lenq)
            @test scaled_dot_product_attention(q, k, v; kws...) ≈ materialized_attention(q, k, v; kws...) rtol=rtol
        end
        @test scaled_dot_product_attention(q, k, v, bias; is_causal=:bottom_right, window=40, lengths...) ≈
              materialized_attention(q, k, v, bias; is_causal=:bottom_right, window=40, lengths...) rtol=rtol

        # grouped-query attention
        kg, vg = k[:, 1:2, :, :], v[:, 1:2, :, :]
        @test scaled_dot_product_attention(q, kg, vg; is_causal=true) ≈
              materialized_attention(q, kg, vg; is_causal=true) rtol=rtol
        @test scaled_dot_product_attention(q, kg, vg; is_causal=true, q_offset=[100, 3]) ≈
              materialized_attention(q, kg, vg; is_causal=true, q_offset=[100, 3]) rtol=rtol
        @test scaled_dot_product_attention(q, kg, vg; window=50, q_offset=[100, 3]) ≈
              materialized_attention(q, kg, vg; window=50, q_offset=[100, 3]) rtol=rtol
    end

    # self-attention over several tiles of keys, where the causal mask ends inside them
    x = randn(Float32, 6, 2, 300, 1)
    @test scaled_dot_product_attention(x, x, x; is_causal=true) ≈
          materialized_attention(x, x, x; is_causal=true)

    # a query attending to no key gives NaN, as in the materialized implementation
    q = rand(Float32, 4, 2, 3, 1)
    k = rand(Float32, 4, 2, 5, 1)
    mask = trues(5, 3)
    mask[:, 2] .= false
    y = scaled_dot_product_attention(q, k, k; mask)
    @test all(isnan, y[:, :, 2, :])
    @test y[:, :, [1, 3], :] ≈ materialized_attention(q, k, k; mask)[:, :, [1, 3], :]

    # ... and so does a sequence without a valid key
    k2 = rand(Float32, 4, 2, 5, 2)
    q2 = rand(Float32, 4, 2, 3, 2)
    y = scaled_dot_product_attention(q2, k2, k2; kv_lengths=[0, 4])
    @test all(isnan, y[:, :, :, 1])
    @test y[:, :, :, 2] ≈ materialized_attention(q2, k2, k2; kv_lengths=[5, 4])[:, :, :, 2]

    # ... and the queries a negative offset leaves without a key
    y = scaled_dot_product_attention(q2, k2, k2; is_causal=true, q_offset=[-2, -5])
    ym = materialized_attention(q2, k2, k2; is_causal=true, q_offset=[-2, -5])
    @test all(isnan, y[:, :, 1:2, 1]) && all(isnan, y[:, :, :, 2])
    @test y[:, :, 3, 1] ≈ ym[:, :, 3, 1]

    # the tilings of the GPU kernel are plain arithmetic: each has to fit the shared
    # memory it is given, whatever the head dimension, the problem size and the eltype
    @testset "kernel tilings" begin
        for budget in (31 * 1024, 47 * 1024), bytes in (4, 8), nqueries in (10, 10^6),
                v_head_dim in (1, 5, 64, 65, 128, 200, 256, 257, 512, 600, 1024, 2048)
            c = NNlib.attention_tiling(v_head_dim, nqueries, budget ÷ bytes)
            @test c !== nothing
            @test NNlib.attention_shmem(c) * bytes <= budget
            @test c.T * c.TM * c.TN == c.BQ * c.BK      # the logits of a tile
            @test c.NR * c.NC == c.T == c.ND * c.NQ
            @test c.TD * c.ND >= v_head_dim             # the outputs of a tile of queries
            @test c.TD * c.TQ <= 64 && c.TM * c.TN <= 64
            @test c.T % c.KD == 0 && c.BK % c.KV == 0
        end
        @test NNlib.attention_tiling(4096, 10^6, 47 * 1024 ÷ 4) === nothing
        @test NNlib.attention_tiling(64, 10^6, 100) === nothing
    end
end

@testset "fused gradient" begin
    # Under Zygote the fused implementation takes its rule, and the materialized one is
    # differentiated through.
    function same_gradients(w, xs...; rtol, kws...)
        g = attention_gradients(scaled_dot_product_attention, w, xs...; kws...)
        gm = attention_gradients(materialized_attention, w, xs...; kws...)
        return all(isapprox.(g, gm; rtol))
    end

    @testset "$T" for T in (Float32, Float64)
        rtol = T == Float32 ? 1f-3 : 1e-8
        # more keys than a tile holds, a number of queries that is not a multiple of four
        head_dim, v_head_dim, nheads, lenq, lenkv, bs = 10, 7, 4, 37, 301, 2
        q = randn(T, head_dim, nheads, lenq, bs) ./ 2
        k = randn(T, head_dim, nheads, lenkv, bs) ./ 2
        v = randn(T, v_head_dim, nheads, lenkv, bs)
        w = randn(T, v_head_dim, nheads, lenq, bs)
        mask = rand(Bool, lenkv, lenq, 1, bs)
        mask[1, :, :, :] .= true
        lengths = (; q_lengths=[37, 12], kv_lengths=[150, 301])

        @test same_gradients(w, q, k, v; rtol)
        @test same_gradients(w, q, k, v; rtol, is_causal=true)
        @test same_gradients(w, q, k, v; rtol, scale=0.3)
        @test same_gradients(w, q, k, v; rtol, lengths...)
        @test same_gradients(w, q, k, v; rtol, mask)
        @test same_gradients(w, q, k, v; rtol, mask=BitArray(mask), is_causal=true, lengths...)
        @test same_gradients(w, q, k[:, 1:2, :, :], v[:, 1:2, :, :]; rtol, is_causal=true)
        @test NNlib.@disallow_spawns same_gradients(w, q, k, v; rtol, is_causal=true)
        @test same_gradients(w, q, k, v; rtol, is_causal=true, q_offset=lenkv - lenq)
        @test same_gradients(w, q, k, v; rtol, mask, is_causal=true, q_offset=[100, 3], lengths...)
        @test same_gradients(w, q, k[:, 1:2, :, :], v[:, 1:2, :, :]; rtol, is_causal=true, q_offset=[100, 3])
        @test same_gradients(w, q, k, v; rtol, is_causal=:bottom_right, lengths...)
        @test same_gradients(w, q, k, v; rtol, is_causal=true, window=50, q_offset=lenkv - lenq)
        @test same_gradients(w, q, k, v; rtol, window=50, q_offset=[100, 3], lengths...)
        @test same_gradients(w, q, k[:, 1:2, :, :], v[:, 1:2, :, :]; rtol, is_causal=:bottom_right, window=30,
                             lengths...)

        dq, dk, dv = attention_gradients(scaled_dot_product_attention, w, q, k, v; lengths...)
        @test dq isa Array{T,4} && dk isa Array{T,4} && dv isa Array{T,4}
        # no gradient reaches the padding
        @test all(iszero, dq[:, :, 13:end, 2])
        @test all(iszero, dk[:, :, 151:end, 1]) && all(iszero, dv[:, :, 151:end, 1])

        # with respect to the scale, which can be zero
        for scale in (T(0.3), zero(T))
            ds = gradient(s -> sum(w .* scaled_dot_product_attention(q, k, v; scale=s)), scale)[1]
            @test ds ≈ gradient(s -> sum(w .* materialized_attention(q, k, v; scale=s)), scale)[1] rtol=rtol
        end
    end

    # Every pair of a range of queries and a range of keys is taken once, and the pairs of
    # a round share no range.
    for (nrq, nrk) in ((1, 1), (3, 3), (2, 5), (5, 2))
        pairs = [NNlib.attention_pair(x, round, nrq, nrk)
                 for round in 0:(max(nrq, nrk) - 1), x in 0:(min(nrq, nrk) - 1)]
        @test sort(vec(pairs)) == sort(vec([(t, u) for t in 0:(nrq - 1), u in 0:(nrk - 1)]))
        @test all(row -> allunique(first.(row)) && allunique(last.(row)), eachrow(pairs))
    end

    # Several ranges of queries and of keys, when there are threads to pair them in rounds.
    @testset "$lenq queries, $lenkv keys" for (lenq, lenkv) in ((700, 700), (300, 1100), (1100, 300), (128, 1100))
        q = randn(Float32, 8, 1, lenq, 1) ./ 2
        k = randn(Float32, 8, 1, lenkv, 1) ./ 2
        v = randn(Float32, 5, 1, lenkv, 1)
        w = randn(Float32, 5, 1, lenq, 1)
        @test same_gradients(w, q, k, v; rtol=1f-3)
        @test same_gradients(w, q, k, v; rtol=1f-3, is_causal=true, q_lengths=[lenq - 5], kv_lengths=[lenkv - 9])
        @test same_gradients(w, q, k, v; rtol=1f-3, is_causal=true, q_offset=max(lenkv - lenq, 0) + 70)
        # windows that leave no query without a key
        if lenq <= lenkv
            @test same_gradients(w, q, k, v; rtol=1f-3, is_causal=:bottom_right, window=150)
            @test same_gradients(w, q, k, v; rtol=1f-3, window=150, q_offset=lenkv - lenq)
        end
    end

    # the same array as query, key and value
    x = randn(Float32, 6, 2, 300, 1)
    w = randn(Float32, 6, 2, 300, 1)
    @test gradient(x -> sum(w .* scaled_dot_product_attention(x, x, x; is_causal=true)), x)[1] ≈
          gradient(x -> sum(w .* materialized_attention(x, x, x; is_causal=true)), x)[1] rtol=1f-3

    q = randn(Float32, 8, 2, 20, 2) ./ 2
    k = randn(Float32, 8, 2, 30, 2) ./ 2
    v = randn(Float32, 6, 2, 30, 2)
    w = randn(Float32, 6, 2, 20, 2)

    # half precision is accumulated in single precision
    half = map(x -> Float16.(x), (w, q, k, v))
    g16 = attention_gradients(scaled_dot_product_attention, half...; is_causal=true)
    g32 = attention_gradients(scaled_dot_product_attention, w, q, k, v; is_causal=true)
    @test all(x -> x isa Array{Float16,4}, g16)
    @test all(isapprox.(g16, g32; rtol=2f-2))

    # a cotangent that is not an `Array`
    @test gradient(q -> sum(scaled_dot_product_attention(q, k, v)), q)[1] ≈
          gradient(q -> sum(materialized_attention(q, k, v)), q)[1] rtol=1f-3

    # A query attending to no key has a NaN output, and takes no part in the gradients:
    # those of the other queries are what they are without it.
    mask = trues(30, 20)
    mask[:, 2] .= false
    dq, dk, dv = attention_gradients(scaled_dot_product_attention, w, q, k, v; mask)
    rest = setdiff(1:20, 2)
    dqr, dkr, dvr = attention_gradients(scaled_dot_product_attention, w[:, :, rest, :], q[:, :, rest, :], k, v)
    @test all(iszero, dq[:, :, 2, :])
    @test dq[:, :, rest, :] ≈ dqr
    @test dk ≈ dkr && dv ≈ dvr

    # ... so do those a negative offset leaves without a key
    dq, dk, dv = attention_gradients(scaled_dot_product_attention, w, q, k, v; is_causal=true, q_offset=-2)
    dqr, dkr, dvr = attention_gradients(scaled_dot_product_attention, w[:, :, 3:end, :], q[:, :, 3:end, :], k, v;
                                        is_causal=true, q_offset=0)
    @test all(iszero, dq[:, :, 1:2, :])
    @test dq[:, :, 3:end, :] ≈ dqr
    @test dk ≈ dkr && dv ≈ dvr

    # With a bias the materialized implementation is the one differentiated: the fused
    # one has no gradient with respect to it.
    bias = randn(Float32, 30, 20)
    @test NNlib.is_fused_attention(q, k, v, bias)
    @test gradient(b -> sum(w .* scaled_dot_product_attention(q, k, v, b)), bias)[1] ≈
          gradient(b -> sum(w .* materialized_attention(q, k, v, b)), bias)[1]
    @test_throws ArgumentError gradient(b -> sum(NNlib.fused_attention(q, k, v, b, nothing, nothing, false, 0, nothing,
                                                                       nothing, nothing, nothing)), bias)
end

if NNLIB_TEST_ENZYME

# Enzyme has rules of its own for the fused implementation: reverse mode calls its
# gradient, and forward mode differentiates the materialized implementation.
@testset "fused gradient: EnzymeRules" begin
    q = randn(3, 2, 4, 2) ./ 2
    k = randn(3, 2, 5, 2) ./ 2
    v = randn(2, 2, 5, 2)
    rest(Tscale) = ((nothing, EnzymeCore.Const), (nothing, EnzymeCore.Const), (0.3, Tscale),
                    (true, EnzymeCore.Const), ([1, 0], EnzymeCore.Const), (nothing, EnzymeCore.Const),
                    ([4, 2], EnzymeCore.Const), (nothing, EnzymeCore.Const), (3, EnzymeCore.Const))
    D, BD, C = EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated, EnzymeCore.Const

    for (Tret, Tq, Tk, Tv, Tscale) in ((D, D, D, D, EnzymeCore.Active), (D, C, D, D, C), (D, D, C, C, EnzymeCore.Active),
                                       (BD, BD, BD, BD, C), (C, D, D, D, C))
        EnzymeTestUtils.test_reverse(NNlib.fused_attention, Tret, (q, Tq), (k, Tk), (v, Tv), rest(Tscale)...;
                                     atol=1e-6, rtol=1e-6)
    end
    for (Tret, Tq, Tk, Tv) in ((D, D, D, D), (D, D, C, D), (BD, BD, BD, BD), (C, D, D, D))
        EnzymeTestUtils.test_forward(NNlib.fused_attention, Tret, (q, Tq), (k, Tk), (v, Tv), rest(C)...;
                                     atol=1e-6, rtol=1e-6)
    end

    # through the public function, the three arguments being the same array
    x = randn(3, 2, 4, 2)
    w = randn(3, 2, 4, 2)
    dx = zero(x)
    Enzyme.autodiff(Enzyme.Reverse, Const(x -> sum(w .* scaled_dot_product_attention(x, x, x; is_causal=true))),
                    Active, Duplicated(x, dx))
    @test dx ≈ gradient(x -> sum(w .* materialized_attention(x, x, x; is_causal=true)), x)[1]

    # Constant arguments next to active ones, as a cache of keys and values, in the fused
    # implementation, in the materialized one (with a bias) and with several batch dimensions.
    q, k, v = randn(3, 4, 2, 2), randn(3, 2, 4, 2), randn(2, 2, 4, 2)
    xs5 = (randn(3, 2, 2, 2, 2), randn(3, 2, 4, 2, 2), randn(2, 2, 4, 2, 2))
    xs3 = (randn(3, 4, 5), randn(3, 2, 6), randn(2, 2, 6))
    @testset "$name, active $act" for (name, xs, bias, kws) in (
                ("fused", (q, k, v), nothing, (; is_causal=:bottom_right, kv_lengths=[3, 4])),
                ("materialized", (q, k, v), randn(4, 2), (; is_causal=true)),
                ("5 dimensions", xs5, nothing, (;)),
                ("packed", xs3, nothing, (; is_causal=true, q_segments=[2, 3], kv_segments=[4, 2]))),
            act in ((true, false, false), (false, true, true))
        shadows = map(zero, xs)
        Enzyme.autodiff(Enzyme.Reverse, Const((xs...) -> sum(scaled_dot_product_attention(xs..., bias; kws...))),
                        Active, map((x, s, a) -> a ? Duplicated(x, s) : Const(x), xs, shadows, act)...)
        grads = gradient((xs...) -> sum(materialized_attention(xs..., bias; kws...)), xs...)
        @test all(i -> !act[i] || shadows[i] ≈ grads[i], 1:3)
    end
end

end # NNLIB_TEST_ENZYME

@testset "grouped-query attention" begin
    head_dim, nheads, nkvheads = 4, 6, 2
    q = rand(Float32, head_dim, nheads, 5, 2)
    k = rand(Float32, head_dim, nkvheads, 3, 2)
    v = rand(Float32, head_dim, nkvheads, 3, 2)

    y = scaled_dot_product_attention(q, k, v)
    α = attention_weights(q, k)
    @test size(y) == (head_dim, nheads, 5, 2)
    @test size(α) == (3, 5, nheads, 2)
    @test sum(α, dims=1) ≈ ones(Float32, 1, 5, nheads, 2)

    # GQA must match standard MHA with the kv heads expanded along the heads axis
    r = nheads ÷ nkvheads
    kf = repeat(k; inner=(1, r, 1, 1))
    vf = repeat(v; inner=(1, r, 1, 1))
    @test y ≈ scaled_dot_product_attention(q, kf, vf)
    @test α ≈ attention_weights(q, kf)

    # number of query heads must be divisible by the number of kv heads
    badk = rand(Float32, head_dim, 4, 3, 2)   # 6 % 4 != 0
    @test_throws ArgumentError scaled_dot_product_attention(q, badk, badk)

    qd = rand(head_dim, nheads, 5, 2)
    kd = rand(head_dim, nkvheads, 3, 2)
    vd = rand(head_dim, nkvheads, 3, 2)
    @test test_gradients((qq, kk, vv) -> scaled_dot_product_attention(qq, kk, vv), qd, kd, vd)
end
