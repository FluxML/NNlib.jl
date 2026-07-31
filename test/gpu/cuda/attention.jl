@testset "scaled dot-product attention" begin
    q = rand(Float32, 32, 2, 16, 1) ./ 4
    k = rand(Float32, 32, 2, 16, 1) ./ 4
    v = rand(Float32, 32, 2, 16, 1) ./ 4
    qd, kd, vd = cu(Float16.(q)), cu(Float16.(k)), cu(Float16.(v))

    @test Array(scaled_dot_product_attention(qd, kd, vd)) ≈
          scaled_dot_product_attention(q, k, v) rtol=2f-2
    @test Array(scaled_dot_product_attention(qd, kd, vd; is_causal=true)) ≈
          scaled_dot_product_attention(q, k, v; is_causal=true) rtol=2f-2
    @test Array(scaled_dot_product_attention(qd, kd, vd; scale=0.25f0)) ≈
          scaled_dot_product_attention(q, k, v; scale=0.25f0) rtol=2f-2

    mask = make_causal_mask(q)
    @test Array(scaled_dot_product_attention(qd, kd, vd; mask)) ≈
          scaled_dot_product_attention(q, k, v; mask) rtol=2f-2

    qg = rand(Float32, 32, 4, 16, 1) ./ 4
    kg = rand(Float32, 32, 2, 16, 1) ./ 4
    vg = rand(Float32, 32, 2, 16, 1) ./ 4
    @test Array(scaled_dot_product_attention(cu(Float16.(qg)), cu(Float16.(kg)),
                                              cu(Float16.(vg)))) ≈
          scaled_dot_product_attention(qg, kg, vg) rtol=2f-2

    qunsupported = cu(Float16.(rand(Float32, 60, 2, 16, 1) ./ 4))
    @test scaled_dot_product_attention(qunsupported, qunsupported, qunsupported) isa CuArray{Float16,4}
end

@testset "scaled dot-product attention gradient" begin
    q = rand(Float32, 32, 2, 16, 1) ./ 4
    k = rand(Float32, 32, 2, 16, 1) ./ 4
    v = rand(Float32, 32, 2, 16, 1) ./ 4
    qd, kd, vd = cu(Float16.(q)), cu(Float16.(k)), cu(Float16.(v))
    for kws in ((;), (; is_causal=true), (; scale=0.25f0))
        refgrads = gradient((q, k, v) -> sum(scaled_dot_product_attention(q, k, v; kws...)), q, k, v)
        grads = gradient((q, k, v) -> sum(scaled_dot_product_attention(q, k, v; kws...)), qd, kd, vd)
        for (g, refg) in zip(grads, refgrads)
            @test g isa CuArray{Float16}
            @test Array(g) ≈ refg rtol=5f-2
        end
    end

    qg = rand(Float32, 32, 4, 16, 1) ./ 4
    kg = rand(Float32, 32, 2, 16, 1) ./ 4
    vg = rand(Float32, 32, 2, 16, 1) ./ 4
    refgrads = gradient((q, k, v) -> sum(scaled_dot_product_attention(q, k, v)), qg, kg, vg)
    grads = gradient((q, k, v) -> sum(scaled_dot_product_attention(q, k, v)),
                     cu(Float16.(qg)), cu(Float16.(kg)), cu(Float16.(vg)))
    for (g, refg) in zip(grads, refgrads)
        @test g isa CuArray{Float16}
        @test Array(g) ≈ refg rtol=5f-2
    end

    qunsupported = cu(Float16.(rand(Float32, 60, 2, 16, 1) ./ 4))
    @test gradient(q -> sum(scaled_dot_product_attention(q, q, q)), qunsupported)[1] isa
          CuArray{Float16}
end
