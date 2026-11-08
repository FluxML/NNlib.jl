function sequence_scans_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    T = nameof(Backend) === :MetalBackend ? Float32 : Float64   # Metal has no Float64
    test_gpu = Backend != CPU
    rng = StableRNG(17)
    compare = test_gpu ? [AutoZygote(), AutoEnzyme()] : [AutoZygote(), AutoEnzyme(), AutoMooncake()]
    l2norm(x) = x ./ sqrt.(sum(abs2, x; dims=1))

    # the recurrences, token by token
    function ssd_reference(x, dt, A, B, C, D, state)
        P, h, n, b = size(x)
        N, r = size(B, 1), h ÷ size(B, 2)
        S = state === nothing ? zeros(eltype(x), P, N, h, b) : copy(state)
        y = similar(x)
        for i in 1:b, j in 1:h, t in 1:n
            G = (j - 1) ÷ r + 1
            S[:, :, j, i] = exp(dt[j, t, i] * A[j]) * S[:, :, j, i] + dt[j, t, i] * x[:, j, t, i] * B[:, G, t, i]'
            y[:, j, t, i] = S[:, :, j, i] * C[:, G, t, i] + (D === nothing ? 0 : D[j]) * x[:, j, t, i]
        end
        return y, S
    end

    function delta_reference(q, k, v, g, beta, state; scale=1 / sqrt(size(q, 1)))
        dk, hk, n, b = size(q)
        dv, h = size(v, 1), size(v, 2)
        S = state === nothing ? zeros(eltype(v), dv, dk, h, b) : copy(state)
        o = similar(v)
        for i in 1:b, j in 1:h, t in 1:n
            kt, qt = k[:, (j - 1) ÷ (h ÷ hk) + 1, t, i], q[:, (j - 1) ÷ (h ÷ hk) + 1, t, i]
            s = exp(g[j, t, i]) * S[:, :, j, i]
            S[:, :, j, i] = s + beta[j, t, i] * (v[:, j, t, i] - s * kt) * kt'
            o[:, j, t, i] = S[:, :, j, i] * (scale * qt)
        end
        return o, S
    end

    tokens(x, r, i=:) = x[ntuple(_ -> :, ndims(x) - 2)..., r, i]

    function ssd_inputs(P, h, N, G, n, b)
        x, dt, A = randn(rng, T, P, h, n, b), rand(rng, T, h, n, b), -rand(rng, T, h)
        B, C, D = randn(rng, T, N, G, n, b), randn(rng, T, N, G, n, b), randn(rng, T, h)
        return x, dt, A, B, C, D, randn(rng, T, P, N, h, b)
    end

    function delta_inputs(dk, hk, dv, h, n, b)
        q, k = l2norm(randn(rng, T, dk, hk, n, b)), l2norm(randn(rng, T, dk, hk, n, b))
        v, g, beta = randn(rng, T, dv, h, n, b), -rand(rng, T, h, n, b), rand(rng, T, h, n, b)
        return q, k, v, g, beta, randn(rng, T, dv, dk, h, b)
    end

    @testset "ssd_scan" begin
        x, dt, A, B, C, D, S0 = ssd_inputs(3, 4, 5, 2, 23, 2)
        ssd(x, dt, B, C; kws...) =
            cpu.(NNlib.ssd_scan(device.((x, dt, A, B, C))...; D=device(D), kws...))
        for G in (1, 2), D in (nothing, D), S0 in (nothing, S0)
            Bg, Cg = B[:, 1:G, :, :], C[:, 1:G, :, :]
            y_ref, S_ref = ssd_reference(x, dt, A, Bg, Cg, D, S0)
            for chunk_size in (4, 23, 64)
                y, S = cpu.(NNlib.ssd_scan(device.((x, dt, A, Bg, Cg))...; D=device(D),
                                           initial_state=device(S0), chunk_size))
                @test y ≈ y_ref
                @test S ≈ S_ref
            end
        end

        # in pieces, or token by token, from the state of the previous piece
        y, S = ssd(x, dt, B, C; initial_state=device(S0), chunk_size=4)
        y1, S1 = ssd(tokens.((x, dt, B, C), Ref(1:9))...; initial_state=device(S0), chunk_size=4)
        y2, S2 = ssd(tokens.((x, dt, B, C), Ref(10:23))...; initial_state=device(S1), chunk_size=4)
        @test cat(y1, y2; dims=3) ≈ y
        @test S2 ≈ S
        s = S0
        for t in 1:23
            yt, s = ssd(tokens.((x, dt, B, C), Ref(t:t))...; initial_state=device(s))
            @test yt ≈ y[:, :, t:t, :]
        end
        @test s ≈ S

        # in a batch padded on the right, the state at the end of each sequence
        for lengths in ([10, 23], device([10, 23]), [0, 1])
            y, S = ssd(x, dt, B, C; initial_state=device(S0), lengths, chunk_size=4)
            for (i, n) in enumerate(cpu(lengths))
                y_ref, S_ref = ssd_reference(tokens.((x, dt), Ref(1:n), Ref(i:i))..., A,
                                             tokens.((B, C), Ref(1:n), Ref(i:i))..., D,
                                             S0[:, :, :, i:i])
                @test y[:, :, 1:n, i:i] ≈ y_ref
                @test S[:, :, :, i:i] ≈ S_ref
            end
        end
    end

    @testset "gated_delta_rule" begin
        q, k, v, g, beta, S0 = delta_inputs(4, 2, 3, 4, 37, 2)
        delta(q, k, v, g, beta; kws...) =
            cpu.(NNlib.gated_delta_rule(device.((q, k, v, g, beta))...; kws...))
        # 1, 2 or 4 heads of queries and keys for the 4 heads of values
        for heads in (1:1, 1:2, [1, 1, 2, 2]), S0 in (nothing, S0)
            qh, kh = q[:, heads, :, :], k[:, heads, :, :]
            o_ref, S_ref = delta_reference(qh, kh, v, g, beta, S0)
            for chunk_size in (8, 64)
                o, S = delta(qh, kh, v, g, beta; initial_state=device(S0), chunk_size)
                @test o ≈ o_ref
                @test S ≈ S_ref
            end
        end
        @test delta(q, k, v, g, beta; scale=0.3)[1] ≈ delta_reference(q, k, v, g, beta, nothing; scale=0.3)[1]

        # in pieces, or token by token, from the state of the previous piece
        o, S = delta(q, k, v, g, beta; initial_state=device(S0), chunk_size=8)
        o1, S1 = delta(tokens.((q, k, v, g, beta), Ref(1:13))...; initial_state=device(S0), chunk_size=8)
        o2, S2 = delta(tokens.((q, k, v, g, beta), Ref(14:37))...; initial_state=device(S1), chunk_size=8)
        @test cat(o1, o2; dims=3) ≈ o
        @test S2 ≈ S
        s = S0
        for t in 1:37
            ot, s = delta(tokens.((q, k, v, g, beta), Ref(t:t))...; initial_state=device(s))
            @test ot ≈ o[:, :, t:t, :]
        end
        @test s ≈ S

        # in a batch padded on the right, the state at the end of each sequence
        for lengths in ([10, 37], device([10, 37]), [0, 1])
            o, S = delta(q, k, v, g, beta; initial_state=device(S0), lengths, chunk_size=8)
            for (i, n) in enumerate(cpu(lengths))
                o_ref, S_ref = delta_reference(tokens.((q, k, v, g, beta), Ref(1:n), Ref(i:i))...,
                                               S0[:, :, :, i:i])
                @test o[:, :, 1:n, i:i] ≈ o_ref
                @test S[:, :, :, i:i] ≈ S_ref
            end
        end
    end

    @testset "types and errors" begin
        x, dt, A, B, C, D, S0 = device.(ssd_inputs(3, 4, 5, 2, 9, 2))
        for n in (1, 9)
            y, S = @inferred NNlib.ssd_scan(Float32.(tokens(x, 1:n)), Float32.(tokens(dt, 1:n)),
                                            A, tokens(B, 1:n), tokens(C, 1:n); D)
            @test eltype(y) == Float32 && size(y) == (3, 4, n, 2) && size(S) == (3, 5, 4, 2)
        end
        y, S = NNlib.ssd_scan(Float16.(x), Float16.(dt), Float32.(A), Float16.(B), Float16.(C))
        @test eltype(y) == Float16 && eltype(S) == Float32
        @test cpu(y) ≈ cpu(NNlib.ssd_scan(Float32.(x), Float32.(dt), Float32.(A), Float32.(B), Float32.(C))[1]) rtol=1e-2
        @test_throws DimensionMismatch NNlib.ssd_scan(x, dt[1:3, :, :], A, B, C)
        @test_throws DimensionMismatch NNlib.ssd_scan(x, dt, A, B[:, :, 1:3, :], C[:, :, 1:3, :])
        @test_throws DimensionMismatch NNlib.ssd_scan(x, dt, A, B, C[1:4, :, :, :])
        @test_throws DimensionMismatch NNlib.ssd_scan(x[:, 1:3, :, :], dt[1:3, :, :], A[1:3], B, C)
        @test_throws DimensionMismatch NNlib.ssd_scan(x, dt, A, B, C; initial_state=S0[1:2, :, :, :])

        q, k, v, g, beta, S0 = device.(delta_inputs(4, 2, 3, 4, 9, 2))
        for n in (1, 9)
            o, S = @inferred NNlib.gated_delta_rule(Float32.(tokens(q, 1:n)), Float32.(tokens(k, 1:n)),
                                                    Float32.(tokens(v, 1:n)), tokens(g, 1:n), tokens(beta, 1:n))
            @test eltype(o) == Float32 && size(o) == (3, 4, n, 2) && size(S) == (3, 4, 4, 2)
        end
        o, S = NNlib.gated_delta_rule(Float32.(q), Float32.(k), Float16.(v), Float32.(g), Float32.(beta))
        @test eltype(o) == Float16 && eltype(S) == Float32
        @test_throws DimensionMismatch NNlib.gated_delta_rule(q, k[:, 1:1, :, :], v, g, beta)
        @test_throws DimensionMismatch NNlib.gated_delta_rule(q[:, 1:1, :, :], k[:, 1:1, :, :], v[:, 1:3, :, :], g, beta)
        @test_throws DimensionMismatch NNlib.gated_delta_rule(q, k, v, g[:, 1:3, :], beta)
        @test_throws DimensionMismatch NNlib.gated_delta_rule(q, k, v, g, beta; initial_state=S0[1:2, :, :, :])
        @test_throws ArgumentError NNlib.gated_delta_rule(q, k, v, g, beta; chunk_size=12)
    end

    # sizes of several tiles of the kernels of the GPU backends, against the CPU
    test_gpu && @testset "kernel tiles" begin
        function agree(f, args)
            out = f(args...)
            Δ = map(o -> randn(rng, T, size(o)), out)
            loss(Δ) = (args...) -> (o = f(args...); sum(o[1] .* Δ[1]) + sum(o[2] .* Δ[2]))
            @test all(cpu.(f(device.(args)...)) .≈ out)
            for (gd, gc) in zip(Zygote.gradient(loss(device.(Δ)), device.(args)...), Zygote.gradient(loss(Δ), args...))
                @test cpu(gd) ≈ gc
            end
        end
        x, dt, A, B, C, D, S0 = ssd_inputs(70, 4, 130, 2, 300, 2)
        agree((x, dt, A, B, C, D, S0)) do x, dt, A, B, C, D, S0
            NNlib.ssd_scan(x, dt, A, B, C; D, initial_state=S0, chunk_size=128)
        end
        q, k, v, g, beta, S0 = delta_inputs(130, 2, 70, 4, 150, 2)
        agree((q, k, v, g, beta, S0)) do q, k, v, g, beta, S0
            NNlib.gated_delta_rule(q, k, v, g, beta; initial_state=S0, chunk_size=64)
        end
        # one token, as when decoding: several blocks of rows, rows in several parts
        x, dt, A, B, C, D, S0 = ssd_inputs(70, 4, 130, 2, 1, 3)
        q, k, v, g, beta, R0 = delta_inputs(130, 2, 70, 4, 1, 3)
        for init in (false, true), lengths in (nothing, [0, 1, 1])
            S, R = init ? (S0, R0) : (nothing, nothing)
            y = NNlib.ssd_scan(x, dt, A, B, C; D, initial_state=S, lengths)
            @test all(cpu.(NNlib.ssd_scan(device.((x, dt, A, B, C))...; D=device(D),
                                          initial_state=device(S), lengths)) .≈ y)
            o = NNlib.gated_delta_rule(q, k, v, g, beta; initial_state=R, lengths)
            @test all(cpu.(NNlib.gated_delta_rule(device.((q, k, v, g, beta))...;
                                                  initial_state=device(R), lengths)) .≈ o)
        end
        # half-precision inputs, computed in single precision
        x, dt, A, B, C, D, S0 = device.(ssd_inputs(8, 4, 16, 2, 40, 2))
        x16, B16, C16 = Float16.(x), Float16.(B), Float16.(C)
        g16 = Zygote.gradient((x, B, C) -> sum(NNlib.ssd_scan(x, dt, A, B, C; D, chunk_size=16)[1]), x16, B16, C16)
        g32 = Zygote.gradient((x, B, C) -> sum(NNlib.ssd_scan(x, dt, A, B, C; D, chunk_size=16)[1]),
                              map(a -> Float32.(a), (x16, B16, C16))...)
        @test eltype(g16[1]) == Float16
        for (a, b) in zip(g16, g32)
            @test Float32.(cpu(a)) ≈ cpu(b) rtol=1e-2
        end
    end

    @testset "gradients" begin
        # two chunks, the second padded, and the step of one token
        for n in (7, 1)
            x, dt, A, B, C, D, S0 = ssd_inputs(2, 4, 3, 2, n, 2)
            @test test_gradients(x, dt, A, B, C, D, S0; test_gpu, compare) do x, dt, A, B, C, D, S0
                y, S = NNlib.ssd_scan(x, dt, A, B, C; D, initial_state=S0, chunk_size=4)
                return vcat(vec(y), vec(S))
            end
            q, k, v, g, beta, S0 = delta_inputs(3, 1, 2, 2, n, 2)
            @test test_gradients(q, k, v, g, beta, S0; test_gpu, compare) do q, k, v, g, beta, S0
                o, S = NNlib.gated_delta_rule(q, k, v, g, beta; initial_state=S0, chunk_size=4)
                return vcat(vec(o), vec(S))
            end
        end
    end
end
