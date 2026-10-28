function groupedmul_reference(A, B, offsets)
    T = promote_type(eltype(A), eltype(B))
    C = ndims(A) == 3 ? zeros(T, size(A, 1), size(B, 2)) :
        zeros(T, size(A, 1), size(B, 2), length(offsets))
    start = 1
    for (g, stop) in enumerate(offsets)
        r = start:stop
        if ndims(A) == 3
            C[:, r] = A[:, :, g] * B[:, r]
        else
            C[:, :, g] = A[:, r] * B[r, :]
        end
        start = stop + 1
    end
    return C
end

function groupedmul_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    types = Backend == CPU ? (Float32, Float64, ComplexF64) :
        nameof(Backend) === :MetalBackend ? (Float32,) : (Float32, Float64)
    @testset "grouped_mul ($T)" for T in types
        W, X, B = randn(rng, T, 3, 2, 5), randn(rng, T, 2, 9), randn(rng, T, 9, 4)
        wd, xd, bd = device.((W, X, B))
        @testset "offsets $offsets" for offsets in (Int32[0, 2, 2, 6, 6], zeros(Int, 5), [1, 2, 3, 4, 9])
            @test cpu(grouped_mul(wd, xd, offsets)) ≈ groupedmul_reference(W, X, offsets)
            @test cpu(grouped_mul(xd, bd, offsets)) ≈ groupedmul_reference(X, B, offsets)
            for (a, b) in ((wd, xd), (xd, bd))
                ref = groupedmul_reference(cpu(a), cpu(b), offsets)
                C = device(randn(rng, T, size(ref)))
                before = copy(cpu(C))
                @test grouped_mul!(C, a, b, offsets, 0.7, 0.3) === C
                @test cpu(C) ≈ 0.7 .* ref .+ 0.3 .* before
                fill!(C, T(NaN))
                grouped_mul!(C, a, b, offsets)
                @test cpu(C) ≈ ref
            end
        end

        offsets = [0, 2, 2, 6, 6]
        for wrap in (batched_transpose, batched_adjoint)
            w = wrap(device(randn(rng, T, 2, 3, 5)))
            @test cpu(grouped_mul(w, xd, offsets)) ≈ groupedmul_reference(cpu(w), X, offsets)
        end
        for wrap in (transpose, adjoint)
            a, b = wrap(device(randn(rng, T, 9, 2))), wrap(device(randn(rng, T, 4, 9)))
            @test cpu(grouped_mul(a, b, offsets)) ≈ groupedmul_reference(cpu(a), cpu(b), offsets)
        end
        w = view(device(randn(rng, T, 6, 2, 10)), 1:2:6, :, 1:2:10)
        x = view(device(randn(rng, T, 4, 9)), 1:2:4, :)
        @test cpu(grouped_mul(w, x, offsets)) ≈ groupedmul_reference(cpu(w), cpu(x), offsets)
        C = device(zeros(T, 6, 9))
        grouped_mul!(view(C, 1:2:6, :), wd, xd, offsets)
        @test cpu(C)[1:2:6, :] ≈ groupedmul_reference(W, X, offsets)
        @test all(iszero, cpu(C)[2:2:6, :])

        @testset "gradients" begin
            dy = randn(rng, T, 3, 9)
            dw, dx = gradient((w, x) -> real(sum(conj.(device(dy)) .* grouped_mul(w, x, offsets))), wd, xd)
            dw_ref, dx_ref = zeros(T, size(W)), zeros(T, size(X))
            dc = randn(rng, T, 2, 4, 5)
            da, db = gradient((a, b) -> real(sum(conj.(device(dc)) .* grouped_mul(a, b, offsets))), xd, bd)
            da_ref, db_ref = zeros(T, size(X)), zeros(T, size(B))
            start = 1
            for (g, stop) in enumerate(offsets)
                r = start:stop
                dw_ref[:, :, g] = dy[:, r] * X[:, r]'
                dx_ref[:, r] = W[:, :, g]' * dy[:, r]
                da_ref[:, r] = dc[:, :, g] * B[r, :]'
                db_ref[r, :] = X[:, r]' * dc[:, :, g]
                start = stop + 1
            end
            @test cpu(dw) ≈ dw_ref
            @test cpu(dx) ≈ dx_ref
            @test cpu(da) ≈ da_ref
            @test cpu(db) ≈ db_ref
        end
    end
end
