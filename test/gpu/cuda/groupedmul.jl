@testset "grouped_mul CUDA ($T)" for T in (Float32, Float64, Float16, BFloat16)
    T == BFloat16 && CUDA.capability(CUDA.device()) < v"8.0" && continue
    offsets = Int32[0, 2, 2, 6, 6]
    wd = T.(CUDA.randn(Float32, 3, 2, 5))
    xd = T.(CUDA.randn(Float32, 2, 9))
    bd = T.(CUDA.randn(Float32, 9, 4))
    rtol = T == BFloat16 ? 2f-2 : T == Float16 ? 2f-3 : 1f-5
    for ends in (offsets, cu(offsets)), (a, b) in ((wd, xd), (xd, bd),
            (batched_transpose(permutedims(wd, (2, 1, 3))), xd),
            (xd, adjoint(permutedims(bd))))
        af, bf = Float32.(Array(a)), Float32.(Array(b))
        ref = grouped_mul(af, bf, offsets)
        y = grouped_mul(a, b, ends)
        @test eltype(y) == T
        @test Float32.(Array(y)) ≈ ref rtol=rtol
        c = T.(CUDA.randn(Float32, size(y)))
        expected = 0.12345f0 .* ref .+ 0.23456f0 .* Float32.(Array(c))
        grouped_mul!(c, a, b, ends, 0.12345f0, 0.23456f0)
        @test Float32.(Array(c)) ≈ expected rtol=rtol
        g = gradient((a, b) -> sum(abs2, grouped_mul(a, b, ends)), a, b)
        gref = gradient((a, b) -> sum(abs2, grouped_mul(a, b, offsets)), af, bf)
        @test all(isapprox.(map(x -> Float32.(Array(x)), g), gref; rtol=2rtol))
    end
    @test all(iszero, Array(grouped_mul(wd, xd, CUDA.zeros(Int32, 5))))
    @test all(iszero, Array(grouped_mul(xd, bd, CUDA.zeros(Int32, 5))))
    @test all(iszero, Array(grouped_mul(CUDA.zeros(T, 3, 0, 5), CUDA.zeros(T, 0, 9), offsets)))
    @test all(iszero, Array(grouped_mul(CUDA.zeros(T, 3, 2, 0), xd, Int[])))
    @test size(grouped_mul(xd, bd, Int[])) == (2, 4, 0)

    C = CUDA.zeros(T, 6, 9)
    grouped_mul!(view(C, 1:2:6, :), wd, xd, offsets)
    @test Float32.(Array(C[1:2:6, :])) ≈
        grouped_mul(Float32.(Array(wd)), Float32.(Array(xd)), offsets) rtol=rtol
    @test all(iszero, Array(C[2:2:6, :]))
    @test_throws ArgumentError grouped_mul(wd, xd, cu(Int32[0, 2, 1, 6, 6]))

    if T == Float16 || T == BFloat16
        W = CUDA.fill(T(300), 1, 32, 2)
        X = CuArray(repeat(reshape(T[300, -300], 2, 1), 16, 5))
        @test all(iszero, Array(grouped_mul(W, X, [2, 5])))
        A = CUDA.fill(T(300), 1, 32)
        B = CuArray(repeat(reshape(T[300, -300], 2, 1), 16, 1))
        @test all(iszero, Array(grouped_mul(A, B, [16, 32])))
    end
end
