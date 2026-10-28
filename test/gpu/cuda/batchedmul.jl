@testset "batched_mul" begin
    A = randn(Float32, 3,3,2);
    B = randn(Float32, 3,3,2);

    C = batched_mul(A, B)
    @test CuArray(C) ≈ batched_mul(CuArray(A), CuArray(B))

    Ct = batched_mul(batched_transpose(A), B)
    @test CuArray(Ct) ≈ batched_mul(batched_transpose(CuArray(A)), CuArray(B))

    Ca = batched_mul(A, batched_adjoint(B))
    @test CuArray(Ca) ≈ batched_mul(CuArray(A), batched_adjoint(CuArray(B)))

    # 5-arg batched_mul!
    C .= pi
    batched_mul!(C, A, B, 2f0, 3f0)
    cuCpi = CuArray(similar(C)) .= pi
    @test CuArray(C) ≈ batched_mul!(cuCpi, CuArray(A), CuArray(B), 2f0, 3f0)

    # PermutedDimsArray
    @test CuArray(Ct) ≈ batched_mul(PermutedDimsArray(CuArray(A), (2,1,3)), CuArray(B))

    D = permutedims(B, (1,3,2))
    Cp = batched_mul(batched_adjoint(A), B)
    @test CuArray(Cp) ≈ batched_mul(batched_adjoint(CuArray(A)), PermutedDimsArray(CuArray(D), (1,3,2)))

    # Methods which reshape
    M = randn(Float32, 3,3)

    Cm = batched_mul(A, M)
    @test CuArray(Cm) ≈ batched_mul(CuArray(A), CuArray(M))

    Cv = batched_vec(permutedims(A,(3,1,2)), M)
    @test CuArray(Cv) ≈ batched_vec(PermutedDimsArray(CuArray(A),(3,1,2)), CuArray(M))
end

@testset "NNlib storage_type etc." begin
    M = cu(ones(10,10))

    @test NNlib.is_strided(M)
    @test NNlib.is_strided(view(M, 1:2:5,:))
    @test NNlib.is_strided(PermutedDimsArray(M, (2,1)))

    @test !NNlib.is_strided(reshape(view(M, 1:2:10,:), 10,:))
    @test !NNlib.is_strided((M .+ im)')
    @test !NNlib.is_strided(LinearAlgebra.Diagonal(cu(ones(3))))

    @test NNlib.storage_type(M) <: CuArray{Float32,2}
    @test NNlib.storage_type(reshape(view(M, 1:2:10,:), 10,:)) <: CuArray{Float32,2}
end

@testset "half-precision batched_mul ($T)" for T in (Float16, BFloat16)
    T == BFloat16 && CUDA.capability(CUDA.device()) < v"8.0" && continue
    @testset "batches $na, $nb; transpose $ta, $tb" for (na, nb) in ((3, 3), (1, 3), (3, 1)),
            ta in (false, true), tb in (false, true)
        a = T.(CUDA.randn(Float32, ta ? (7, 5, na) : (5, 7, na)))
        b = T.(CUDA.randn(Float32, tb ? (9, 7, nb) : (7, 9, nb)))
        A = ta ? batched_transpose(a) : a
        B = tb ? batched_adjoint(b) : b
        af, bf = Float32.(Array(A)), Float32.(Array(B))
        ref = batched_mul(af, bf)
        @test Float32.(Array(batched_mul(A, B))) ≈ ref rtol=2f-2
        C = T.(CUDA.randn(Float32, size(ref)))
        cref = 0.12345f0 .* ref .+ 0.23456f0 .* Float32.(Array(C))
        @test batched_mul!(C, A, B, 0.12345f0, 0.23456f0) === C
        @test Float32.(Array(C)) ≈ cref rtol=2f-2
        # Directly exercise the single cuBLAS call, including Float32 scalar coefficients.
        fill!(C, 0)
        NNlib._batched_gemm!(typeof(C), ta ? 'T' : 'N', tb ? 'T' : 'N', 1f0, a, b, 0f0, C)
        @test Float32.(Array(C)) ≈ ref rtol=2f-2
    end
    A = T.(CUDA.randn(Float32, 5, 7, 3))
    B = T.(CUDA.randn(Float32, 7, 9, 3))
    @test Float32.(Array(batched_mul(PermutedDimsArray(permutedims(A, (3, 1, 2)), (2, 3, 1)), B))) ≈
          batched_mul(Float32.(Array(A)), Float32.(Array(B))) rtol=2f-2
    @test Float32.(Array(batched_mul(view(A, :, :, 1:2:3), view(B, :, :, 1:2:3)))) ≈
          batched_mul(Float32.(Array(A[:, :, 1:2:3])), Float32.(Array(B[:, :, 1:2:3]))) rtol=2f-2
    C = CUDA.zeros(T, 10, 9, 3)
    batched_mul!(view(C, 1:2:10, :, :), A, B)
    @test Float32.(Array(C[1:2:10, :, :])) ≈ batched_mul(Float32.(Array(A)), Float32.(Array(B))) rtol=2f-2
    @test all(iszero, Array(C[2:2:10, :, :]))
    @test_throws DimensionMismatch batched_mul(A, B[:, :, 1:2])
    g = gradient((a, b) -> sum(batched_mul(a, b)), A, B)
    gref = gradient((a, b) -> sum(batched_mul(a, b)), Float32.(Array(A)), Float32.(Array(B)))
    @test all(isapprox.(map(x -> Float32.(Array(x)), g), gref; rtol=2f-2))
    A = CUDA.fill(T(300), 1, 32, 2)
    B = CuArray(repeat(reshape(T[300, -300], 2, 1, 1), 16, 1, 2))
    @test all(iszero, Array(batched_mul(A, B)))
end
