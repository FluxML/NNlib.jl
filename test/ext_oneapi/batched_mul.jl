@testset "batched_mul" begin
    A = rand(Float32, 3, 3, 2)
    B = rand(Float32, 3, 3, 2)
    dA, dB = oneArray.((A, B))

    C = batched_mul(A, B)
    @test oneArray(C) ≈ batched_mul(dA, dB)

    Ct = batched_mul(batched_transpose(A), B)
    @test oneArray(Ct) ≈ batched_mul(batched_transpose(dA), dB)

    Ca = batched_mul(A, batched_adjoint(B))
    @test oneArray(Ca) ≈ batched_mul(dA, batched_adjoint(dB))

    # 5-arg batched_mul!
    C .= pi
    batched_mul!(C, A, B, 2f0, 3f0)
    Cpi = oneArray(similar(C)) .= pi
    @test oneArray(C) ≈ batched_mul!(Cpi, dA, dB, 2f0, 3f0)

    # PermutedDimsArray
    @test oneArray(Ct) ≈ batched_mul(PermutedDimsArray(dA, (2, 1, 3)), dB)

    # FIXME same but with (1, 3, 2) errors
    D = permutedims(B, (2, 1, 3))
    Cp = batched_mul(batched_adjoint(A), B)
    @test oneArray(Cp) ≈ batched_mul(
        batched_adjoint(dA), PermutedDimsArray(oneArray(D), (2, 1, 3)))

    # Methods which reshape
    M = randn(Float32, 3, 3)
    Cm = batched_mul(A, M)
    @test oneArray(Cm) ≈ batched_mul(dA, oneArray(M))
end
