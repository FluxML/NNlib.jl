@testset "batched_mul" begin
    using NNlib: batched_mul, batched_mul!, batched_vec,
                 batched_adjoint, batched_transpose

    A = randn(Float32, 3,3,2);
    B = randn(Float32, 3,3,2);

    C = batched_mul(A, B)
    @test MtlArray(C) ≈ batched_mul(MtlArray(A), MtlArray(B))

    Ct = batched_mul(batched_transpose(A), B)
    @test MtlArray(Ct) ≈ batched_mul(batched_transpose(MtlArray(A)), MtlArray(B))

    Ca = batched_mul(A, batched_adjoint(B))
    @test MtlArray(Ca) ≈ batched_mul(MtlArray(A), batched_adjoint(MtlArray(B)))

    # 5-arg batched_mul!
    C .= pi
    batched_mul!(C, A, B, 2f0, 3f0)
    gpuCpi = MtlArray(similar(C)) .= pi
    @test MtlArray(C) ≈ batched_mul!(gpuCpi, MtlArray(A), MtlArray(B), 2f0, 3f0)

    # PermutedDimsArray
    @test MtlArray(Ct) ≈ batched_mul(PermutedDimsArray(MtlArray(A), (2,1,3)), MtlArray(B))

    D = permutedims(B, (1,3,2))
    Cp = batched_mul(batched_adjoint(A), B)
    @test_broken MtlArray(Cp) ≈ batched_mul(batched_adjoint(MtlArray(A)), PermutedDimsArray(MtlArray(D), (1,3,2)))

    # Methods which reshape
    M = randn(Float32, 3,3)

    Cm = batched_mul(A, M)
    @test MtlArray(Cm) ≈ batched_mul(MtlArray(A), MtlArray(M))

    Cv = batched_vec(permutedims(A,(3,1,2)), M)
    @test_broken MtlArray(Cv) ≈ batched_vec(PermutedDimsArray(MtlArray(A),(3,1,2)), MtlArray(M))
end

function print_array_strs(x)
    str = sprint((io, x)->show(io, MIME"text/plain"(), x), x)
    return @view split(str, '\n')[2:end]
end

@testset "BatchedAdjOrTrans" begin
    x = rand(Float32, 3, 4, 2)
    y = MtlArray(x)

    bax = batched_adjoint(x)
    btx = batched_transpose(x)
    bay = batched_adjoint(y)
    bty = batched_transpose(y)

    @test sprint(show, bax) == sprint(show, bay)
    @test sprint(show, btx) == sprint(show, bty)

    @test print_array_strs(bax) == print_array_strs(bay)
    @test print_array_strs(btx) == print_array_strs(bty)

    @test Array(bax) == Array(bay)
    @test collect(bax) == collect(bay)
    @test Array(btx) == Array(bty)
    @test collect(btx) == collect(bty)

    for shape in (:, (12, 2))
        rbax = reshape(bax, shape)
        rbtx = reshape(btx, shape)
        rbay = reshape(bay, shape)
        rbty = reshape(bty, shape)

        @test sprint(show, rbax) == sprint(show, rbay)
        @test sprint(show, rbtx) == sprint(show, rbty)

        @test print_array_strs(rbax) == print_array_strs(rbay)
        @test print_array_strs(rbtx) == print_array_strs(rbty)

        @test Array(rbax) == Array(rbay)
        @test collect(rbax) == collect(rbay)
        @test Array(rbtx) == Array(rbty)
        @test collect(rbtx) == collect(rbty)
    end
end
