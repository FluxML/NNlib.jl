function bmm_test(a,b; transA = false, transB = false)
    bs = size(a,3)
    transA && (a = permutedims(a, [2,1,3]))
    transB && (b = permutedims(b, [2,1,3]))
    c = []
    for i = 1:bs
        push!(c, a[:,:,i]*b[:,:,i])
    end

    cat(c...; dims = 3)
end

@testset "Batched Matrix Multiplication" begin
    A = randn(7,5,3)
    B = randn(5,7,3)
    C = randn(7,6,3)

    @test batchedmul(A, B) == bmm_test(A, B)
    @test batchedmul(A, B; transA = true, transB = true) == bmm_test(A, B; transA = true, transB = true)
    @test batchedmul(A, C; transA = true) == bmm_test(A, C; transA = true)
    @test batchedmul(A, A; transB = true) == bmm_test(A, A; transB = true)
end
