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

function bmm_adjtest(a,b; adjA = false, adjB = false)
    bs = size(a,3)
    c = []
    for i = 1:bs
        ai = adjA ? adjoint(a[:,:,i]) : a[:,:,i]
        bi = adjB ? adjoint(b[:,:,i]) : b[:,:,i]
        push!(c, ai*bi)
    end

    cat(c...; dims = 3)
end

using Base.CoreLogging: Debug

@testset "batched_mul: Float64 * $TB" for TB in [Float64, Float32]
    @testset "real" begin

        A = randn(7,5,3)
        B = randn(TB, 5,7,3)
        C = randn(7,6,3)
        A_, C_ = TB.(A), TB.(C)

        @test batched_mul(A, B) ≈ bmm_test(A, B)
        @test batched_mul(batched_transpose(A), batched_transpose(B)) ≈ bmm_test(A, B; transA = true, transB = true)
        @test batched_mul(batched_transpose(A), C_) ≈ bmm_test(A, C_; transA = true)
        @test batched_mul(PermutedDimsArray(A, (2,1,3)), C_) ≈ bmm_test(A, C_; transA = true)
        @test batched_mul(A, batched_transpose(A_)) ≈ bmm_test(A, A_; transB = true)
        @test batched_mul(A, PermutedDimsArray(A_, (2,1,3))) ≈ bmm_test(A, A_; transB = true)

        if TB == Float64
            # check that these go to gemm!, not fallback:
            @test_logs min_level=Debug batched_mul(A, B)
            @test_logs min_level=Debug batched_mul(A, PermutedDimsArray(A_, (2,1,3)))
            @test_logs min_level=Debug batched_mul(PermutedDimsArray(A_, (2,1,3)), C)
        else
            # check that these write logging message:
            @test_logs min_level=Debug (:debug,
                "calling fallback method for batched_mul!") batched_mul(A, B)
            @test_logs min_level=Debug (:debug,
                "calling fallback method for batched_mul!") batched_mul(A, PermutedDimsArray(A_, (2,1,3)))
        end

        @test batched_transpose(batched_transpose(A)) === A
        @test batched_adjoint(batched_adjoint(A)) === A
        # mixed wrappers
        @test batched_transpose(batched_adjoint(A)) === A
        @test batched_adjoint(batched_transpose(A)) === A
        # this should NOT unwrap:
        @test batched_transpose(PermutedDimsArray(A, (2,1,3))) !== A
        @test batched_adjoint(PermutedDimsArray(A, (2,1,3))) !== A

    end
    @testset "complex" begin

        cA = randn(Complex{Float64}, 7,5,3)
        cB = randn(Complex{TB}, 5,7,3)
        cC = randn(Complex{Float64}, 7,6,3)
        cA_, cC_ = complex(TB).(cA), complex(TB).(cC)

        @test batched_mul(cA, cB) ≈ bmm_adjtest(cA, cB)
        @test batched_mul(batched_adjoint(cA), batched_adjoint(cB)) ≈
            bmm_adjtest(cA, cB; adjA = true, adjB = true)
        @test batched_mul(batched_adjoint(cA), cC_) ≈ bmm_adjtest(cA, cC_; adjA = true)
        @test batched_mul(cA, batched_adjoint(cA_)) ≈ bmm_adjtest(cA, cA_; adjB = true)

        if TB == Float64
            @test_logs min_level=Debug batched_mul(cA, cB)
            @test_logs min_level=Debug batched_mul(cA, PermutedDimsArray(cA_, (2,1,3)))
        end

        @test batched_adjoint(batched_adjoint(cA)) === cA
        @test batched_transpose(batched_transpose(cA)) === cA
        @test batched_transpose(PermutedDimsArray(cA, (2,1,3))) !== cA
        @test batched_adjoint(batched_transpose(cA)) != cA

    end
    @testset "integer" begin

        cA = randn(Complex{Float64}, 7,5,3)
        TBi = TB==Float64 ? Int64 : Int32
        iA = rand(1:99, 7,5,3)
        iB = TBi.(rand(1:99, 5,7,3))
        iC = zeros(Int, 7,6,3)

        @test batched_mul(iA, iB) == bmm_adjtest(iA, iB)
        @test batched_mul(cA, iB) ≈ bmm_adjtest(cA, iB)

    end
    @testset "misc" begin

        cA = randn(Complex{Float64}, 7,5,3)

        @test copy(cA[:,:,1]') isa Array
        @test copy(transpose(cA[:,:,1])) isa Array
        @test copy(batched_adjoint(cA)) isa Array
        @test copy(batched_transpose(cA)) isa Array

        @test strides(batched_transpose(cA)) == strides(PermutedDimsArray(cA, (2,1,3)))
        @test strides(batched_adjoint(cA)) == (7, 1, 35)

        @test_throws DimensionMismatch batched_mul(rand(2,2,2), rand(TB, 2,2,10))
        @test_throws DimensionMismatch batched_mul(rand(2,2,2), rand(TB, 10,2,2))
        @test_throws Exception batched_mul!(zeros(2,2,10), rand(2,2,2), rand(TB, 2,2,2))

    end
end

using NNlin: _perm12
@testset "_perm12" begin

    A = rand(Int8, 3,3,3)

    for a in 1:3, b in 1:3, c in 1:3
        perm = (a,b,c)
        isperm(perm) || continue

        B = permutedims(A, perm)
        C = PermutedDimsArray(A, perm)
        @test _perm12(B) == _perm12(C)
    end

end


using NNlib: is_strided
@testset "is_strided" begin

    M = ones(10,10)

    @test is_strided(M)
    @test is_strided(view(M, 1:2:5,:))
    @test is_strided(PermutedDimsArray(M, (2,1)))

    @test !is_strided(reshape(view(M, 1:2:10,:), 10,:))
    @test !is_strided((M.+im)')
    @test !is_strided(Diagonal(ones(3)))
    #=
    using SparseArrays
    @test !is_strided(sparse(M))
    using NamedDims
    @test is_strided(NamedDimsArray(M,(:a, :b))) # and 0.029 ns, 0 allocations
    =#

end
