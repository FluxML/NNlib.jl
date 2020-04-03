using Test

using Base.CoreLogging: Debug

using NNlib
using NNlib: memory_layout, storage_type, batched_mul!

using ArrayLayouts
using ArrayLayouts: DenseColumnMajor, UnitStride, StridedLayout, ConjLayout

# Minimal wrapper which ArrayLayouts knows nothing about
struct TestWrap{T,AT} <: AbstractArray{T,3}
    data::AT
    TestWrap(A::AT) where {AT<:AbstractArray{T,3}} where {T} = new{T,AT}(A)
end
Base.size(A::TestWrap) = size(A.data)
Base.getindex(A::TestWrap, i...) = A.data[i...]
Base.parent(A::TestWrap) = A.data
Base.strides(A::TestWrap) = strides(A.data)
Base.unsafe_convert(::Type{Ptr{T}}, A::TestWrap{T}) where {T} =
    Base.unsafe_convert(Ptr{T}, parent(A))

@testset "ArrayLayouts + storage_type" begin

    A = randn(ComplexF64, 7,5,3)
    @test memory_layout(A) == DenseColumnMajor()

    @test memory_layout(batched_transpose(A)) == UnitStride{2}()
    @test memory_layout(batched_adjoint(A)) == ConjLayout{UnitStride{2}}()

    @test memory_layout(PermutedDimsArray(A, (1,3,2))) == UnitStride{1}()
    @test memory_layout(PermutedDimsArray(A, (2,1,3))) == UnitStride{2}()
    @test memory_layout(PermutedDimsArray(A, (2,3,1))) == UnitStride{3}()

    @test memory_layout(TestWrap(A)) == UnitStride{1}()
    @test memory_layout(TestWrap(batched_transpose(A))) == UnitStride{2}()
    @test memory_layout(TestWrap(batched_adjoint(A))) == ConjLayout{UnitStride{2}}()
    @test stride(TestWrap(A),3) == stride(A,3)

    @test storage_type(TestWrap(A)) == typeof(A)
    @test storage_type(batched_transpose(A)) == typeof(A)

end

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
        @test batched_transpose(PermutedDimsArray(A, (2,1,3))) === A
        @test batched_adjoint(PermutedDimsArray(A, (2,1,3))) === A

    end
    @testset "complex" begin

        cA = randn(Complex{Float64}, 7,5,3)
        cB = randn(Complex{TB}, 5,7,3)
        cC = randn(Complex{Float64}, 7,6,3)
        # cA_, cC_ = complex(TB).(cA), complex(TB).(cC)

        @test batched_mul(cA, cB) ≈ bmm_adjtest(cA, cB)
        @test batched_mul(batched_adjoint(cA), batched_adjoint(cB)) ≈
            bmm_adjtest(cA, cB; adjA = true, adjB = true)
        # @test batched_mul(batched_adjoint(cA), cC_) ≈ bmm_adjtest(cA, cC_; adjA = true)
        # @test batched_mul(cA, batched_adjoint(cA_)) ≈ bmm_adjtest(cA, cA_; adjB = true)

        if VERSION >= v"1.3"
            Z = batched_mul(cA, cB)
            @test batched_mul!(similar(Z), cA, cB, 2) ≈ 2 .* Z
            @test batched_mul!(copy(Z), cA, cB, 1, 1) ≈ 2 .* Z
        end

        if TB == Float64
            @test_logs min_level=Debug batched_mul(cA, cB)
            # @test_logs min_level=Debug batched_mul(cA, PermutedDimsArray(cA_, (2,1,3)))
        end

        @test batched_adjoint(batched_adjoint(cA)) === cA
        @test batched_transpose(batched_transpose(cA)) === cA
        @test batched_transpose(PermutedDimsArray(cA, (2,1,3))) === cA
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
    @testset "permuted output" begin # this is broken!

        A = rand(3,3,3)
        B = rand(TB, 3,3,3)
        C = PermutedDimsArray(zeros(3,3,3), (2,1,3))
        @test batched_mul(A, B) ≈ batched_mul!(C, A, B)
        @test batched_mul!(C, A, B) === C # check it returns C, not an un-wrapped version

        A = A .+ im;
        B = batched_transpose(B .+ im);
        C = PermutedDimsArray(zeros(ComplexF64, 3,3,3), (3,1,2))
        @test batched_mul(A, B) ≈ batched_mul!(C, A, B)
        @test batched_mul(A, B) ≈ C # check it mutated C
        if TB == Float64 # check it doesn't go to fallback
            @test_logs min_level=Debug batched_mul!(C, A, B)
        end

        A = batched_adjoint(A)
        @test batched_mul(A, B) ≈ batched_mul!(C, A, B)
        # now this goes to the fallback, becuase this is 'C'*'T' not 'N'
        memory_layout(batched_transpose(A)) == ConjLayout{UnitStride{1}}()

    end
    if TB == Float32
        @testset "all PermutedDimsArrays" begin

            _FUNS = [identity, batched_adjoint]
            _PERMS = [(1,2,3), (1,3,2), (2,1,3), (3,1,2), (2,3,1), (3,2,1)]

            @testset "permutations (A ~ $p) & (B ~ $q)" for p in _PERMS, q in _PERMS
                A = PermutedDimsArray(rand(TB, 3,3,3), p)
                A0 = collect(A)
                B = PermutedDimsArray(rand(TB, 3,3,3), q)
                B0 = collect(B)
                @testset "functions $f & $g" for f in _FUNS, g in _FUNS
                    C0 = batched_mul(f(A0), g(B0))
                    @test batched_mul(f(A), g(B)) ≈ C0
                    # TestWrap forces it to check strides at runtime
                    @test batched_mul(TestWrap(f(A)), TestWrap(g(B))) ≈ C0
                end
            end

        end
    end
end
