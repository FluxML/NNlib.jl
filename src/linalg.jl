## Low level gemm! call with pointers
## Borrowed from Knet.jl

using LinearAlgebra
using LinearAlgebra.BLAS: libblas, BlasInt,  @blasfunc

# C := alpha*op(A)*op(B) + beta*C, where:
# op(X) is one of op(X) = X, or op(X) = XT, or op(X) = XH,
# alpha and beta are scalars,
# A, B and C are matrices:
# op(A) is an m-by-k matrix,
# op(B) is a k-by-n matrix,
# C is an m-by-n matrix.

for (gemm, elty) in ((:dgemm_,:Float64), (:sgemm_,:Float32))
    @eval begin
        function gemm!(transA::Char, transB::Char, M::Int, N::Int, K::Int, alpha::($elty), A::Ptr{$elty}, B::Ptr{$elty}, beta::($elty), C::Ptr{$elty})
            if transA=='N'; lda=M; else; lda=K; end
            if transB=='N'; ldb=K; else; ldb=N; end
            ldc = M;
            ccall((@blasfunc($(gemm)), libblas), Nothing,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                   Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ptr{$elty},
                   Ref{BlasInt}),
                  transA, transB, M, N, K,
                  alpha, A, lda, B, ldb, beta, C, ldc)
        end
    end
end


##  borrow BatchedRoutines.jl
# batched gemm for 3d-array
# C[:,:,i] := alpha*op(A[:,:,i])*op(B[:,:,i]) + beta*C[:,:,i], where:
# i is the specific batch number,
# op(X) is one of op(X) = X, or op(X) = XT, or op(X) = XH,
# alpha and beta are scalars,
# A, B and C are 3d Array:
# op(A) is an m-by-k-by-b 3d Array,
# op(B) is a k-by-n-by-b 3d Array,
# C is an m-by-n-by-b 3d Array.

for (gemm, elty) in
        ((:dgemm_,:Float64),
         (:sgemm_,:Float32),
         (:zgemm_,:ComplexF64),
         (:cgemm_,:ComplexF32))
    @eval begin
        function batched_gemm!(transA::AbstractChar,
                               transB::AbstractChar,
                               alpha::($elty),
                               A::AbstractArray{$elty, 3},
                               B::AbstractArray{$elty, 3},
                               beta::($elty),
                               C::AbstractArray{$elty, 3})
            @assert !LinearAlgebra.BLAS.has_offset_axes(A, B, C)
            @assert size(A, 3) == size(B, 3) == size(C, 3) "batch size mismatch"
            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)
            if ka != kb || m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
            end
            LinearAlgebra.BLAS.chkstride1(A)
            LinearAlgebra.BLAS.chkstride1(B)
            LinearAlgebra.BLAS.chkstride1(C)

            ptrA = Base.unsafe_convert(Ptr{$elty}, A)
            ptrB = Base.unsafe_convert(Ptr{$elty}, B)
            ptrC = Base.unsafe_convert(Ptr{$elty}, C)

            for k in 1:size(A, 3)
                ccall((LinearAlgebra.BLAS.@blasfunc($gemm), LinearAlgebra.BLAS.libblas), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{LinearAlgebra.BLAS.BlasInt},
                     Ref{LinearAlgebra.BLAS.BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{LinearAlgebra.BLAS.BlasInt},
                     Ptr{$elty}, Ref{LinearAlgebra.BLAS.BlasInt}, Ref{$elty}, Ptr{$elty},
                     Ref{LinearAlgebra.BLAS.BlasInt}),
                     transA, transB, m, n,
                     ka, alpha, ptrA, max(1,Base.stride(A,2)),
                     ptrB, max(1,Base.stride(B,2)), beta, ptrC,
                     max(1,Base.stride(C,2)))

                ptrA += size(A, 1) * size(A, 2) * sizeof($elty)
                ptrB += size(B, 1) * size(B, 2) * sizeof($elty)
                ptrC += size(C, 1) * size(C, 2) * sizeof($elty)
            end

            C
        end
    end
end
