## Low level gemm! call with pointers
## Borrowed from Knet.jl

using Base.LinAlg
using Base.LinAlg.BLAS: libblas, BlasInt,  @blasfunc

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
            ccall((@blasfunc($gemm), libblas), Void,
                  (Ptr{UInt8}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty},
                   Ptr{BlasInt}),
                  &transA, &transB, &M, &N, &K,
                  &alpha, A, &lda, B, &ldb, &beta, C, &ldc)
        end
    end
end
