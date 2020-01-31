## Low level gemm! call with pointers
## Borrowed from Knet.jl, adapted for compile-time constants

using LinearAlgebra
using LinearAlgebra.BLAS: libblas, BlasInt, @blasfunc

"""
    gemm!()

Low-level gemm!() call with pointers, borrowed from Knet.jl

Calculates `C = alpha*op(A)*op(B) + beta*C`, where:
  - `transA` and `transB` set `op(X)` to be either `identity()` or `transpose()`
  - alpha and beta are scalars
  - op(A) is an (M, K) matrix
  - op(B) is a (K, N) matrix
  - C is an (M, N) matrix.
"""
gemm!

# These are the datatypes we have fast GEMM for
gemm_datatype_mappings = (
    (:dgemm_, Float64),
    (:sgemm_, Float32),
    (:zgemm_, ComplexF64),
    (:cgemm_, ComplexF32),
)
for (gemm, elt) in gemm_datatype_mappings
    @eval begin
        @inline function gemm!(transA::Val, transB::Val,
                               M::Int, N::Int, K::Int,
                               alpha::$(elt), A::Ptr{$elt}, B::Ptr{$elt},
                               beta::$(elt), C::Ptr{$elt})
            # Convert our compile-time transpose marker to a char for BLAS
            convtrans(V::Val{false}) = 'N'
            convtrans(V::Val{true})  = 'T'

            if transA == Val(false)
                lda = M
            else
                lda = K
            end
            if transB == Val(false)
                ldb = K
            else
                ldb = N
            end
            ldc = M
            ccall((@blasfunc($(gemm)), libblas), Nothing,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                   Ref{BlasInt}, Ref{$elt}, Ptr{$elt}, Ref{BlasInt},
                   Ptr{$elt}, Ref{BlasInt}, Ref{$elt}, Ptr{$elt},
                   Ref{BlasInt}),
                  convtrans(transA), convtrans(transB), M, N, K,
                  alpha, A, lda, B, ldb, beta, C, ldc)
        end
    end
end
