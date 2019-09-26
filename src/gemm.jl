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

for (gemm, elt) in gemm_datatype_mappings
    @eval begin
        @inline function batched_gemm!(transA::AbstractChar,
                               transB::AbstractChar,
                               alpha::($elt),
                               A::AbstractArray{$elt, 3},
                               B::AbstractArray{$elt, 3},
                               beta::($elt),
                               C::AbstractArray{$elt, 3})
            @assert !Base.has_offset_axes(A, B, C)
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

            ptrA = Base.unsafe_convert(Ptr{$elt}, A)
            ptrB = Base.unsafe_convert(Ptr{$elt}, B)
            ptrC = Base.unsafe_convert(Ptr{$elt}, C)

            for k in 1:size(A, 3)
                ccall((@blasfunc($(gemm)), libblas), Nothing,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                       Ref{BlasInt}, Ref{$elt}, Ptr{$elt}, Ref{BlasInt},
                       Ptr{$elt}, Ref{BlasInt}, Ref{$elt}, Ptr{$elt},
                       Ref{BlasInt}),
                      transA, transB, m, n,
                      ka, alpha, ptrA, max(1,Base.stride(A,2)),
                      ptrB, max(1,Base.stride(B,2)), beta, ptrC,
                      max(1,Base.stride(C,2)))

                ptrA += size(A, 1) * size(A, 2) * sizeof($elt)
                ptrB += size(B, 1) * size(B, 2) * sizeof($elt)
                ptrC += size(C, 1) * size(C, 2) * sizeof($elt)
            end

            C
        end
    end
end
