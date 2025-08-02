## Low level gemm! call with pointers
## Borrowed from Knet.jl, adapted for compile-time constants

using LinearAlgebra.BLAS: get_num_threads, set_num_threads

if isdefined(LinearAlgebra.BLAS, :libblastrampoline)
    const libblas = LinearAlgebra.BLAS.libblastrampoline
else
    const libblas = Base.libblas_name
end

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
            convtrans(V::Val{true})  = 'C'

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
            @assert size(A, 3) == 1 || size(A, 3) == size(C, 3) "batch size mismatch: A != C"
            @assert size(B, 3) == 1 || size(B, 3) == size(C, 3) "batch size mismatch: B != C"

            m = size(A, transA == 'N' ? 1 : 2)
            ka = size(A, transA == 'N' ? 2 : 1)
            kb = size(B, transB == 'N' ? 1 : 2)
            n = size(B, transB == 'N' ? 2 : 1)
            if ka != kb || m != size(C,1) || n != size(C,2)
                throw(DimensionMismatch("A1 has size ($m,$ka), B1 has size ($kb,$n), C1 has size $(size(C)[1:2])"))
            end
            LinearAlgebra.BLAS.chkstride1(A)
            LinearAlgebra.BLAS.chkstride1(B)
            LinearAlgebra.BLAS.chkstride1(C)

            ptrA = pointer(A)
            ptrB = pointer(B)
            ptrC = pointer(C)

            strA = size(A, 3) == 1 ? 0 : Base.stride(A, 3)
            strB = size(B, 3) == 1 ? 0 : Base.stride(B, 3)
            strC = Base.stride(C, 3)

            n_threads = min(
                Threads.nthreads(:default),
                1 + max(length(A), length(B)) ÷ 8000)
            # In some tests, size (20,20,20) is worth splitting between two threads,
            # as is size (32,32,8).

            if n_threads > 1

                old_threads = get_num_threads()
                set_num_threads(1)

                parts = Iterators.partition(1:size(C, 3), cld(size(C, 3), n_threads))

                function gemm!_part(ks)
                    for k in ks

                        ptrAk = ptrA + (k-1) * strA * sizeof($elt)
                        ptrBk = ptrB + (k-1) * strB * sizeof($elt)
                        ptrCk = ptrC + (k-1) * strC * sizeof($elt)

                        ccall((@blasfunc($(gemm)), libblas), Nothing,
                            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                            Ref{BlasInt}, Ref{$elt}, Ptr{$elt}, Ref{BlasInt},
                            Ptr{$elt}, Ref{BlasInt}, Ref{$elt}, Ptr{$elt},
                            Ref{BlasInt}),
                            transA, transB, m, n,
                            ka, alpha, ptrAk, max(1,Base.stride(A,2)),
                            ptrBk, max(1,Base.stride(B,2)), beta, ptrCk,
                            max(1,Base.stride(C,2)))
                    end
                end
                if should_use_spawn() && length(parts) > 1
                    Threads.@sync for ks in parts
                        Threads.@spawn gemm!_part(ks)
                    end
                else
                    for ks in parts
                        gemm!_part(ks)
                    end
                end
                set_num_threads(old_threads)

            else # small problem, no threads

                for k in 1:size(C, 3)
                    # Identical loop body

                    ptrAk = ptrA + (k-1) * strA * sizeof($elt)
                    ptrBk = ptrB + (k-1) * strB * sizeof($elt)
                    ptrCk = ptrC + (k-1) * strC * sizeof($elt)

                    ccall((@blasfunc($(gemm)), libblas), Nothing,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                           Ref{BlasInt}, Ref{$elt}, Ptr{$elt}, Ref{BlasInt},
                           Ptr{$elt}, Ref{BlasInt}, Ref{$elt}, Ptr{$elt},
                           Ref{BlasInt}),
                          transA, transB, m, n,
                          ka, alpha, ptrAk, max(1,Base.stride(A,2)),
                          ptrBk, max(1,Base.stride(B,2)), beta, ptrCk,
                          max(1,Base.stride(C,2)))
                end

            end

            return C
        end
    end
end
