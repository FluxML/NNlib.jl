# batch-wise matrix multiplication
# wrapper for batched_gemm!

function batchedmul(a::AbstractArray{T, 3}, b::AbstractArray{T, 3};
                    transA::Bool = false, transB::Bool = false) where T
    (bs = size(a, 3)) == size(b, 3) || error("batch size mismatch")
    res = similar(a, size(a, transA ? 2 : 1), size(b, transB ? 1 : 2), bs)
    batched_mul!(res, a, b; transA=transA, transB=transB)
    return res
end

function batched_mul!(C::AbstractArray{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3};
                      transA::Bool = false, transB::Bool = false) where T
    At = transA ? 'T' : 'N'
    Bt = transB ? 'T' : 'N'
    batched_gemm!(At, Bt, one(T), A, B, zero(T), C)
    C
end

#gradient function for batchedmul
function ∇batchedmul(Δ::AbstractArray{T, 3}, a::AbstractArray{T, 3}, b::AbstractArray{T, 3};
                     transA::Bool = false, transB::Bool = false) where T
    if transA
        if transB
            (batchedmul(b, Δ; transA=true, transB=true), batchedmul(Δ, a; transA=true, transB=true))
        else
            (batchedmul(b, Δ; transB=true), batchedmul(a, Δ))
        end
    else
        if transB
            (batchedmul(Δ, b), batchedmul(Δ, a; transA=true))
        else
            (batchedmul(Δ, b; transB=true), batchedmul(a, Δ; transA=true))
        end
    end
end
