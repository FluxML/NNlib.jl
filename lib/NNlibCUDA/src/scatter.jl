ATM_OPS = Dict((+) => CUDA.atomic_add!, (-) => CUDA.atomic_sub!, (max) => CUDA.atomic_max!, (min) => CUDA.atomic_min!,
               (*) => CUDA.atomic_mul!, (/) => CUDA.atomic_div!, (&) => CUDA.atomic_and!, (|) => CUDA.atomic_or!)

for (op, atm_op) in ATM_OPS
    @eval function scatter_kernel!(op::typeof($(op)), dst, src, idx)
        index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if index <= length(idx)
            i = Base._to_linear_index(dst, idx[index]...)
            $(atm_op)(pointer(dst, i), src[index])
        end
        return nothing
    end

    @eval function scatter_kernel!(op::typeof($(op)), dst, src, idx, dims::Val{N}, max_idx, max_dims_idx, dims_size) where {N}
        index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

        @inbounds if index <= max_idx
            j, k = divrem(index-1, max_dims_idx)
            dims_i = CartesianIndices(dims_size)[k+1]
            i = Base._to_linear_index(dst, Tuple(dims_i)..., idx[j+1]...)
            $(atm_op)(pointer(dst, i), src[index])
        end
        return nothing
    end

    @eval function NNlib.scatter!(op::typeof($(op)), dst::CuArray{Tdst}, src::CuArray{Tsrc}, idx::CuArray{<:IntOrIntTuple}, dims::Val{N}) where {Tdst,Tsrc,N}
        if N == 0
            max_idx = length(idx)
            threads = min(MAX_THREADS, max_idx)
            blocks = ceil(Int, max_idx / threads)
            @cuda blocks=blocks threads=threads scatter_kernel!(op, dst, src, idx)
            return dst
        else
            dims_size = size(dst)[1:N]
            max_dims_idx = prod(dims_size)
            max_idx = max_dims_idx * length(idx)
            threads = min(MAX_THREADS, max_idx)
            blocks = ceil(Int, max_idx / threads)
            @cuda blocks=blocks threads=threads scatter_kernel!(op, dst, src, idx, dims, max_idx, max_dims_idx, dims_size)
            return dst
        end
    end
end
