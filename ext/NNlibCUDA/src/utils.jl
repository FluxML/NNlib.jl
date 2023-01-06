NNlib._rng_from_array(::CuArray) = CUDA.default_rng()

function divide_kernel!(xs, ys, max_idx)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        xs[index] = xs[index] / ys[index]
    end
    return nothing
end

function divide_kernel!(xs, counts, max_idx, max_dims_idx, dims_size)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        j, k = divrem(index-1, max_dims_idx)
        dims_i = Tuple(CartesianIndices(dims_size)[k+1])
        CUDA.@atomic xs[dims_i..., j+1] = xs[dims_i..., j+1] / counts[j+1]
    end
    return nothing
end

function NNlib.reverse_indices(idx::AnyCuArray{<:Any,N}) where N
    max_dims = NNlib.maximum_dims(idx)
    T = CartesianIndex{N}
    rev = Array{Vector{T}}(undef, max_dims...)
    for i in eachindex(rev)
        rev[i] = T[]
    end
    NNlib.reverse_indices!(rev, idx)
    return map(cu, rev)
end
