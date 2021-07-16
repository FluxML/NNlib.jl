function gather_check_dims(X::AbstractArray{Tx,Nx}, 
                           Y::AbstractArray{Ty,Ny},
                           idx::AbstractArray{Tidx,Nidx}) where
                           {Tx,Ty,Tidx<:IntOrIntTuple,Nx,Ny,Nidx}
    M = NNlib.typelength(Tidx)
    dims = gather_check_dims(Nx, Ny, M, Nidx)
    size(X)[1:dims] == size(Y)[1:dims] || throw(ArgumentError("Incompatible input shapes."))
    size(Y)[dims+1:end] == size(idx) || throw(ArgumentError("Incompatible input shapes."))
    return dims
end

function gather_check_dims(X::AbstractArray{Tx,Nx}, 
                           Y::AbstractArray{Ty,Ny},
                           idx::AbstractArray{CartesianIndex{M},Nidx}) where
                           {Tx,Ty,Nx,Ny,M,Nidx}
    dims = gather_check_dims(Nx, Ny, M, Nidx)
    size(X)[1:dims] == size(Y)[1:dims] || throw(ArgumentError("Incompatible input shapes."))
    size(Y)[dims+1:end] == size(idx) || throw(ArgumentError("Incompatible input shapes."))
    return dims
end

function gather_check_dims(Nx, Ny, M, Nidx)
    @assert Nx - M == Ny - Nidx "Incompatible input shapes of (dst, src, idx) = ($Nx, $Ny, $Nidx)."
    dims = Nx - M
    dims < 0 && throw(ArgumentError("dims must be non-negative but got dims=$dims."))
    return dims
end

function gather_kernel!(dst, src, idx, max_idx, max_dims_idx, dims_size)
    index = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    @inbounds if index <= max_idx
        j, k = divrem(index-1, max_dims_idx)
        dims_i = CartesianIndices(dims_size)[k+1]
        dst[index] = src[dims_i, idx[j+1]...]
    end
    return nothing
end

function NNlib.gather!(dst::AnyCuArray, src::AnyCuArray, idx::AnyCuArray)
    dims = gather_check_dims(src, dst, idx)
    dims_size = size(src)[1:dims]
    max_dims_idx = prod(dims_size)
    max_idx = max_dims_idx * length(idx)
    args = dst, src, idx, max_idx, max_dims_idx, dims_size

    kernel = @cuda launch=false gather_kernel!(args...)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(max_idx, config.threads)
    blocks = cld(max_idx, threads)
    kernel(args...; threads=threads, blocks=blocks)
    return dst
end
