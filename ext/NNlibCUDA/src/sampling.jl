@inline function NNlib._safe_add!(dx::CuDeviceArray{T, 4}, value, ix, iy, c, n) where T
    @inbounds CUDA.@atomic dx[ix, iy, c, n] += value
end

function grid_sample_kernel!(n_elem, output, input, grid, padding_mode)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    if index < n_elem
        iW, iH, iC, _ = size(input)
        _, gW, gH, _ = size(grid)

        w = index % gW + 1
        h = (index ÷ gW) % gH + 1
        n = index ÷ (gW * gH) + 1
        NNlib._grid_sample_kernel!(
            output, input, grid, padding_mode, w, h, n, iW, iH, iC)
    end
    nothing
end

function ∇grid_sample_kernel!(n_elem, dx, dgrid, Δ, input, grid, padding_mode)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    if index < n_elem
        iW, iH, iC, _ = size(input)
        _, gW, gH, _ = size(grid)

        w = index % gW + 1
        h = (index ÷ gW) % gH + 1
        n = index ÷ (gW * gH) + 1
        NNlib._∇grid_sample_kernel!(
            dx, dgrid, Δ, input, grid, padding_mode, w, h, n, iW, iH, iC)
    end
    nothing
end

function NNlib.grid_sample(x::CuArray{T, 4}, grid::CuArray{T, 4}; padding_mode = Val(:zeros)) where T
    _, _, xC, xN = size(x)
    _, gW, gH, _ = size(grid)
    n_elem = gW * gH * xN
    y = similar(x, T, (gW, gH, xC, xN))

    kernel = @cuda launch=false grid_sample_kernel!(
        n_elem, y, x, grid, padding_mode)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(n_elem, config.threads)
    blocks = cld(n_elem, threads)
    kernel(n_elem, y, x, grid, padding_mode; threads=threads, blocks=blocks)
    y
end

function NNlib.∇grid_sample(Δ::CuArray{T, 4}, x::CuArray{T, 4}, grid::CuArray{T, 4}; padding_mode = Val(:zeros)) where T
    xN = size(x, 4)
    _, gW, gH, _ = size(grid)
    n_elem = gW * gH * xN
    dx, dgrid = CUDA.zeros(T, size(x)), similar(grid)

    kernel = @cuda launch=false ∇grid_sample_kernel!(
        n_elem, dx, dgrid, Δ, x, grid, padding_mode)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(n_elem, config.threads)
    blocks = cld(n_elem, threads)
    kernel(n_elem, dx, dgrid, Δ, x, grid, padding_mode; threads=threads, blocks=blocks)
    dx, dgrid
end
