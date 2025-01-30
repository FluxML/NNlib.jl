@inline function NNlib._safe_add!(dx::CuDeviceArray{T, 4}, value, ix, iy, c, n) where T
    @inbounds CUDA.@atomic dx[ix, iy, c, n] += value
end

function grid_sample_kernel!(n_elem, output::AbstractArray{T, 4}, input::AbstractArray{T, 4}, grid::AbstractArray{V, 4}, padding_mode) where {T,V}
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    if index < n_elem
        iW, iH, iC, _ = size(input)
        _, gW, gH, _ = size(grid)

        w = index % gW + 1
        h = (index ÷ gW) % gH + 1
        n = index ÷ (gW * gH) + 1
        NNlib._grid_sample_kernel!(output, input, grid, padding_mode, w, h, n, iW, iH, iC)
    end
    nothing
end

function ∇grid_sample_kernel!(n_elem, dx::AbstractArray{T, 4}, dgrid::AbstractArray{V, 4}, Δ::AbstractArray{T, 4}, input::AbstractArray{T, 4}, grid::AbstractArray{V, 4}, padding_mode) where {T,V}
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    if index < n_elem
        iW, iH, iC, _ = size(input)
        _, gW, gH, _ = size(grid)

        w = index % gW + 1
        h = (index ÷ gW) % gH + 1
        n = index ÷ (gW * gH) + 1
        NNlib._∇grid_sample_kernel!(dx, dgrid, Δ, input, grid, padding_mode, w, h, n, iW, iH, iC)
    end
    nothing
end

function NNlib.grid_sample(x::CuArray{T, 4}, grid::CuArray{V, 4}; padding_mode = :zeros) where {T, V}
    pad = Val(padding_mode)
    _, _, xC, xN = size(x)
    _, gW, gH, _ = size(grid)
    n_elem = gW * gH * xN
    y = similar(x, T, (gW, gH, xC, xN))

    kernel = @cuda launch=false grid_sample_kernel!(n_elem, y, x, grid, pad)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(n_elem, config.threads)
    blocks = cld(n_elem, threads)
    kernel(n_elem, y, x, grid, pad; threads=threads, blocks=blocks)
    y
end

function NNlib.∇grid_sample(Δ::CuArray{T, 4}, x::CuArray{T, 4}, grid::CuArray{V, 4}; padding_mode = :zeros) where {T, V}
    pad = Val(padding_mode)
    xN = size(x, 4)
    _, gW, gH, _ = size(grid)
    n_elem = gW * gH * xN
    dx, dgrid = CUDA.zeros(T, size(x)), similar(grid)

    kernel = @cuda launch=false ∇grid_sample_kernel!(n_elem, dx, dgrid, Δ, x, grid, pad)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(n_elem, config.threads)
    blocks = cld(n_elem, threads)
    kernel(n_elem, dx, dgrid, Δ, x, grid, pad; threads=threads, blocks=blocks)
    dx, dgrid
end


@inline function NNlib._safe_add!(dx::CuDeviceArray{T, 5}, value, ix, iy, iz, c, n) where T
    @inbounds CUDA.@atomic dx[ix, iy, iz, c, n] += value
end

function grid_sample_kernel!(n_elem, output::AbstractArray{T, 5}, input::AbstractArray{T, 5}, grid::AbstractArray{V, 5}, padding_mode) where {T,V}
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    if index < n_elem
        iW, iH,iD, iC, _ = size(input)
        _, gW, gH, gD, _ = size(grid)

        w = index % gW + 1
        h = (index ÷ gW) % gH + 1
        d = (index ÷ (gW * gH)) % gD + 1
        n = index ÷ (gW * gH * gD) + 1
        # n = index ÷ (gW * gH) + 1
        # d = (index ÷ (gW * gH * n)) + 1

        NNlib._grid_sample_kernel!(output, input, grid, padding_mode, w, h, d, n, iW, iH, iD, iC)
    end
    nothing
end

function ∇grid_sample_kernel!(n_elem, dx::AbstractArray{T, 5}, dgrid::AbstractArray{V, 5}, Δ::AbstractArray{T, 5}, input::AbstractArray{T, 5}, grid::AbstractArray{V, 5}, padding_mode) where {T,V}
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    if index < n_elem
        iW, iH, iD, iC, _ = size(input)
        _, gW, gH, gD, _ = size(grid)

        w = index % gW + 1
        h = (index ÷ gW) % gH + 1
        d = (index ÷ (gW * gH)) % gD + 1
        n = index ÷ (gW * gH * gD) + 1
        # n = index ÷ (gW * gH) + 1
        # d = (index ÷ (gW * gH * n)) + 1

        NNlib._∇grid_sample_kernel!(dx, dgrid, Δ, input, grid, padding_mode, w, h, d, n, iW, iH, iD, iC)
    end
    nothing
end

function NNlib.grid_sample(x::CuArray{T, 5}, grid::CuArray{V, 5}; padding_mode = :zeros) where {T, V}
    pad = Val(padding_mode)
    _, _, _, xC, xN = size(x)
    _, gW, gH, gD, _ = size(grid)
    n_elem = gW * gH * gD * xN
    y = similar(x, T, (gW, gH, gD, xC, xN))

    kernel = @cuda launch=false grid_sample_kernel!(n_elem, y, x, grid, pad)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(n_elem, config.threads)
    blocks = cld(n_elem, threads)
    kernel(n_elem, y, x, grid, pad; threads=threads, blocks=blocks)
    y
end

function NNlib.∇grid_sample(Δ::CuArray{T, 5}, x::CuArray{T, 5}, grid::CuArray{V, 5}; padding_mode = :zeros) where {T, V}
    pad = Val(padding_mode)
    xN = size(x, 5)
    _, gW, gH, gD, _ = size(grid)
    n_elem = gW * gH * gD * xN
    dx, dgrid = CUDA.zeros(T, size(x)), similar(grid)

    kernel = @cuda launch=false ∇grid_sample_kernel!(n_elem, dx, dgrid, Δ, x, grid, pad)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(n_elem, config.threads)
    blocks = cld(n_elem, threads)
    kernel(n_elem, dx, dgrid, Δ, x, grid, pad; threads=threads, blocks=blocks)
    dx, dgrid
end