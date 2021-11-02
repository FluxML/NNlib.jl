function grid_sample_kernel!(n_elem, output, input, grid, padding_mode)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    if index < n_elem
        iW, iH, iC, _ = size(input)
        _, gW, gH, _ = size(grid)

        w = index % gW + 1
        h = (index ÷ gW) % gH + 1
        n = index ÷ (gW * gH) + 1
        # Get the corresponding (x, y) coordinates from the grid.
        @inbounds x, y = grid[1, w, h, n], grid[2, w, h, n]
        ix = NNlib.compute_source_index(x, iW, padding_mode)
        iy = NNlib.compute_source_index(y, iH, padding_mode)
        # Get corner pixel values from (ix, iy) in north-east-south-west directions.
        ix_nw, iy_nw = floor(Int, ix), floor(Int64, iy)
        ix_ne, iy_ne = ix_nw + 1, iy_nw
        ix_sw, iy_sw = ix_nw, iy_nw + 1
        ix_se, iy_se = ix_ne, iy_sw
        # Get surfaces to each neighbor (a.k.a. interpolation weights).
        nw = (ix_se - ix) * (iy_se - iy)
        ne = (ix - ix_sw) * (iy_sw - iy)
        sw = (ix_ne - ix) * (iy - iy_ne)
        se = (ix - ix_nw) * (iy - iy_nw)
        # ∀ channel: Calculate billinear weighted pixel value.
        @inbounds for c in 1:iC
            r = 0.0
            NNlib.in_bounds(iy_nw, ix_nw, iH, iW) && (r += input[ix_nw, iy_nw, c, n] * nw)
            NNlib.in_bounds(iy_ne, ix_ne, iH, iW) && (r += input[ix_ne, iy_ne, c, n] * ne)
            NNlib.in_bounds(iy_sw, ix_sw, iH, iW) && (r += input[ix_sw, iy_sw, c, n] * sw)
            NNlib.in_bounds(iy_se, ix_se, iH, iW) && (r += input[ix_se, iy_se, c, n] * se)
            output[w, h, c, n] = r
        end
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
        # Get corresponding (x, y) from grid.
        @inbounds x, y = grid[1, w, h, n], grid[2, w, h, n]
        # Compute multipliers for gradinets on ix, iy.
        ix, gix_mult = NNlib.∇compute_source_index(x, iW, padding_mode)
        iy, giy_mult = NNlib.∇compute_source_index(y, iH, padding_mode)
        # Get corner pixel values from (ix, iy) in north-east-south-west directions.
        ix_nw, iy_nw = floor(Int, ix), floor(Int64, iy)
        ix_ne, iy_ne = ix_nw + 1, iy_nw
        ix_sw, iy_sw = ix_nw, iy_nw + 1
        ix_se, iy_se = ix_ne, iy_sw
        # Get surfaces to each neighbor (a.k.a. interpolation weights).
        nw = (ix_se - ix) * (iy_se - iy)
        ne = (ix - ix_sw) * (iy_sw - iy)
        sw = (ix_ne - ix) * (iy - iy_ne)
        se = (ix - ix_nw) * (iy - iy_nw)
        # ∀ channel: Calculate billinear weighted pixel value.
        gix, giy = 0.0, 0.0
        @inbounds for c in 1:iC
            g_out = Δ[w, h, c, n]
            # Calculate dx and dgrid partials.
            if NNlib.in_bounds(iy_nw, ix_nw, iH, iW)
                CUDA.@atomic dx[ix_nw, iy_nw, c, n] += g_out * nw
                nw_val = input[ix_nw, iy_nw, c, n]
                gix -= nw_val * (iy_se - iy) * g_out
                giy -= nw_val * (ix_se - ix) * g_out
            end
            if NNlib.in_bounds(iy_ne, ix_ne, iH, iW)
                CUDA.@atomic dx[ix_ne, iy_ne, c, n] += g_out * ne
                ne_val = input[ix_ne, iy_ne, c, n]
                gix += ne_val * (iy_sw - iy) * g_out
                giy -= ne_val * (ix - ix_sw) * g_out
            end
            if NNlib.in_bounds(iy_sw, ix_sw, iH, iW)
                CUDA.@atomic dx[ix_sw, iy_sw, c, n] += g_out * sw
                sw_val = input[ix_sw, iy_sw, c, n]
                gix -= sw_val * (iy - iy_ne) * g_out
                giy += sw_val * (ix_ne - ix) * g_out
            end
            if NNlib.in_bounds(iy_se, ix_se, iH, iW)
                CUDA.@atomic dx[ix_se, iy_se, c, n] += g_out * se
                se_val = input[ix_se, iy_se, c, n]
                gix += se_val * (iy - iy_nw) * g_out
                giy += se_val * (ix - ix_nw) * g_out
            end
        end
        @inbounds dgrid[1, w, h, n] = gix_mult * gix
        @inbounds dgrid[2,w, h, n] = giy_mult * giy
    end
    nothing
end

function NNlib.grid_sample(x::CuArray{T, 4}, grid::CuArray{T, 4}; padding_mode) where T
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

function NNlib.∇grid_sample(Δ::CuArray{T, 4}, x::CuArray{T, 4}, grid::CuArray{T, 4}; padding_mode) where T
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
