@inbounds in_bounds(h, w, H, W) = 1 ≤ h ≤ H && 1 ≤ w ≤ W
@inbounds clip_coordinate(coordinate, dim_size) = min(dim_size, max(1, coordinate))
"""
For the gradient calculation, borders are considered out-of-bounds.
"""
function ∇clip_coordinate(coordinate, dim_size)
    if coordinate ≤ 1
        return 1, 0
    elseif coordinate ≥ dim_size
        return dim_size, 0
    end
    coordinate, 1
end

"""
Unnormalize coordinates from `[-1, 1]` to `[1, dim_size]`.
Align corners - true:
    -1 → 1
    1 → dim_size
"""
function unnormalize(coordinate, dim_size)
    ((coordinate + 1.0) / 2.0) * (dim_size - 1.0) + 1.0
end
function unnormalize_grad(coordinate, dim_size)
    grad = (dim_size - 1.0) * 0.5 # TODO do - 1 or not?
    unnormalize(coordinate, dim_size), grad
end

function compute_source_index(coordinate, dim_size, padding_mode)
    source_coordinate = unnormalize(coordinate, dim_size)
    padding_mode == 0 ? source_coordinate : clip_coordinate(source_coordinate, dim_size)
end
function ∇compute_source_index(coordinate, dim_size, padding_mode)
    source_coordinate, grad_in = unnormalize_grad(coordinate, dim_size)
    if padding_mode == 1
        source_coordinate, grad_clip = ∇clip_coordinate(source_coordinate, dim_size)
        grad_in *= grad_clip
    end
    source_coordinate, grad_in
end

function grid_sampler_kernel!(n_elem, output, input, grid, padding_mode)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    if index < n_elem
        iW, iH, iC, _ = size(input)
        gW, gH = size(grid, 1), size(grid, 2)

        w = index % gW + 1
        h = (index ÷ gW) % gH + 1
        n = index ÷ (gW * gH) + 1
        # Get the corresponding (x, y) coordinates from the grid.
        x, y = grid[w, h, 1, n], grid[w, h, 2, n]
        ix = compute_source_index(x, iW, padding_mode)
        iy = compute_source_index(y, iH, padding_mode)
        # Get corner pixel values from (ix, iy) in north-east-south-west directions.
        ix_nw = floor(Int64, ix)
        ix_ne = ix_nw + 1
        ix_sw = ix_nw
        ix_se = ix_ne

        iy_nw = floor(Int64, iy)
        iy_ne = iy_nw
        iy_sw = iy_nw + 1
        iy_se = iy_sw
        # Get surfaces to each neighbor (a.k.a. interpolation weights).
        nw = (ix_se - ix) * (iy_se - iy)
        ne = (ix - ix_sw) * (iy_sw - iy)
        sw = (ix_ne - ix) * (iy - iy_ne)
        se = (ix - ix_nw) * (iy - iy_nw)
        # ∀ channel: Calculate billinear weighted pixel value.
        for c in 1:iC
            r = 0.0
            in_bounds(iy_nw, ix_nw, iH, iW) && (r += input[ix_nw, iy_nw, c, n] * nw)
            in_bounds(iy_ne, ix_ne, iH, iW) && (r += input[ix_ne, iy_ne, c, n] * ne)
            in_bounds(iy_sw, ix_sw, iH, iW) && (r += input[ix_sw, iy_sw, c, n] * sw)
            in_bounds(iy_se, ix_se, iH, iW) && (r += input[ix_se, iy_se, c, n] * se)
            output[w, h, c, n] = r
        end
    end
    nothing
end

function ∇grid_sampler_kernel!(n_elem, dx, dgrid, Δ, input, grid, padding_mode)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x
    if index < n_elem
        iW, iH, iC, _ = size(input)
        gW, gH = size(grid, 1), size(grid, 2)

        w = index % gW + 1
        h = (index ÷ gW) % gH + 1
        n = index ÷ (gW * gH) + 1
        # Get corresponding (x, y) from grid.
        x, y = grid[w, h, 1, n], grid[w, h, 2, n]
        # Compute multipliers for gradinets on ix, iy.
        ix, gix_mult = ∇compute_source_index(x, iW, padding_mode)
        iy, giy_mult = ∇compute_source_index(y, iH, padding_mode)
        # Get corner pixel values from (ix, iy) in north-east-south-west directions.
        ix_nw = floor(Int64, ix)
        ix_ne = ix_nw + 1
        ix_sw = ix_nw
        ix_se = ix_ne

        iy_nw = floor(Int64, iy)
        iy_ne = iy_nw
        iy_sw = iy_nw + 1
        iy_se = iy_sw
        # Get surfaces to each neighbor (a.k.a. interpolation weights).
        nw = (ix_se - ix) * (iy_se - iy)
        ne = (ix - ix_sw) * (iy_sw - iy)
        sw = (ix_ne - ix) * (iy - iy_ne)
        se = (ix - ix_nw) * (iy - iy_nw)
        # ∀ channel: Calculate billinear weighted pixel value.
        gix, giy = 0.0, 0.0
        for c in 1:iC
            g_out = Δ[w, h, c, n]
            # Calculate dx and dgrid partials.
            if in_bounds(iy_nw, ix_nw, iH, iW)
                CUDA.@atomic dx[ix_nw, iy_nw, c, n] += g_out * nw
                nw_val = input[ix_nw, iy_nw, c, n]
                gix -= nw_val * (iy_se - iy) * g_out
                giy -= nw_val * (ix_se - ix) * g_out
            end
            if in_bounds(iy_ne, ix_ne, iH, iW)
                CUDA.@atomic dx[ix_ne, iy_ne, c, n] += g_out * ne
                ne_val = input[ix_ne, iy_ne, c, n]
                gix += ne_val * (iy_sw - iy) * g_out
                giy -= ne_val * (ix - ix_sw) * g_out
            end
            if in_bounds(iy_sw, ix_sw, iH, iW)
                CUDA.@atomic dx[ix_sw, iy_sw, c, n] += g_out * sw
                sw_val = input[ix_sw, iy_sw, c, n]
                gix -= sw_val * (iy - iy_ne) * g_out
                giy += sw_val * (ix_ne - ix) * g_out
            end
            if in_bounds(iy_se, ix_se, iH, iW)
                CUDA.@atomic dx[ix_se, iy_se, c, n] += g_out * se
                se_val = input[ix_se, iy_se, c, n]
                gix += se_val * (iy - iy_nw) * g_out
                giy += se_val * (ix - ix_nw) * g_out
            end
        end
        dgrid[w, h, 1, n] = gix_mult * gix
        dgrid[w, h, 2, n] = giy_mult * giy
    end
    nothing
end

function grid_sampler(x::CuArray{T, 4}, grid::CuArray{T, 4}, padding_mode) where T
    xC, xN = size(x, 3), size(x, 4)
    gW, gH = size(grid, 1), size(grid, 2)
    n_elem = gW * gH * xN

    y = similar(x, T, (gW, gH, xC, xN))

    kernel = @cuda launch=false grid_sampler_kernel!(
        n_elem, y, x, grid, padding_mode)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(n_elem, config.threads)
    blocks = cld(n_elem, threads)
    kernel(n_elem, y, x, grid, padding_mode; threads=threads, blocks=blocks)
    y
end

function ∇grid_sampler(Δ::CuArray{T, 4}, x::CuArray{T, 4}, grid::CuArray{T, 4}, padding_mode) where T
    xN = size(x, 4)
    gW, gH = size(grid, 1), size(grid, 2)
    n_elem = gW * gH * xN

    dx = similar(x)
    dgrid = similar(grid)

    kernel = @cuda launch=false ∇grid_sampler_kernel!(
        n_elem, dx, dgrid, Δ, x, grid, padding_mode)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = min(n_elem, config.threads)
    blocks = cld(n_elem, threads)
    kernel(n_elem, dx, dgrid, Δ, x, grid, padding_mode; threads=threads, blocks=blocks)
    dx, dgrid
end
