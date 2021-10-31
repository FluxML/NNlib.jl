@inline in_bounds(h, w, H, W) = 1 ≤ h ≤ H && 1 ≤ w ≤ W
@inline clip_coordinate(coordinate, dim_size) = min(dim_size, max(1, coordinate))

# Borders are considered out-of-bounds.
@inline function ∇clip_coordinate(coordinate::C, dim_size) where C
    if coordinate ≤ 1
        return C(1), C(0)
    elseif coordinate ≥ dim_size
        return C(dim_size), C(0)
    end
    coordinate, C(1)
end

@inline unnormalize(coordinate, dim_size) = ((coordinate + 1.0) / 2.0) * (dim_size - 1.0) + 1.0
@inline ∇unnormalize(coordinate, dim_size) = unnormalize(coordinate, dim_size), (dim_size - 1.0) * 0.5

@inline compute_source_index(coordinate, dim_size, ::Val{:zeros}) = unnormalize(coordinate, dim_size)
@inline compute_source_index(coordinate, dim_size, ::Val{:border}) = clip_coordinate(unnormalize(coordinate, dim_size), dim_size)

@inline ∇compute_source_index(coordinate, dim_size, ::Val{:zeros}) = ∇unnormalize(coordinate, dim_size)
@inline function ∇compute_source_index(coordinate, dim_size, ::Val{:border})
    source_coordinate, grad_in = ∇unnormalize(coordinate, dim_size)
    source_coordinate, grad_clip = ∇clip_coordinate(source_coordinate, dim_size)
    source_coordinate, grad_in * grad_clip
end

"""
# Modes:
align corners - true only
padding mode - zeros, border
interpolation mode - billinear

# Arguments

- `input`: Input array in `WHCN` shape.
- `grid`: Input grid in `W'H'2N` shape. Where for each `W'H'N`
    grid contains `(x, y)` coordinates to sample from `input`.
    `W'` can be different from `W` (same for `H'`).
- `padding_mode`: Out-of-bound padding.
    `0` for zero-padding, `1` - for border padding.

# Returns:

`W'H'CN` sampled tensor from `x`.
"""
function grid_sampler(input, grid, padding_mode)
    T = eltype(input)
    _, _, iC, iN = size(input)
    _, gW, gH, _ = size(grid)
    output = similar(input, T, (gW, gH, iC, iN))
    grid_sampler!(output, input, grid, padding_mode)
end
function grid_sampler!(output, input, grid, padding_mode)
    iW, iH, iC, iN = size(input)
    _, gW, gH, _ = size(grid)
    # Loop over each output pixel.
    Threads.@threads for n in 1:iN
        for w in 1:gW, h in 1:gH
            # Get the corresponding (x, y) coordinates from the grid.
            @inbounds x, y = grid[1, w, h, n], grid[2, w, h, n]
            ix = compute_source_index(x, iW, padding_mode)
            iy = compute_source_index(y, iH, padding_mode)
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
            # ∀ channel: Calculate bilinear weighted pixel value.
            @inbounds for c in 1:iC
                r = 0.0
                if in_bounds(iy_nw, ix_nw, iH, iW)
                    r += input[ix_nw, iy_nw, c, n] * nw
                end
                if in_bounds(iy_ne, ix_ne, iH, iW)
                    r += input[ix_ne, iy_ne, c, n] * ne
                end
                if in_bounds(iy_sw, ix_sw, iH, iW)
                    r += input[ix_sw, iy_sw, c, n] * sw
                end
                if in_bounds(iy_se, ix_se, iH, iW)
                    r += input[ix_se, iy_se, c, n] * se
                end
                output[w, h, c, n] = r
            end
        end
    end
    output
end

"""
# Arguments:

- `∇output`: Gradient in `W'H'CN` shape
    (same shape as for the output from forward pass).
- `input`: Input from forward pass in `WHCN` shape.
- `grid`: Grid from forward pass in `W'H'2N` shape. Where for each `W'H'N`
    grid contains `(x, y)` coordinates to sample from `input`.
- `padding_mode`: Out-of-bound padding.
    `0` for zero-padding, `1` - for border padding.
    Should be the same as in the forward pass.
"""
function ∇grid_sampler(Δ, input, grid, padding_mode)
    T = eltype(input)
    dx = zeros(T, size(input))
    dgrid = similar(grid)
    ∇grid_sampler!(dx, dgrid, Δ, input, grid, padding_mode)
end
function ∇grid_sampler!(dx, dgrid, Δ, input, grid, padding_mode)
    iW, iH, iC, iN = size(input)
    gW, gH = size(grid, 2), size(grid, 3)
    # Loop over each output pixel.
    Threads.@threads for n in 1:iN
        for w in 1:gW, h in 1:gH
            # Get corresponding (x, y) from grid.
            @inbounds x, y = grid[1, w, h, n], grid[2, w, h, n]
            # Compute multipliers for gradinets on ix, iy.
            ix, gix_mult = ∇compute_source_index(x, iW, padding_mode)
            iy, giy_mult = ∇compute_source_index(y, iH, padding_mode)
            # Get corner pixel values from (ix, iy) in north-east-south-west directions.
            ix_nw, iy_nw = floor(Int, ix), floor(Int, iy)
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
                if in_bounds(iy_nw, ix_nw, iH, iW)
                    dx[ix_nw, iy_nw, c, n] += g_out * nw
                    nw_val = input[ix_nw, iy_nw, c, n]
                    gix -= nw_val * (iy_se - iy) * g_out
                    giy -= nw_val * (ix_se - ix) * g_out
                end
                if in_bounds(iy_ne, ix_ne, iH, iW)
                    dx[ix_ne, iy_ne, c, n] += g_out * ne
                    ne_val = input[ix_ne, iy_ne, c, n]
                    gix += ne_val * (iy_sw - iy) * g_out
                    giy -= ne_val * (ix - ix_sw) * g_out
                end
                if in_bounds(iy_sw, ix_sw, iH, iW)
                    dx[ix_sw, iy_sw, c, n] += g_out * sw
                    sw_val = input[ix_sw, iy_sw, c, n]
                    gix -= sw_val * (iy - iy_ne) * g_out
                    giy += sw_val * (ix_ne - ix) * g_out
                end
                if in_bounds(iy_se, ix_se, iH, iW)
                    dx[ix_se, iy_se, c, n] += g_out * se
                    se_val = input[ix_se, iy_se, c, n]
                    gix += se_val * (iy - iy_nw) * g_out
                    giy += se_val * (ix - ix_nw) * g_out
                end
            end
            @inbounds dgrid[1, w, h, n] = gix_mult * gix
            @inbounds dgrid[2, w, h, n] = giy_mult * giy
        end
    end
    dx, dgrid
end
