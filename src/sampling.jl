export grid_sample, ∇grid_sample

@inline in_bounds(h, w, H, W) = 1 ≤ h ≤ H && 1 ≤ w ≤ W
# Borders are considered out-of-bounds for gradient.
@inline clip_coordinate(coordinate, dim_size) = min(dim_size, max(1, coordinate))
@inline function ∇clip_coordinate(coordinate::C, dim_size) where C
    if coordinate ≤ 1
        return C(1), C(0)
    elseif coordinate ≥ dim_size
        return C(dim_size), C(0)
    end
    coordinate, C(1)
end

@inline unnormalize(coordinate, dim_size) = ((coordinate + 1.0) * 0.5) * (dim_size - 1.0) + 1.0
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
    grid_sample(input::AbstractArray{T, 4}, grid::AbstractArray{T, 4}; padding_mode = Val(:zeros))

Given `input`, compute output by sampling `input` values at pixel
locations from `grid`. Uses bilinear interpolation to calculate output values.

This implementation assumes the extrema (`-1` and `1`) are considered
as referring to the center points of the input’s corner pixels
(i.e. align corners is `true`).

# Arguments

- `input`: Input array in `(W_in, H_in, C, N)` shape.
- `grid`: Input grid in `(2, W_out, H_out, N)` shape.
    Where for each `(W_out, H_out, N)` grid contains `(x, y)`
    coordinates that specify sampling locations normalized by the `input` shape.

    Therefore, it should have values mostly in `[-1, 1]` range.
    For example, values `x = -1, y = -1` is the left-top pixel of `input`,
    and values `x = 1, y = 1` is the right-bottom pixel of `input`.

    Out-of-bound values are handled accroding to the `padding_mode`.
- `padding_mode`: Out-of-bound padding.
    `Val(:zeros)` to use `0` for out-of-bound grid locations.
    `Val(:border)` to use border values for out-of-bound grid locations.
    Default is `Val(:zeros)`.

# Returns

`(W_out, H_out, C, N)` sampled grid from `input`.

# Examples

In the example below, grid contains two out-of-bound sampling locations,
which are handled differently, depending on the `padding_mode`.

```jldoctest
julia> x = reshape(collect(1.0:4.0), (2, 2, 1, 1))
2×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 1.0  3.0
 2.0  4.0

julia> grid = Array{Float64}(undef, 2, 3, 2, 1);

julia> grid[:, 1, 1, 1] .= (-3, -1);

julia> grid[:, 2, 1, 1] .= (0, -1);

julia> grid[:, 3, 1, 1] .= (1, -1);

julia> grid[:, 1, 2, 1] .= (-1, 1);

julia> grid[:, 2, 2, 1] .= (0, 1);

julia> grid[:, 3, 2, 1] .= (3, 1);

julia> grid_sample(x, grid; padding_mode=Val(:zeros))
3×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  3.0
 1.5  3.5
 2.0  0.0

julia> grid_sample(x, grid; padding_mode=Val(:border))
3×2×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 1.0  3.0
 1.5  3.5
 2.0  4.0
```
"""
function grid_sample(input::AbstractArray{T, 4}, grid::AbstractArray{T, 4}; padding_mode = Val(:zeros)) where T
    _, _, iC, iN = size(input)
    _, gW, gH, _ = size(grid)
    output = similar(input, T, (gW, gH, iC, iN))
    grid_sample!(output, input, grid, padding_mode)
end
function grid_sample!(output, input, grid, padding_mode)
    iW, iH, iC, iN = size(input)
    _, gW, gH, _ = size(grid)
    # Loop over each output pixel.
    Threads.@threads for n in 1:iN
        for w in 1:gW, h in 1:gH
            _grid_sample_kernel!(
                output, input, grid, padding_mode, w, h, n, iW, iH, iC)
        end
    end
    output
end
@inline function _grid_sample_kernel!(
    output, input, grid, padding_mode, w, h, n, iW, iH, iC,
)
    # Get the corresponding (x, y) coordinates from the grid.
    @inbounds x, y = grid[1, w, h, n], grid[2, w, h, n]
    ix = compute_source_index(x, iW, padding_mode)
    iy = compute_source_index(y, iH, padding_mode)
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

"""
    ∇grid_sample(Δ::AbstractArray{T, 4}, input::AbstractArray{T, 4}, grid::AbstractArray{T, 4}; padding_mode = Val(:zeros)) where T

# Arguments

- `Δ`: Input gradient in `(W_out, H_out, C, N)` shape
    (same as output of the primal computation).
- `input`: Input from primal computation in `(W_in, H_in, C, N)` shape.
- `grid`: Grid from primal computation in `(2, W_out, H_out, N)` shape.
- `padding_mode`: Out-of-bound padding.
    `Val(:zeros)` to use `0` for out-of-bound grid locations.
    `Val(:border)` to use border values for out-of-bound grid locations.
    Should be the same as in primal computation.
    Default is `Val(:zeros)`.

# Returns

`dinput` (same shape as `input`) and `dgrid` (same shape as `grid`) gradients.
"""
function ∇grid_sample(Δ::AbstractArray{T, 4}, input::AbstractArray{T, 4}, grid::AbstractArray{T, 4}; padding_mode = Val(:zeros)) where T
    dx = zeros(T, size(input))
    dgrid = similar(grid)
    ∇grid_sample!(dx, dgrid, Δ, input, grid, padding_mode)
end
function ∇grid_sample!(dx, dgrid, Δ, input, grid, padding_mode)
    iW, iH, iC, iN = size(input)
    gW, gH = size(grid, 2), size(grid, 3)
    # Loop over each output pixel.
    Threads.@threads for n in 1:iN
        for w in 1:gW, h in 1:gH
            _∇grid_sample_kernel!(
                dx, dgrid, Δ, input, grid, padding_mode, w, h, n, iW, iH, iC)
        end
    end
    dx, dgrid
end
@inline function _∇grid_sample_kernel!(
    dx, dgrid, Δ, input, grid, padding_mode, w, h, n, iW, iH, iC,
)
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
            _safe_add!(dx, g_out * nw, ix_nw, iy_nw, c, n)
            nw_val = input[ix_nw, iy_nw, c, n]
            gix -= nw_val * (iy_se - iy) * g_out
            giy -= nw_val * (ix_se - ix) * g_out
        end
        if in_bounds(iy_ne, ix_ne, iH, iW)
            _safe_add!(dx, g_out * ne, ix_ne, iy_ne, c, n)
            ne_val = input[ix_ne, iy_ne, c, n]
            gix += ne_val * (iy_sw - iy) * g_out
            giy -= ne_val * (ix - ix_sw) * g_out
        end
        if in_bounds(iy_sw, ix_sw, iH, iW)
            _safe_add!(dx, g_out * sw, ix_sw, iy_sw, c, n)
            sw_val = input[ix_sw, iy_sw, c, n]
            gix -= sw_val * (iy - iy_ne) * g_out
            giy += sw_val * (ix_ne - ix) * g_out
        end
        if in_bounds(iy_se, ix_se, iH, iW)
            _safe_add!(dx, g_out * se, ix_se, iy_se, c, n)
            se_val = input[ix_se, iy_se, c, n]
            gix += se_val * (iy - iy_nw) * g_out
            giy += se_val * (ix - ix_nw) * g_out
        end
    end
    @inbounds dgrid[1, w, h, n] = gix_mult * gix
    @inbounds dgrid[2, w, h, n] = giy_mult * giy
end

@inline function _safe_add!(dx, value, ix, iy, c, n)
    @inbounds dx[ix, iy, c, n] += value
end

function rrule(::typeof(grid_sample), x, grid; padding_mode)
    y = grid_sample(x, grid; padding_mode=padding_mode)
    function grid_sample_pullback(Δ)
        ∇x, ∇grid = @thunk(∇grid_sample(unthunk(Δ), x, grid; padding_mode=padding_mode))
        NoTangent(), ∇x, ∇grid
    end
    return y, grid_sample_pullback
end
