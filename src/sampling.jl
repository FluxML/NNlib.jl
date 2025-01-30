@inline in_bounds(h, w, H, W) = 1 ≤ h ≤ H && 1 ≤ w ≤ W
@inline in_bounds(h, w, d, H, W, D) = 1 ≤ h ≤ H && 1 ≤ w ≤ W && 1 ≤ d ≤ D
# Borders are considered out-of-bounds for gradient.
@inline clip_coordinate(coordinate, dim_size) = min(dim_size, max(1, coordinate))
@inline function ∇clip_coordinate(coordinate::C, dim_size) where {C}
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
    grid_sample(input::AbstractArray{T, 4}, grid::AbstractArray{T, 4}; padding_mode = :zeros)
    grid_sample(input::AbstractArray{T, 5}, grid::AbstractArray{T, 4}; padding_mode = :zeros)

    Given `input`, compute output by sampling `input` values at pixel
    locations from `grid`. Uses bilinear interpolation to calculate output values.

    This implementation assumes the extrema (`-1` and `1`) are considered
    as referring to the center points of the input’s corner pixels
    (i.e. align corners is `true`).

    # Arguments

    - `input`: Input array in `(W_in, H_in, [D_in,] C, N)` shape.
    - `grid`: Input grid in `(2, W_out, H_out, [D_out,] N)` shape.
        Where for each `(W_out, H_out, [D_out,] N)` grid contains `(x, y [,z])`
        coordinates that specify sampling locations normalized by the `input` shape.

        Therefore, `x`, `y` and [`z`] should have values in `[-1, 1]` range.
        For example, `(x = -1, y = -1, [z = -1])` is the left-top[-front] pixel of `input`,
        and `(x = 1, y = 1, [z = 1])` is the right-bottom-back pixel of `input`.

        Out-of-bound values are handled according to the `padding_mode`.
    - `padding_mode`: Out-of-bound padding.
        `:zeros` to use `0` for out-of-bound grid locations.
        `:border` to use border values for out-of-bound grid locations.
        Default is `:zeros`.

    # Returns

    `(W_out, H_out, [D_out,] C, N)` sampled grid from `input`.

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

    julia> grid_sample(x, grid; padding_mode=:zeros)
    3×2×1×1 Array{Float64, 4}:
    [:, :, 1, 1] =
    0.0  3.0
    1.5  3.5
    2.0  0.0

    julia> grid_sample(x, grid; padding_mode=:border)
    3×2×1×1 Array{Float64, 4}:
    [:, :, 1, 1] =
    1.0  3.0
    1.5  3.5
    2.0  4.0
    ```
"""
function grid_sample(input::AbstractArray{T,N}, grid; padding_mode = :zeros) where {T,N}
    if N ∉ (4,5)
        error("grid_sample is only supported for 4D and 5D arrays.") 
    end
    iC, iN = size(input)[end-1:end] 
    output_size = size(grid)[2:end-1] # W_out, H_out, [D_out]
    output = similar(input, T, (output_size..., iC, iN))
    grid_sample!(output, input, grid, padding_mode)
end

function grid_sample!(output::AbstractArray{T,4}, input::AbstractArray{T,4}, grid, padding_mode=:zeros) where {T}
    pad = Val(padding_mode)
    iW, iH, iC, iN = size(input)
    _, gW, gH, _ = size(grid)
    # Loop over each output pixel.
    Threads.@threads for n in 1:iN
        for w in 1:gW, h in 1:gH
            _grid_sample_kernel!(output, input, grid, pad, w, h, n, iW, iH, iC)
        end
    end
    output
end

function grid_sample!(output::AbstractArray{T,5}, input::AbstractArray{T,5}, grid, padding_mode=:zeros) where {T}
    pad = Val(padding_mode)
    iW, iH, iD, iC, iN = size(input)
    _, gW, gH, gD, _ = size(grid)
    # Loop over each output pixel.
    Threads.@threads for n in 1:iN
        for w in 1:gW, h in 1:gH, d in 1:gD
            _grid_sample_kernel!(output, input, grid, pad, w, h, d, n, iW, iH, iD, iC)
        end
    end
    output
end

@inline function _grid_sample_kernel!(
    output::AbstractArray{T,4}, input::AbstractArray{T,4}, grid, padding_mode, w, h, n, iW, iH, iC,
) where {T}
    # Get the corresponding (x, y) coordinates from the grid.
    @inbounds x, y = grid[1, w, h, n], grid[2, w, h, n]
    ix = compute_source_index(x, iW, padding_mode)
    iy = compute_source_index(y, iH, padding_mode)
    # Get corner pixel values from (ix, iy) in north-east-south-west directions.
    ix_nw, iy_nw = unsafe_trunc(Int, floor(ix)), unsafe_trunc(Int, floor(iy))
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
        r = zero(T)
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

@inline function _grid_sample_kernel!(
    output::AbstractArray{T,5}, input::AbstractArray{T,5}, grid, padding_mode, w, h, d, n, iW, iH, iD, iC,
) where {T}
    # Get the corresponding (x, y, z) coordinates from the grid.
    @inbounds x, y, z = grid[1, w, h, d, n], grid[2, w, h, d, n], grid[3, w, h, d, n]
    ix = compute_source_index(x, iW, padding_mode)
    iy = compute_source_index(y, iH, padding_mode)
    iz = compute_source_index(z, iD, padding_mode)

    # Get corner voxel values from (ix, iy, iz) in 8 directions (north-east-south-west-bottom-up).
    ix_nw, iy_nw, iz_nw = unsafe_trunc(Int, floor(ix)), unsafe_trunc(Int, floor(iy)), unsafe_trunc(Int, floor(iz))
    ix_ne, iy_ne, iz_ne = ix_nw + 1, iy_nw, iz_nw
    ix_sw, iy_sw, iz_sw = ix_nw, iy_nw + 1, iz_nw
    ix_se, iy_se, iz_se = ix_ne, iy_sw, iz_nw
    ix_nw_u, iy_nw_u, iz_nw_u = ix_nw, iy_nw, iz_nw + 1
    ix_ne_u, iy_ne_u, iz_ne_u = ix_ne, iy_ne, iz_ne + 1
    ix_sw_u, iy_sw_u, iz_sw_u = ix_sw, iy_sw, iz_sw + 1
    ix_se_u, iy_se_u, iz_se_u = ix_se, iy_se, iz_se + 1

    # Get volumes to each neighbor (a.k.a. interpolation weights).
    nw = (ix_se - ix) * (iy_se - iy) * (iz_se_u - iz)
    ne = (ix - ix_sw) * (iy_sw - iy) * (iz_sw_u - iz)
    sw = (ix_ne - ix) * (iy - iy_ne) * (iz_ne_u - iz)
    se = (ix - ix_nw) * (iy - iy_nw) * (iz_nw_u - iz)
    nw_u = (ix_se - ix) * (iy_se - iy) * (iz - iz_nw)
    ne_u = (ix - ix_sw) * (iy_sw - iy) * (iz - iz_sw)
    sw_u = (ix_ne - ix) * (iy - iy_ne) * (iz - iz_ne)
    se_u = (ix - ix_nw) * (iy - iy_nw) * (iz - iz_nw)

    # ∀ channel: Calculate trilinear weighted voxel value.
    @inbounds for c in 1:iC
        r = zero(T)
        if in_bounds(iy_nw, ix_nw, iz_nw, iH, iW, iD)
            r += input[ix_nw, iy_nw, iz_nw, c, n] * nw
        end
        if in_bounds(iy_ne, ix_ne, iz_ne, iH, iW, iD)
            r += input[ix_ne, iy_ne, iz_ne, c, n] * ne
        end
        if in_bounds(iy_sw, ix_sw, iz_sw, iH, iW, iD)
            r += input[ix_sw, iy_sw, iz_sw, c, n] * sw
        end
        if in_bounds(iy_se, ix_se, iz_se, iH, iW, iD)
            r += input[ix_se, iy_se, iz_se, c, n] * se
        end
        if in_bounds(iy_nw_u, ix_nw_u, iz_nw_u, iH, iW, iD)
            r += input[ix_nw_u, iy_nw_u, iz_nw_u, c, n] * nw_u
        end
        if in_bounds(iy_ne_u, ix_ne_u, iz_ne_u, iH, iW, iD)
            r += input[ix_ne_u, iy_ne_u, iz_ne_u, c, n] * ne_u
        end
        if in_bounds(iy_sw_u, ix_sw_u, iz_sw_u, iH, iW, iD)
            r += input[ix_sw_u, iy_sw_u, iz_sw_u, c, n] * sw_u
        end
        if in_bounds(iy_se_u, ix_se_u, iz_se_u, iH, iW, iD)
            r += input[ix_se_u, iy_se_u, iz_se_u, c, n] * se_u
        end
        output[w, h, d, c, n] = r
    end
end


"""
    ∇grid_sample(Δ::AbstractArray{T, 4}, input::AbstractArray{T, 4}, grid::AbstractArray{T, 4}; padding_mode = :zeros) where T

# Arguments

- `Δ`: Input gradient in `(W_out, H_out, C, N)` shape
    (same as output of the primal computation).
- `input`: Input from primal computation in `(W_in, H_in, C, N)` shape.
- `grid`: Grid from primal computation in `(2, W_out, H_out, N)` shape.
- `padding_mode`: Out-of-bound padding.
    `:zeros` to use `0` for out-of-bound grid locations.
    `:border` to use border values for out-of-bound grid locations.
    Should be the same as in primal computation.
    Default is `:zeros`.

# Returns

`dinput` (same shape as `input`) and `dgrid` (same shape as `grid`) gradients.
"""
function ∇grid_sample(Δ::AbstractArray{T,N}, input::AbstractArray{T,N}, grid; padding_mode=:zeros) where {T, N}
    if N ∉ (4,5)
        error("∇grid_sample is only supported for 4D and 5D arrays.") 
    end
    dx = zeros(T, size(input))
    dgrid = similar(grid)
    ∇grid_sample!(dx, dgrid, Δ, input, grid, padding_mode)
end

function ∇grid_sample!(dx::AbstractArray{T,4}, dgrid::AbstractArray{T,4}, Δ::AbstractArray{T,4}, input::AbstractArray{T,4}, grid::AbstractArray{T,4}, padding_mode) where {T}
    pad = Val(padding_mode)
    iW, iH, iC, iN = size(input)
    gW, gH = size(grid, 2), size(grid, 3)
    # Loop over each output pixel.
    Threads.@threads for n in 1:iN
        for w in 1:gW, h in 1:gH
            _∇grid_sample_kernel!(dx, dgrid, Δ, input, grid, pad, w, h, n, iW, iH, iC)
        end
    end
    dx, dgrid
end

function ∇grid_sample!(dx::AbstractArray{T,5}, dgrid::AbstractArray{T,5}, Δ::AbstractArray{T,5}, input::AbstractArray{T,5}, grid::AbstractArray{T,5}, padding_mode) where {T}
    pad = Val(padding_mode)
    iW, iH, iD, iC, iN = size(input)
    gW, gH, gD = size(grid, 2), size(grid, 3), size(grid, 4)
    # Loop over each output voxel.
    Threads.@threads for n in 1:iN
        for w in 1:gW, h in 1:gH, d in 1:gD
            _∇grid_sample_kernel!(dx, dgrid, Δ, input, grid, pad, w, h, d, n, iW, iH, iD, iC)
        end
    end
    dx, dgrid
end

@inline function _∇grid_sample_kernel!(
    dx::AbstractArray{T,4}, dgrid::AbstractArray{V,4}, Δ::AbstractArray{T,4}, input::AbstractArray{T,4}, grid::AbstractArray{V,4}, padding_mode, w, h, n, iW, iH, iC,
) where {T,V}
    # Get corresponding (x, y) from grid.
    @inbounds x, y = grid[1, w, h, n], grid[2, w, h, n]
    # Compute multipliers for gradients on ix, iy.
    ix, gix_mult = ∇compute_source_index(x, iW, padding_mode)
    iy, giy_mult = ∇compute_source_index(y, iH, padding_mode)
    # Get corner pixel values from (ix, iy) in north-east-south-west directions.
    ix_nw, iy_nw = unsafe_trunc(Int, floor(ix)), unsafe_trunc(Int, floor(iy))
    ix_ne, iy_ne = ix_nw + 1, iy_nw
    ix_sw, iy_sw = ix_nw, iy_nw + 1
    ix_se, iy_se = ix_ne, iy_sw
    # Get surfaces to each neighbor (a.k.a. interpolation weights).
    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)
    # ∀ channel: Calculate billinear weighted pixel value.
    gix, giy = zero(V), zero(V)
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

@inline function _∇grid_sample_kernel!(
    dx::AbstractArray{T,5}, dgrid::AbstractArray{V,5}, Δ::AbstractArray{T,5}, input::AbstractArray{T,5}, grid::AbstractArray{V,5}, padding_mode, w, h, d, n, iW, iH, iD, iC,
) where {T,V}
    # Get corresponding (x, y, z) from grid.
    @inbounds x, y, z = grid[1, w, h, d, n], grid[2, w, h, d, n], grid[3, w, h, d, n]
    # Compute multipliers for gradients on ix, iy, iz.
    ix, gix_mult = ∇compute_source_index(x, iW, padding_mode)
    iy, giy_mult = ∇compute_source_index(y, iH, padding_mode)
    iz, giz_mult = ∇compute_source_index(z, iD, padding_mode)
     
    # Get corner pixel values from (ix, iy, iz)
    ix_0 = unsafe_trunc(Int, floor(ix))
    iy_0 = unsafe_trunc(Int, floor(iy))
    iz_0 = unsafe_trunc(Int, floor(iz))
    ix_1 = ix_0 + 1
    iy_1 = iy_0 + 1
    iz_1 = iz_0 + 1
    
    # Get difference of coordinate
    wx_0 = ix - ix_0
    wy_0 = iy - iy_0
    wz_0 = iz - iz_0
    wx_1 = ix_1 - ix
    wy_1 = iy_1 - iy
    wz_1 = iz_1 - iz
    
    # Calculate weights (volume of diagnal vertex cube) 
    # w_{abc} = wx_{¬a}*wy_{¬b}*wz_{¬c}
    weight_000 = wx_1 * wy_1 * wz_1
    weight_001 = wx_1 * wy_1 * wz_0
    weight_010 = wx_1 * wy_0 * wz_1
    weight_011 = wx_1 * wy_0 * wz_0
    weight_100 = wx_0 * wy_1 * wz_1
    weight_101 = wx_0 * wy_1 * wz_0
    weight_110 = wx_0 * wy_0 * wz_1
    weight_111 = wx_0 * wy_0 * wz_0

    # ∂w_{abc}/∂x=(-1)^{¬a} wy_{¬b}*wz_{¬c}, ∂w/∂y = (-1)^{¬b} wx_{¬a}*wz_{¬c}, ∂w/∂z=(-1)^{¬c} wx_{¬a}*wy_{¬b}
    # abc are the index of the vertex of the cube (001,010...)

    # Initialize gradient accumulators
    gix, giy, giz = zero(V), zero(V), zero(V)
    
    @inbounds for c in 1:iC
        g_out = Δ[w, h, d, c, n]
        
        # Calculate dx and dgrid partials for all 8 corners
        if in_bounds(iy_0, ix_0, iz_0, iH, iW, iD)
            _safe_add!(dx, g_out * weight_000, ix_0, iy_0, iz_0, c, n)
            val = input[ix_0, iy_0, iz_0, c, n]
            gix -= val * wy_1 * wz_1 * g_out
            giy -= val * wx_1 * wz_1 * g_out
            giz -= val * wx_1 * wy_1 * g_out
        end

        if in_bounds(iy_0, ix_0, iz_1, iH, iW, iD)
            _safe_add!(dx, g_out * weight_001, ix_0, iy_0, iz_1, c, n)
            val = input[ix_0, iy_0, iz_1, c, n]
            gix -= val * wy_1 * wz_0 * g_out
            giy -= val * wx_1 * wz_0 * g_out
            giz += val * wx_1 * wy_1 * g_out
        end
        
        if in_bounds(iy_1, ix_0, iz_0, iH, iW, iD)
            _safe_add!(dx, g_out * weight_010, ix_0, iy_1, iz_0, c, n)
            val = input[ix_0, iy_1, iz_0, c, n]
            gix -= val * wy_0 * wz_1 * g_out
            giy += val * wx_1 * wz_1 * g_out
            giz -= val * wx_1 * wy_0 * g_out
        end
        
        if in_bounds(iy_1, ix_0, iz_1, iH, iW, iD)
            _safe_add!(dx, g_out * weight_011, ix_0, iy_1, iz_1, c, n)
            val = input[ix_0, iy_1, iz_1, c, n]
            gix -= val * wy_0 * wz_0 * g_out
            giy += val * wx_1 * wz_0 * g_out
            giz += val * wx_1 * wy_0 * g_out
        end

        if in_bounds(iy_0, ix_1, iz_0, iH, iW, iD)
            _safe_add!(dx, g_out * weight_100, ix_1, iy_0, iz_0, c, n)
            val = input[ix_1, iy_0, iz_0, c, n]
            gix += val * wy_1 * wz_1 * g_out
            giy -= val * wx_0 * wz_1 * g_out
            giz -= val * wx_0 * wy_1 * g_out
        end
        
        if in_bounds(iy_0, ix_1, iz_1, iH, iW, iD)
            _safe_add!(dx, g_out * weight_101, ix_1, iy_0, iz_1, c, n)
            val = input[ix_1, iy_0, iz_1, c, n]
            gix += val * wy_1 * wz_0 * g_out
            giy -= val * wx_0 * wz_0 * g_out
            giz += val * wx_0 * wy_1 * g_out
        end

        if in_bounds(iy_1, ix_1, iz_0, iH, iW, iD)
            _safe_add!(dx, g_out * weight_110, ix_1, iy_1, iz_0, c, n)
            val = input[ix_1, iy_1, iz_0, c, n]
            gix += val * wy_0 * wz_1 * g_out
            giy += val * wx_0 * wz_1 * g_out
            giz -= val * wx_0 * wy_0 * g_out
        end
        
        if in_bounds(iy_1, ix_1, iz_1, iH, iW, iD)
            _safe_add!(dx, g_out * weight_111, ix_1, iy_1, iz_1, c, n)
            val = input[ix_1, iy_1, iz_1, c, n]
            gix += val * wy_0 * wz_0 * g_out
            giy += val * wx_0 * wz_0 * g_out
            giz += val * wx_0 * wy_0 * g_out
        end
    end
    
    @inbounds dgrid[1, w, h, d, n] = gix_mult * gix
    @inbounds dgrid[2, w, h, d, n] = giy_mult * giy
    @inbounds dgrid[3, w, h, d, n] = giz_mult * giz
end

@inline function _safe_add!(dx, value, ix, iy, c, n)
    @inbounds dx[ix, iy, c, n] += value
end

@inline function _safe_add!(dx, value, ix, iy, iz, c, n)
    @inbounds dx[ix, iy, iz, c, n] += value
end

function rrule(::typeof(grid_sample), x, grid; padding_mode)
    y = grid_sample(x, grid; padding_mode=padding_mode)
    function grid_sample_pullback(Δ)
        ∇x, ∇grid = ∇grid_sample(unthunk(Δ), x, grid; padding_mode=padding_mode)
        NoTangent(), ∇x, ∇grid
    end
    return y, grid_sample_pullback
end
