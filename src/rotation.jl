# this rotates the coordinates and either applies round(nearest neighbour)
# or floor (:bilinear interpolation)
@inline function rotate_coordinates(sinθ, cosθ, i, j, midpoint, round_or_floor)
    y = i - midpoint[1]
    x = j - midpoint[2]
    yrot = cosθ * y - sinθ * x + midpoint[1]
    xrot = sinθ * y + cosθ * x + midpoint[2]
    yrot_f = round_or_floor(yrot)
    xrot_f = round_or_floor(xrot)
    yrot_int = round_or_floor(Int, yrot)
    xrot_int = round_or_floor(Int, xrot)
    return yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int
end


# helper function for bilinear
@inline function bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int, imax, jmax)
        xdiff = (xrot - xrot_f)
        xdiff_diff = 1 - xdiff
        ydiff = (yrot - yrot_f)
        ydiff_diff = 1 - ydiff
        # in case we hit the boundary stripe, then we need to avoid out of bounds access
        Δi = 1#yrot_int != imax
        Δj = 1#xrot_int != jmax
      
        # we need to avoid that we access arr[0]
        # in rare cases the rounding clips off values on the left and top border
        # we still try to access them with this extra comparison
        Δi_min = 0#yrot_int == 0
        Δj_min = 0#xrot_int == 0
        return Δi, Δj, Δi_min, Δj_min, ydiff, ydiff_diff, xdiff, xdiff_diff
end

"""
    _prepare_imrotate(arr, θ, midpoint)

Prepate `sin` and `cos`, creates the output array and converts type
of `midpoint` if required.
"""
function _prepare_imrotate(arr::AbstractArray{T}, θ, midpoint) where T
    # needed for rotation matrix
    θ = mod(real(T)(θ), real(T)(2π))
    midpoint = real(T).(midpoint)
    sinθ, cosθ = sincos(real(T)(θ)) 
    out = similar(arr)
    fill!(out, 0)
    return sinθ, cosθ, midpoint, out
end

"""
    _check_trivial_rotations!(out, arr, θ, midpoint) 

When `θ = 0 || π /2 || π || 3/2 || π` and if `midpoint` 
is in the middle of the array.
For an even array of size 4, the midpoint would need to be 2.5.
For an odd array of size 5, the midpoint would need to be 3.

In those cases, rotations are trivial just by reversing or swapping some axes.
"""
function _check_trivial_rotations!(out, arr, θ, midpoint)
    if iszero(θ)
        out .= arr
        return true 
    end
    # check for special cases where rotations are trivial
    if ((midpoint[1] ≈ size(arr, 1) ÷ 2 + 0.5 && midpoint[2] ≈ size(arr, 2) ÷ 2 + 0.5) ||
        (midpoint[1] == size(arr, 1) ÷ 2 + 1 && midpoint[1] == size(arr, 2) ÷ 2 + 1))
        if θ ≈ π / 2 
            out .= reverse(PermutedDimsArray(out, (2, 1, 3, 4)), dims=(2,))
            return true
        elseif θ ≈ π
            out .= reverse(out, dims=(1,2))
            return true
        elseif θ ≈ 3 / 2 * π
            out .= reverse(PermutedDimsArray(out, (2, 1, 3, 4)), dims=(1,))
            return true
        end
    end

    return false
end


"""
    imrotate(arr::AbstractArray{T, 4}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1)

Rotates a matrix around the center pixel `midpoint`.
The angle `θ` is interpreted in radians.

The adjoint is defined with ChainRulesCore.jl. This method also runs with CUDA (and in principle all KernelAbstractions.jl supported backends).

# Keywords
* `method=:bilinear` for bilinear interpolation or `method=:nearest` for nearest neighbour
* `midpoint=size(arr) .÷ 2 .+ 1` means there is always a real center pixel around it is rotated.

# Examples
```julia-repl

```
"""
function imrotate(arr::AbstractArray{T, 4}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1,
                  adjoint=false) where T
    @assert (T <: Integer && method==:nearest || !(T <: Integer)) "If the array has an Int eltype, only method=:nearest is supported"
    @assert typeof(midpoint) <: Tuple "midpoint keyword has to be a tuple"
    
    # prepare out, the sin and cos and type of midpoint
    sinθ, cosθ, midpoint, out = _prepare_imrotate(arr, θ, midpoint) 
    # such as 0°, 90°, 180°, 270° and only if the midpoint is suitable
    _check_trivial_rotations!(out, arr, θ, midpoint) && return out

    # KernelAbstractions specific
    backend = get_backend(arr)
    if method == :bilinear
        kernel! = imrotate_kernel_bilinear!(backend)
    elseif method == :nearest
        kernel! = imrotate_kernel_nearest!(backend)
    else 
        throw(ArgumentError("No interpolation method such as $method"))
    end
    kernel!(out, arr, sinθ, cosθ, midpoint, size(arr, 1), size(arr, 2),
            ndrange=(size(arr, 1), size(arr, 2), size(arr, 3), size(arr, 4)))
	return out
end

function ∇imrotate(arr::AbstractArray{T, 4}, θ; method=:bilinear, 
                                               midpoint=size(arr) .÷ 2 .+ 1) where T
    
    sinθ, cosθ, midpoint, out = _prepare_imrotate(arr, θ, midpoint) 
    _check_trivial_rotations!(out, arr, θ, midpoint) || return out

    backend = get_backend(arr)
    if method == :bilinear
        kernel! = ∇imrotate_kernel_bilinear_adj!(backend)
    elseif method == :nearest
        kernel! = ∇imrotate_kernel_nearest_adj!(backend)
    else 
        throw(ArgumentError("No interpolation method such as $method"))
    end
    kernel!(out, arr, sinθ, cosθ, midpoint, size(arr, 1), size(arr, 2),
            ndrange=(size(arr, 1), size(arr, 2), size(arr, 3), size(arr, 4)))
    return out
end


@kernel function imrotate_kernel_nearest!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, c, b = @index(Global, NTuple)

    _, _, _, _, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, round) 
    if 1 ≤ yrot_int ≤ imax && 1 ≤ xrot_int ≤ jmax
        @inbounds out[i, j, c, b] = arr[yrot_int, xrot_int, c, b]
    end
end


@kernel function imrotate_kernel_bilinear!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, c, b = @index(Global, NTuple)
    
    yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, floor) 
    if 1 ≤ yrot_int ≤ imax - 1&& 1 ≤ xrot_int ≤ jmax - 1 

        Δi, Δj, Δi_min, Δj_min, ydiff, ydiff_diff, xdiff, xdiff_diff = 
            bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int, imax, jmax)
        @inbounds out[i, j, c, b] = 
            (   xdiff_diff  * ydiff_diff    * arr[yrot_int + Δi_min, xrot_int + Δj_min, c, b]
             +  xdiff_diff  * ydiff         * arr[yrot_int + Δi,     xrot_int + Δj_min, c, b]
             +  xdiff       * ydiff_diff    * arr[yrot_int + Δi_min, xrot_int + Δj,     c, b] 
             +  xdiff       * ydiff         * arr[yrot_int + Δi,     xrot_int + Δj,     c, b])
    end
end


@kernel function ∇imrotate_kernel_nearest!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, c, b = @index(Global, NTuple)

    _, _, _, _, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, round) 
    if 1 ≤ yrot_int ≤ imax && 1 ≤ xrot_int ≤ jmax 
        Atomix.@atomic out[yrot_int, xrot_int, c, b] += arr[i, j, c, b]
    end
end


@kernel function ∇imrotate_kernel_bilinear!(out, arr, sinθ, cosθ, midpoint, imax, jmax)
    i, j, c, b = @index(Global, NTuple)

    yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int = rotate_coordinates(sinθ, cosθ, i, j, midpoint, floor) 
    if 1 ≤ yrot_int ≤ imax - 1 && 1 ≤ xrot_int ≤ jmax - 1
        o = arr[i, j, c, b]
        Δi, Δj, Δi_min, Δj_min, ydiff, ydiff_diff, xdiff, xdiff_diff = 
            bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int, imax, jmax)
        Atomix.@atomic out[yrot_int + Δi_min,   xrot_int + Δj_min, c, b]  += (1 - xdiff)  * (1 - ydiff) * o
        Atomix.@atomic out[yrot_int + Δi,       xrot_int + Δj_min, c, b]  += (1 - xdiff)  * ydiff       * o
        Atomix.@atomic out[yrot_int + Δi_min,   xrot_int + Δj,     c, b]  += xdiff        * (1 - ydiff) * o
        Atomix.@atomic out[yrot_int + Δi,       xrot_int + Δj,     c, b]  += xdiff        * ydiff       * o
    end
end


# is this rrule good? 
# no @thunk and @unthunk
function ChainRulesCore.rrule(::typeof(imrotate), array, θ; method=:bilinear, midpoint=size(array) .÷ 2 .+ 1)
    res = imrotate(array, θ; method, midpoint)
    function pb_rotate(dy)
        ad = imrotate(dy, θ; method, midpoint, adjoint=true)
        return NoTangent(), ad, NoTangent()
    end    
	return res, pb_rotate
end
