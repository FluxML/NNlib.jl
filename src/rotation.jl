"""
    rotate_coordinates(sinθ, cosθ, i, j, midpoint, round_or_floor)

this rotates the coordinates and either applies round(nearest neighbour)
or floor for :bilinear interpolation)
"""
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


"""
   bilinear_helper(yrot, xrot, yrot_f, xrot_f, yrot_int, xrot_int) 

Some helper variables
"""
@inline function bilinear_helper(yrot, xrot, yrot_f, xrot_f)
    xdiff = (xrot - xrot_f)
    xdiff_1minus = 1 - xdiff
    ydiff = (yrot - yrot_f)
    ydiff_1minus = 1 - ydiff
    
    return ydiff, ydiff_1minus, xdiff, xdiff_1minus
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
function _check_trivial_rotations!(out, arr, θ, midpoint; adjoint=false)
    if iszero(θ)
        out .= arr
        return true 
    end
    # check for special cases where rotations are trivial
    if (iseven(size(arr, 1)) && iseven(size(arr, 2)) && 
        midpoint[1] ≈ size(arr, 1) ÷ 2 + 0.5 && midpoint[2] ≈ size(arr, 2) ÷ 2 + 0.5) ||
        (isodd(size(arr, 1)) && isodd(size(arr, 2)) && 
        (midpoint[1] == size(arr, 1) ÷ 2 + 1 && midpoint[1] == size(arr, 2) ÷ 2 + 1))
        if θ ≈ π / 2 
            if adjoint == false
                out .= reverse(PermutedDimsArray(arr, (2, 1, 3, 4)), dims=(2,))
            else
                out .= reverse(PermutedDimsArray(arr, (2, 1, 3, 4)), dims=(1,))
            end
            return true
        elseif θ ≈ π
            out .= reverse(arr, dims=(1,2))
            return true
        elseif θ ≈ 3 / 2 * π
            if adjoint == false
                out .= reverse(PermutedDimsArray(arr, (2, 1, 3, 4)), dims=(1,))
            else
                out .= reverse(PermutedDimsArray(arr, (2, 1, 3, 4)), dims=(2,))
            end
            return true
        end
    end

    return false
end


"""
    imrotate(arr::AbstractArray{T, 4}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1)

Rotates a matrix around the center pixel `midpoint`. `midpoint` is defined such that there
is a real center pixel for even and odd values which is rotated around.
The angle `θ` is interpreted in radians.

The adjoint is defined with ChainRulesCore.jl. This method also runs with CUDA (and in principle all KernelAbstractions.jl supported backends).

# Keywords
* `method=:bilinear` for bilinear interpolation or `method=:nearest` for nearest neighbour
* `midpoint=size(arr) .÷ 2 .+ 1` means there is a real center pixel around it is rotated.

# Examples
```julia-repl
julia> arr = zeros((4,4,1,1)); arr[2,2,1,1] = 1;

julia> arr
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> NNlib.imrotate(arr, deg2rad(90)) # rotation around (3,3)
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  1.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> NNlib.imrotate(arr, deg2rad(90), midpoint=(2,2))
4×4×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  0.0  0.0
 0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0
 0.0  0.0  0.0  0.0

julia> arr = zeros((3,3,1,1)); arr[1,2,1,1] = 1
1

julia> arr
3×3×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  1.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> NNlib.imrotate(arr, deg2rad(45))
3×3×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.207107  0.0
 0.0  0.0       0.207107
 0.0  0.0       0.0

julia> NNlib.imrotate(arr, deg2rad(45), method=:nearest)
3×3×1×1 Array{Float64, 4}:
[:, :, 1, 1] =
 0.0  0.0  1.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```
"""
function imrotate(arr::AbstractArray{T, 4}, θ; method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1) where T
    @assert (T <: Integer && method==:nearest || !(T <: Integer)) "If the array has an Int eltype, only method=:nearest is supported"
    @assert typeof(midpoint) <: Tuple "midpoint keyword has to be a tuple"
    
    # prepare out, the sin and cos and type of midpoint
    sinθ, cosθ, midpoint, out = _prepare_imrotate(arr, θ, midpoint) 
    # such as 0°, 90°, 180°, 270° and only if the midpoint is suitable
    _check_trivial_rotations!(out, arr, θ, midpoint) && return out

    # KernelAbstractions specific
    backend = KernelAbstractions.get_backend(arr)
    if method == :bilinear
        kernel! = imrotate_kernel_bilinear!(backend)
    elseif method == :nearest
        kernel! = imrotate_kernel_nearest!(backend)
    else 
        throw(ArgumentError("No interpolation method such as $method"))
    end
    kernel!(out, arr, sinθ, cosθ, midpoint, size(arr, 1), size(arr, 2),
            ndrange=size(arr))
	return out
end


"""
    ∇imrotate(dy, arr::AbstractArray{T, 4}, θ; method=:bilinear,
                                               midpoint=size(arr) .÷ 2 .+ 1)

Adjoint for `imrotate`. Gradient only with respect to `arr` and not `θ`.

# Arguments
* `dy`: input gradient 
* `arr`: Input from primal computation
* `θ`: rotation angle in radians
* `method=:bilinear` or `method=:nearest`
* `midpoint=size(arr) .÷ 2 .+ 1` rotates around a real center pixel for even and odd sized arrays
"""
function ∇imrotate(dy, arr::AbstractArray{T, 4}, θ; method=:bilinear, 
                                               midpoint=size(arr) .÷ 2 .+ 1) where T
    
    sinθ, cosθ, midpoint, out = _prepare_imrotate(arr, θ, midpoint) 
    # for the adjoint, the trivial rotations go in the other direction!
    # pass dy and not arr
    _check_trivial_rotations!(out, dy, θ, midpoint, adjoint=true) && return out

    backend = KernelAbstractions.get_backend(arr)
    if method == :bilinear
        kernel! = ∇imrotate_kernel_bilinear!(backend)
    elseif method == :nearest
        kernel! = ∇imrotate_kernel_nearest!(backend)
    else 
        throw(ArgumentError("No interpolation method such as $method"))
    end
    # don't pass arr but dy! 
    kernel!(out, dy, sinθ, cosθ, midpoint, size(arr, 1), size(arr, 2),
            ndrange=size(arr))
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
    if 1 ≤ yrot_int ≤ imax - 1 && 1 ≤ xrot_int ≤ jmax - 1 

        ydiff, ydiff_1minus, xdiff, xdiff_1minus = 
            bilinear_helper(yrot, xrot, yrot_f, xrot_f)
        @inbounds out[i, j, c, b] = 
            (   xdiff_1minus    * ydiff_1minus  * arr[yrot_int      , xrot_int      , c, b]
             +  xdiff_1minus    * ydiff         * arr[yrot_int + 1  , xrot_int      , c, b]
             +  xdiff           * ydiff_1minus  * arr[yrot_int      , xrot_int + 1  , c, b] 
             +  xdiff           * ydiff         * arr[yrot_int + 1  , xrot_int + 1  , c, b])
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
        ydiff, ydiff_1minus, xdiff, xdiff_1minus = 
            bilinear_helper(yrot, xrot, yrot_f, xrot_f)
        Atomix.@atomic out[yrot_int     ,   xrot_int    , c, b]  += xdiff_1minus    * ydiff_1minus * o
        Atomix.@atomic out[yrot_int + 1 ,   xrot_int    , c, b]  += xdiff_1minus    * ydiff      * o
        Atomix.@atomic out[yrot_int     ,   xrot_int + 1, c, b]  += xdiff           * ydiff_1minus * o
        Atomix.@atomic out[yrot_int + 1 ,   xrot_int + 1, c, b]  += xdiff           * ydiff      * o
    end
end


# is this rrule good? 
# no @thunk and @unthunk
function ChainRulesCore.rrule(::typeof(imrotate), arr::AbstractArray{T}, θ; 
                              method=:bilinear, midpoint=size(arr) .÷ 2 .+ 1) where T
    res = imrotate(arr, θ; method, midpoint)
    function pb_rotate(dy)
        ad = ∇imrotate(unthunk(dy), arr, θ; method, midpoint)
        return NoTangent(), ad, NoTangent()
    end    

	return res, pb_rotate
end
