export dense_bias_act, bias_act!

"""
    dense_bias_act(σ, w, x)
    dense_bias_act(σ, w, x, b)

This is equivalent to `σ.((w * x) .+ b)`, but should be more efficient.
Calls [`bias_act!`](@ref), but this mutates only the result `w * x`
which is allocated within this function.

See also [`conv_bias_act`](@ref) and [`bias_act!`](@ref).
"""
dense_bias_act(σ, w, x, b) = bias_act!(σ, w * x, b)
dense_bias_act(σ, w, x) = bias_act!(σ, w * x)
# dense_bias_act(σ, w, x, b) = bias_act!(σ, muladd(w, x, b)) # another possibility, slower?

_INPLACE_STR = join(first.(INPLACE_ACTS), ", ")
"""
    bias_act!(σ, x)
    bias_act!(σ, x, b)

This is equivalent to `σ.(x .+ b)`, but faster because it will:
1. overwrite `x` to save memory,
2. re-use memory in the the gradient,
3. replace `sigmoid` & `tanh` with `sigmoid_fast` & `tanh_fast`.

The greatest re-use will be when `x isa StridedArray{<:AbstractFloat}`, 
and the activation is `σ ∈ ($_INPLACE_STR)`.
"""
bias_act!(σ::Function, x::AbstractArray, b) = σ.(x .+ b)
bias_act!(σ::Function, x::AbstractArray) = σ.(x)

for (act, grad) in INPLACE_ACTS

    # Forwards
    # NB: signatures exclude `x::Array{Int}` and `b::Vector{Dual}`, which won't work in-place.
    @eval bias_act!(σ::typeof($act), x::StridedArray{<:AbstractFloat}, b::AbstractVector{<:AbstractFloat}) =
        _bias_act!(σ, x, b)
    @eval bias_act!(σ::typeof($act), x::StridedArray{<:AbstractFloat}, b::Bool) =
        _bias_act!(σ, x, b)
    @eval bias_act!(σ::typeof($act), x::StridedArray{<:AbstractFloat}) =
        _bias_act!(σ, x)

    # Gradient
    pullback = Symbol(:bias_act_, act, :_pullback)

    @eval function rrule(::typeof(bias_act!), σ::typeof($act), x::StridedArray{<:AbstractFloat}, b::B) where {B}
        Ω = bias_act!(σ, x, b)  # this overwrites, now x === Ω
        size_b = size(b)
        function $pullback(Δ)
            dx = @. Δ * $grad  # tempting to overwrite again, but only safe if you call pullback at most once
            if eltype(B) == Bool
                db = NoTangent()
            else
                dims = filter(d -> get(size_b, d, 1)==1, ntuple(identity, ndims(dx)))
                db = reshape(sum(dx; dims = dims), size_b)
            end
            return (NoTangent(), NoTangent(), dx, db)
        end
        return Ω, $pullback
    end
    # The same, without `b`, as I couldn't see a tidy way to combine them.
    @eval function rrule(::typeof(bias_act!), σ::typeof($act), x::StridedArray{<:AbstractFloat})
        Ω = bias_act!(σ, x)
        function $pullback(Δ)
            dx = @. Δ * $grad 
            return (NoTangent(), NoTangent(), dx)
        end
        return Ω, $pullback
    end

end

# Inner function, always in-place. Could be overloaded by vmap! etc. 
_bias_act!(σ, x, b) = x .= σ.(x .+ b)
_bias_act!(σ, x) = x .= σ.(x)
# ... and which simplifies the dispatch for some even more special activation functions:
_bias_act!(::typeof(identity), x, b::Bool) = b ? (x.=x.+b) : x
_bias_act!(::typeof(identity), x) = x
# for (fast, slow) in [
#         (tanh_faster, tanh), 
#         (sigmoid_faster, sigmoid),
#     ]
#     @eval _bias_act!(::typeof($slow), x, b) = x .= $fast.(x .+ b)
#     @eval _bias_act!(::typeof($slow), x) = x .= $fast.(x)
# end
# It would also be easy here to restrict the use of tanh_fast to CPU arrays.

#=
# Fallback, we can still save some memory compared to unfused broadcast.
# Not required though, since all mutating cases take the fast path above.

function rrule(::RuleCofig{??}, ::typeof(bias_act!), σ::Function, x::StridedArray{<:AbstractFloat}, b::B) where {B}
    x .= x .+ b  # overwrites x but still before activation
    Ω, uncast = rrule_via_ad(config, broadcast, σ, x)
    function bias_act_pullback(Δ)
        maybethunk = uncast(Δ)[3]
        if maybethunk isa InplaceableThunk  # then we can overwrite x with its gradient
            dx = maybethunk(fill!(x, false)) # but Zygote doesn't do this right now, oh well.
        else
            dx = unthunk(maybethunk)
        end
        if B <: AbstractArray
            dims = ntuple(d -> d+1, ndims(dx)-1)
            sum(dx; dims = dims)
        else
            db = NoTangent()
        end
        (NoTangent(), NoTangent(), dx, db)
    end
    return Ω, bias_act_pullback
end
=#



#=

# Some benchmarks, Julia 1.7 + M1 rosetta.
# This isn't really faster, but it does use half as much memory,
# and provides a non-piratical place to hook on faster `tanh`, `vmap`, etc.

julia> using NNlib, BenchmarkTools

julia> w, b = rand(100, 100), rand(100);

julia> @btime bias_act!(relu, $w, $b);
  4.083 μs (2 allocations: 64 bytes)

julia> @btime relu.($w .+ $b);  # faster, that's odd?
  2.990 μs (2 allocations: 78.17 KiB)

julia> @btime bias_act!(tanh, $w, $b);
  98.166 μs (2 allocations: 64 bytes)

julia> @btime tanh.($w .+ $b);
  100.000 μs (2 allocations: 78.17 KiB)

julia> @btime bias_act!(tanh_fast, $w, $b);  # 
  33.084 μs (2 allocations: 64 bytes)

julia> @btime bias_act!(tanh_faster, $w, $b);
  24.416 μs (2 allocations: 64 bytes)

# gradients:

julia> using Zygote

julia> @btime gradient((w,b) -> sum(bias_act!(relu, w, b)), $w, $b);
  19.292 μs (15 allocations: 79.39 KiB)

julia> @btime gradient((w,b) -> sum(relu.(w .+ b)), $w, $b);
  19.792 μs (29 allocations: 236.19 KiB)

julia> @btime gradient((w,b) -> sum(bias_act!(tanh, w, b)), $w, $b);
  109.916 μs (15 allocations: 79.39 KiB)

julia> @btime gradient((w,b) -> sum(tanh.(w .+ b)), $w, $b);
  111.708 μs (14 allocations: 235.62 KiB)


# with matmul too:

julia> w, b = rand(100, 100), rand(100); x = rand(size(w)...);

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(relu, w, x, b)), $w, $x, $b);
  132.000 μs (38 allocations: 314.59 KiB)

julia> @btime gradient((w,x,b) -> sum(relu.((w * x) .+ b)), $w, $x, $b);
  128.875 μs (20 allocations: 470.14 KiB)

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(tanh, w, x, b)), $w, $x, $b);
  162.166 μs (38 allocations: 314.59 KiB)

julia> @btime gradient((w,x,b) -> sum(tanh.((w * x) .+ b)), $w, $x, $b);
  160.916 μs (20 allocations: 470.14 KiB)


# memory

julia> 314.59 / 470.14
0.6691411069043264

julia> @btime copy($w);
  2.635 μs (2 allocations: 78.17 KiB)

julia> 314.59 / 78.17
4.02443392605859

=#
