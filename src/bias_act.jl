export dense_bias_act, bias_act!

"""
    dense_bias_act(σ, w, x, b)

This is equivalent to `σ.((w * x) .+ b)`, but should be more efficient.
Calls [`bias_act!`](@ref), but this mutates only the result `w * x`
which is allocated within this function.

See also [`conv_bias_act`](@ref) and [`bias_act!`](@ref).
"""
dense_bias_act(σ, w, x, b) = bias_act!(σ, w * x, b)
# dense_bias_act(σ, w, x, b) = bias_act!(σ, muladd(w, x, b)) # another possibility

_INPLACE_STR = join(first.(INPLACE_ACTS), ", ")
"""
    bias_act!(σ, x)
    bias_act!(σ, x, b)

This is equivalent to `σ.(x .+ b)`, but faster because it will:
1. overwrite `x` to save memory,
2. re-use memory in the the gradient,
3. replace `tanh` with `tanh_fast`. ?? maybe!

The greatest re-use will be when `x isa StridedArray{<:AbstractFloat}`, 
and the activation is `σ ∈ ($_INPLACE_STR)`.
"""
bias_act!(σ::Function, x::AbstractArray, b) = σ.(x .+ b)
bias_act!(σ::Function, x::AbstractArray) = σ.(x)

# bias_act!(σ::typeof(tanh), x::AbstractArray, b=false) = bias_act!(tanh_fast, x, b)

for (act, grad) in INPLACE_ACTS  # Best case!

    # Forward pass
    # if act === :tanh
    #     # The above method (σ::typeof(tanh), x::AbstractArray, ...) isn't specific enough
    #     @eval function bias_act!(σ::typeof($act), x::StridedArray{<:AbstractFloat}, b=false)
    #         x .= tanh_fast.(x .+ b)
    #     end
    # else
        @eval function bias_act!(σ::typeof($act), x::StridedArray{<:AbstractFloat}, b=false) # where {F<:typeof($act)}
            x .= σ.(x .+ b)
        end
    # end

    # Gradient
    pullback = Symbol(:bias_act_, act, :_pullback)
    # @eval function rrule(::typeof(bias_act!), σ::typeof($act), x::StridedArray{<:AbstractFloat}, b::B=nothing) where {B}
    #     if B == Nothing
    #         Ω = bias_act!(σ, x)
    #     else
    #         Ω = bias_act!(σ, x, b)  # this overwrites, now x === Ω
    #     end
    #     function $pullback(Δ)
    #         dx = @. Ω = Δ * $grad  # overwrites again, now x === dx
    #         if B <: AbstractArray{<:AbstractFloat}
    #             dims = ntuple(d -> d+1, ndims(dx)-1)
    #             db_splat = (sum(dx; dims = dims),)
    #         elseif B === Nothing  # a bit of a hack, 2-argument form
    #             db_splat = ()
    #         else
    #             db_splat = (NoTangent(),)
    #         end
    #         return (NoTangent(), NoTangent(), dx, db_splat...)
    #     end
    #     return Ω, $pullback
    # end

    @eval function rrule(::typeof(bias_act!), σ::typeof($act), x::StridedArray{<:AbstractFloat}, b::B) where {B}
        Ω = bias_act!(σ, x, b)  # this overwrites, now x === Ω
        function $pullback(Δ)
            dx = @. Ω = Δ * $grad  # overwrites again, now x === dx
            if B <: AbstractArray{<:AbstractFloat}
                dims = ntuple(d -> d+1, ndims(dx)-1)
                db = sum(dx; dims = dims)
            else
                db = (NoTangent(),)
            end
            return (NoTangent(), NoTangent(), dx, db)
        end
        return Ω, $pullback
    end
    # The same, without `b`, as I couldn't see a tidy way to combine them.
    @eval function rrule(::typeof(bias_act!), σ::typeof($act), x::StridedArray{<:AbstractFloat})
        Ω = bias_act!(σ, x)
        function $pullback(Δ)
            dx = @. Ω = Δ * $grad 
            return (NoTangent(), NoTangent(), dx)
        end
        return Ω, $pullback
    end


end

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

# some benchmarks, Julia 1.7 + M1 rosetta

julia> w, b = rand(100, 100), rand(100);

julia> @btime bias_act!(relu, $w, $b);
  4.131 μs (0 allocations: 0 bytes)
  3.891 μs (0 allocations: 0 bytes) # with simpler relu

julia> @btime relu.($w .+ $b);  # faster, that's odd?
  3.302 μs (2 allocations: 78.17 KiB)
  3.271 μs (2 allocations: 78.17 KiB) # with simpler relu

julia> @btime bias_act!(tanh, $w, $b);
  64.291 μs (0 allocations: 0 bytes)

julia> @btime tanh.($w .+ $b);
  113.666 μs (2 allocations: 78.17 KiB)

julia> @btime NNlib.tanh_fast.($w .+ $b);
  62.083 μs (2 allocations: 78.17 KiB)

# gradients:

julia> @btime gradient((w,b) -> sum(bias_act!(relu, w, b)), $w, $b);
  15.709 μs (5 allocations: 1.02 KiB)
  23.208 μs (40 allocations: 1.98 KiB)  # with muladd

julia> @btime gradient((w,b) -> sum(relu.(w .+ b)), $w, $b);
  19.166 μs (29 allocations: 236.19 KiB)

julia> @btime gradient((w,b) -> sum(bias_act!(tanh, w, b)), $w, $b);
  112.791 μs (46 allocations: 2.22 KiB)
  69.166 μs (5 allocations: 1.02 KiB)  # with tanh_fast

julia> @btime gradient((w,b) -> sum(tanh.(w .+ b)), $w, $b);
  110.541 μs (14 allocations: 235.62 KiB)

julia> @btime gradient((w,b) -> sum(NNlib.tanh_fast.(w .+ b)), $w, $b);
  76.166 μs (29 allocations: 236.19 KiB)

# with matmul too:

julia> x = rand(size(w)...);

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(relu, w, x, b)), $w, $x, $b);
  126.583 μs (46 allocations: 236.73 KiB)
  124.833 μs (52 allocations: 236.88 KiB)  # with muladd

julia> @btime gradient((w,x,b) -> sum(relu.((w * x) .+ b)), $w, $x, $b);
  118.083 μs (20 allocations: 470.14 KiB)

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(tanh, w, x, b)), $w, $x, $b);
  156.458 μs (52 allocations: 236.97 KiB)
  180.708 μs (52 allocations: 236.97 KiB)  # with tanh_fast

julia> @btime gradient((w,x,b) -> sum(tanh.((w * x) .+ b)), $w, $x, $b);
  157.958 μs (20 allocations: 470.14 KiB)

julia> @btime gradient((w,x,b) -> sum(NNlib.tanh_fast.((w * x) .+ b)), $w, $x, $b);
  184.709 μs (20 allocations: 470.14 KiB)

# memory

julia> 236.97 / 470.14
0.5040413493852895

julia> @btime copy($w);
  2.635 μs (2 allocations: 78.17 KiB)

=#

#=
# This doesn't work, macros need to be inside another file, 
# you can paste it into the REPL, doesn't seem to help though?

@init @require LoopVectorization = "bdcacae8-1622-11e9-2a5c-532679323890" begin
    # using .LoopVectorization: @turbo
    function NNlib.bias_act!(σ::typeof(tanh), x::Array{<:Union{Float32, Float64}}, b=false)
        @turbo x .= tanh.(x .+ b)
    end
end

# Ideally this would apply to the `gradient dx = @. Ω = Δ * $grad` too,
# that would need to be pulled out as a function to overload easily.
=#