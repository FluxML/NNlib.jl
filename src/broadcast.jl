
"""
    map!!(f, A)

This applies `A .= f.(A)`, provided `A` is mutable.

When used with Zygote, it overwrites `A` only when this is known
not to be needed for the gradient calculation.
"""
map!!(f, A::AbstractArray) = f.(A)
map!!(f, A::StridedArray) = A .= f.(A)

"""
    add_map!!(f, A, b)

This applies `A .= f.(A .+ b)`, provided `A` is mutable.

Perhaps this and `map!!` should be methods of the same function?
"""
add_map!!(f, A::AbstractArray, b) = f.(A) .+ b
add_map!!(f, A::StridedArray, b) = A .= f.(A) .+ b


@inline function tanh_fast(x) # less accurate but faster
   exp2x = exp(x + x)
   (exp2x - one(x)) / (exp2x + one(x))
end
# and map!! is an easy place to hook this up without broadcast piracy:
map!!(f::typeof(tanh), A::StridedArray) = A .= tanh_fast.(A)
add_map!!(f::typeof(tanh), A::StridedArray, b) = A .= tanh_fast.(A) .+ b

for (f, ∇f) in [(:σ, :∇σ), (:tanh, :∇tanh), (:relu, :∇relu)]
    @eval begin

        function ChainRulesCore.rrule(::typeof(map!!), ::typeof($f), x::AbstractArray)
            y = map!!($f, x)
            map_back(dy) = (NO_FIELDS, NO_FIELDS, $∇f.(y, dy))
            return y, map_back
        end

        function ChainRulesCore.rrule(::typeof(add_map!!), ::typeof($f), x::AbstractArray, b::Bool)
            y = add_map!!($f, x, b)
            add_map_back(dy) = (NO_FIELDS, NO_FIELDS, $∇f.(y, dy), Zero())
            return y, add_map_back
        end
        function ChainRulesCore.rrule(::typeof(add_map!!), ::typeof($f), x::AbstractArray, b::AbstractArray)
            y = add_map!!($f, x, b)
            add_map_back(dy) = (NO_FIELDS, NO_FIELDS, $∇f.(y, dy), sum!(similar(b), dy))
            return y, add_map_back
        end

    end
end

#=

# This should only apply when LoopVectorization is loaded

using LoopVectorization
map!!(f, A::Array{<:LinearAlgebra.BlasReal}) = @avx A .= f.(A)
add_map!!(f, A::Array{<:LinearAlgebra.BlasReal}, b) = @avx A .= f.(A) .+ b

∇σ(y::Array{<:LinearAlgebra.BlasReal}, dy::Array{<:LinearAlgebra.BlasReal}) = @avx dy .* conj.(y .* (1 .- y))
∇tanh(y::Array{<:LinearAlgebra.BlasReal}, dy::Array{<:LinearAlgebra.BlasReal}) = @avx dy .* conj.(1 .- y.^2)
∇relu(y::Array{<:LinearAlgebra.BlasReal}, dy::Array{<:LinearAlgebra.BlasReal}) = @avx ifelse.(y .> 0, one(y), zero(y))

using Zygote.FillArrays
∇σ(y::Array{<:LinearAlgebra.BlasReal}, dy::Fill) = (dyval = dy.value; @avx dyval .* (y .* (1 .- y)) )
∇tanh(y::Array{<:LinearAlgebra.BlasReal}, dy::Fill) = (dyval = dy.value; @avx dyval .* (1 .- y.^2) )



# This is how Flux could use this:

function (a::Dense)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    # NNlib.add_map!!(σ, W*x, b)
    NNlib.map!!(σ, muladd(W, x, b))
end

function (c::Conv)(x::AbstractArray)
    cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
    conv_bias_activate(c, conv(x, c.weight, cdims))
end

function conv_bias_activate(c::Conv, x)
    if c.bias isa AbstractVector
        b3 = reshape(c.bias, map(_->1, c.stride)..., :, 1)
        NNlib.add_map!!(c.σ, x, b3)
    else
        NNlib.map!!(c.σ, x)
    end
end

=#

#=
# Without LV -- memory savings but no huge speedups

julia> using NNlib, Zygote; import NNlib: map!!, add_map!!

julia> x100, W100, b100 = randn(100,100), randn(100,100), randn(100);

julia> @btime gradient(x -> sum(relu.(x)), $x100);
  22.800 μs (4 allocations: 156.41 KiB)

julia> @btime gradient(x -> sum(map!!(relu, x)), $x100); # half the memory
  20.305 μs (4 allocations: 78.23 KiB)

julia> @btime tanh.($x100);
  124.317 μs (2 allocations: 78.20 KiB)

julia> @btime tanh_fast.($x100); # not bad!
  69.273 μs (2 allocations: 78.20 KiB)

julia> @btime gradient(x -> sum(tanh.(x)), $x100);
  147.071 μs (5 allocations: 156.42 KiB)

julia> @btime gradient(x -> sum(map!!(tanh, x)), $x100); # without tanh_fase
  137.726 μs (4 allocations: 78.23 KiB)

julia> @btime gradient(x -> sum(map!!(tanh, x)), $x100); # with tanh_fast
  83.147 μs (2 allocations: 78.20 KiB)


julia> @btime relu.($W100 * $x100 .+ $b100); # present behaviour
  52.530 μs (4 allocations: 156.41 KiB)

julia> @btime add_map!!(relu, $W100 * $x100, $b100); # fuse outer
  52.937 μs (6 allocations: 78.30 KiB)

julia> @btime map!!(relu, muladd($W100, $x100, $b100)); # fuse inner -- needs ChainRules PR for gradient
  53.230 μs (4 allocations: 78.23 KiB)


julia> @btime gradient((W,x,b) -> sum(relu.(W*x .+ b)), $W100, $x100, $b100);
  160.351 μs (20 allocations: 470.34 KiB)

julia> @btime gradient((W,x,b) -> sum(add_map!!(relu, W*x, b)), $W100, $x100, $b100);
  152.735 μs (30 allocations: 314.47 KiB)

julia> @btime gradient((W,x,b) -> sum(map!!(relu, muladd(W,x,b))), $W100, $x100, $b100);
  149.161 μs (12 allocations: 313.75 KiB)



julia> using LoopVectorization, LinearAlgebra

julia> import NNlib: map!!, add_map!!, ∇σ, ∇tanh, ∇relu

# then load some definitions above!

julia> @btime gradient(x -> sum(tanh.(W * x)), x)  setup=(x=rand(100,100); W=rand(100,100));
  175.390 μs (16 allocations: 391.44 KiB)

julia> @btime gradient(x -> sum(map!!(tanh, W * x)), x)  setup=(x=rand(100,100); W=rand(100,100));
  205.760 μs (29 allocations: 313.98 KiB) --- WTF?

julia> @btime gradient(x -> sum(tanh.(x)), x)  setup=(x=rand(100,100));
  147.240 μs (5 allocations: 156.42 KiB)

julia> @btime gradient(x -> sum(map!!(tanh, x)), x)  setup=(x=rand(100,100));
  78.316 μs (2 allocations: 78.20 KiB)

julia> @btime gradient(x -> sum(relu.(x)), x)  setup=(x=rand(100,100));
  22.990 μs (4 allocations: 156.41 KiB)

julia> @btime gradient(x -> sum(map!!(relu, x)), x)  setup=(x=rand(100,100));
  20.579 μs (4 allocations: 78.23 KiB)



=#
