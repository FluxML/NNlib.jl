
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
"""
add_map!!(f, A::AbstractArray, b) = f.(A) .+ b
add_map!!(f, A::StridedArray, b) = A .= f.(A) .+ b


# This should be in Zygote
using Zygote
for (f, ∇f) in [(:σ, :∇σ), (:tanh, :∇tanh), (:relu, :∇relu)]
    @eval begin

        Zygote.@adjoint function map!!(::typeof($f), x::AbstractArray)
            y = map!!($f, x)
            y, dy -> (nothing, $∇f(y, dy))
        end
        Zygote.@adjoint function add_map!!(::typeof($f), x::AbstractArray, b::Bool)
            y = add_map!!($f, x, b)
            y, dy -> (nothing, $∇f(y, dy), nothing)
        end
        Zygote.@adjoint function add_map!!(::typeof($f), x::AbstractArray, b::AbstractArray)
            y = add_map!!($f, x, b)
            y, dy -> (nothing, $∇f(y, dy), sum!(similar(b), dy))
        end

    end
end


# This should only apply when LoopVectorization is loaded
using LoopVectorization
map!!(f, A::Array{<:LinearAlgebra.BlasReal}) = @avx A .= f.(A)
add_map!!(f, A::Array{<:LinearAlgebra.BlasReal}, b) = @avx A .= f.(A) .+ b

∇σ(y::Array{<:LinearAlgebra.BlasReal}, dy::Array{<:LinearAlgebra.BlasReal}) = @avx dy .* conj.(y .* (1 .- y))
∇tanh(y::Array{<:LinearAlgebra.BlasReal}, dy::Array{<:LinearAlgebra.BlasReal}) = @avx dy .* conj.(1 .- y.^2)

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
        NNlib.add_map!!(c.σ, x)
    end
end


#=

julia> @btime gradient(x -> sum(tanh.(W * x)), x)  setup=(x=rand(100,100); W=rand(100,100));
  150.178 μs (18 allocations: 391.48 KiB)

julia> @btime gradient(x -> sum(map!!(tanh, W * x)), x)  setup=(x=rand(100,100); W=rand(100,100));
  110.091 μs (16 allocations: 313.25 KiB)


julia> @btime gradient(x -> sum(tanh.(x)), x)  setup=(x=rand(100,100));
  230.622 μs (19 allocations: 156.80 KiB)

julia> @btime gradient(x -> sum(map!!(tanh, x)), x)  setup=(x=rand(100,100));
  33.935 μs (4 allocations: 78.22 KiB)


julia> @btime gradient(x -> sum(relu.(x)), x)  setup=(x=rand(100,100));
  13.050 μs (19 allocations: 156.80 KiB)

julia> @btime gradient(x -> sum(map!!(relu, x)), x)  setup=(x=rand(100,100));
  7.356 μs (6 allocations: 5.63 KiB)


# Without LV


julia> @btime gradient(x -> sum(tanh.(x)), x)  setup=(x=rand(100,100));
  231.557 μs (19 allocations: 156.80 KiB)

julia> @btime gradient(x -> sum(map!!(tanh, x)), x)  setup=(x=rand(100,100));
  224.781 μs (4 allocations: 78.23 KiB)


julia> @btime gradient(x -> sum(relu.(x)), x)  setup=(x=rand(100,100));
  12.912 μs (19 allocations: 156.80 KiB)

julia> @btime gradient(x -> sum(map!!(relu, x)), x)  setup=(x=rand(100,100));
  11.653 μs (8 allocations: 5.66 KiB)


=#
