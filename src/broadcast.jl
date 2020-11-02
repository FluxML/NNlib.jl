
"""
    activate!!(f, A, [b=0])

This applies `A .= f.(A) .+ b`, provided `A` is mutable.

When used with Zygote, it overwrites `A` only when this is known
not to be needed for the gradient calculation.
"""
activate!!(f, A::AbstractArray) = f.(A)
activate!!(f, A::AbstractArray, b) = f.(A) .+ b

activate!!(f, A::StridedArray) = A .= f.(A)
activate!!(f, A::StridedArray, b) = A .= f.(A) .+ b


# This should be in Zygote
using Zygote
for (f, ∇f) in [(:σ, :∇σ), (:tanh, :∇tanh), (:relu, :∇relu)]
    @eval begin

        Zygote.@adjoint function activate!!(::typeof($f), x::AbstractArray)
            y = activate!!($f, x)
            y, dy -> (nothing, $∇f(y, dy))
        end
        Zygote.@adjoint function activate!!(::typeof($f), x::AbstractArray, b::Bool)
            y = activate!!($f, x, b)
            y, dy -> (nothing, $∇f(y, dy), nothing)
        end
        Zygote.@adjoint function activate!!(::typeof($f), x::AbstractArray, b::AbstractArray)
            y = activate!!($f, x, b)
            y, dy -> (nothing, $∇f(y, dy), sum!(similar(b), dy))
        end

    end
end


# This should only apply when LoopVectorization is loaded
using LoopVectorization
activate!!(f, A::Array{<:LinearAlgebra.BlasReal}) = @avx A .= f.(A)
activate!!(f, A::Array{<:LinearAlgebra.BlasReal}, b) = @avx A .= f.(A) .+ b

∇σ(y::Array{<:LinearAlgebra.BlasReal}, dy::Array{<:LinearAlgebra.BlasReal}) = @avx dy .* conj.(y .* (1 .- y))
∇tanh(y::Array{<:LinearAlgebra.BlasReal}, dy::Array{<:LinearAlgebra.BlasReal}) = @avx dy .* conj.(1 .- y.^2)

using Zygote.FillArrays
∇σ(y::Array{<:LinearAlgebra.BlasReal}, dy::Fill) = (dyval = dy.value; @avx dyval .* (y .* (1 .- y)) )
∇tanh(y::Array{<:LinearAlgebra.BlasReal}, dy::Fill) = (dyval = dy.value; @avx dyval .* (1 .- y.^2) )


# This is how Flux could use this:
function (a::Dense)(x::AbstractArray)
    W, b, σ = a.W, a.b, a.σ
    # NNlib.activate!!(σ, W*x, b)
    NNlib.activate!!(σ, muladd(W, x, b))
end
function (c::Conv)(x::AbstractArray)
    cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
    b = c.bias isa AbstractVector ? reshape(c.bias, map(_->1, c.stride)..., :, 1) : c.bais
    NNlib.activate!!(c.σ, conv(x, c.weight, cdims), b)
end


#= # Not such a big improvement:

julia> @btime gradient(x -> sum(tanh.(W * x)), x)  setup=(x=rand(100,100); W=rand(100,100));
  85.898 μs (18 allocations: 391.48 KiB)

julia> @btime gradient(x -> sum(activate!!(tanh, W * x)), x)  setup=(x=rand(100,100); W=rand(100,100));
  55.881 μs (16 allocations: 313.25 KiB)

julia> 313.25 / 391.48
0.8001685909880454

=#
