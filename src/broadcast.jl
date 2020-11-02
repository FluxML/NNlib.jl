
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

# This should only apply when LoopVectorization is loaded
activate!!(f, A::Array{<:LinearAlgebra.BlasReal}) = @avx A .= f.(A)
activate!!(f, A::Array{<:LinearAlgebra.BlasReal}, b) = @avx A .= f.(A) .+ b

# This should be in Zygote
for (f, ∇f) in [(:σ, :∇σ), (:tanh, :∇tanh), (:relu, :∇relu)]
    @eval begin

        Zygote.@adjoint activate!!(::typeof($f), x::AbstractArray)
            activate!!($f, x), dy -> (nothing, $∇f(y, dy))

        Zygote.@adjoint activate!!(::typeof($f), x::AbstractArray, b::Bool)
            activate!!($f, x), dy -> (nothing, $∇f(y, dy), nothing)

        Zygote.@adjoint activate!!(::typeof($f), x::AbstractArray, b::AbstractArray)
            activate!!($f, x), dy -> (nothing, $∇f(y, dy), sum!(similar(b), dy))

    end
end

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
