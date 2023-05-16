### Deprecated while v0.8 was latest

export ∇softmax,
    ∇softmax!,
    logsoftmax,
    logsoftmax!,
    ∇logsoftmax,
    ∇logsoftmax!

function ∇softmax!(out::AbstractArray, Δ::AbstractArray, 
                    x::AbstractArray, y::AbstractArray; dims = 1)
    Base.depwarn("`∇softmax!(dx, dy, x, y)` is deprecated, just use `∇softmax_data(dy, y)`", :∇softmax!)
    # Removed because using a mutating function blocks 2nd derivatives, and
    # the CUDA overload was slow anyway, https://github.com/FluxML/NNlibCUDA.jl/issues/30
    out .= Δ .* y
    out .= out .- y .* sum(out; dims)
end

function ∇logsoftmax!(out::AbstractArray, Δ::AbstractArray,
                    x::AbstractArray, y::AbstractArray; dims = 1) 
    Base.depwarn("`∇logsoftmax!(dx, dy, x, y)` is deprecated, just use `∇logsoftmax_data(dy, y)`", :∇softmax!)
    out .= Δ .- sum(Δ; dims) .* exp.(y)
end

function ∇softmax(dy::AbstractArray{T}, x::AbstractArray, y::AbstractArray{S}; dims = 1) where {T,S}
    # Removed because there's no need to close over `x` here, that was done only to distinguish
    # this from `∇softmax(Δ, x; dims = 1)` which re-computed `y = softmax(x)`, which is slow.
    Base.depwarn("`∇softmax(dy, x, y)` should be replaced with `∇softmax_data(dy, y)`", :∇softmax)
    ∇softmax_data(dy, y)
end

function ∇logsoftmax(dy::AbstractArray, x::AbstractArray, y::AbstractArray; dims = 1)
    Base.depwarn("`∇logsoftmax(dy, x, y)` should be replaced with `∇logsoftmax_data(dy, y)`", :∇softmax)
    ∇logsoftmax_data(dy, y)
end

