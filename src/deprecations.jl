
### Deprecated while v0.7 was latest

function ∇softmax(Δ, x; dims = 1)
    # This 2-arg version recomputes the forward pass, which is slow.
    # Removed from use in 0.7, but only prints a warning during 0.8:
    Base.depwarn("`∇softmax(Δ, x)` without `y = softmax(x)` argument is deprecated, as this is inefficient, please use `∇softmax_data(dy, y)`", :∇softmax)
    ∇softmax(Δ, x, softmax(x; dims); dims)
end
∇softmax!(Δ, x; dims = 1) = Δ .= ∇softmax(Δ, x; dims)
∇softmax!(out, Δ, x; dims = 1) = out .= ∇softmax(Δ, x; dims)

function ∇logsoftmax(Δ, x; dims = 1)
    Base.depwarn("`∇logsoftmax(Δ, x)` without `y = logsoftmax(x)` argument is deprecated, please use `∇logsoftmax_data(dy, y)`", :∇logsoftmax)
    ∇logsoftmax(Δ, x, logsoftmax(x; dims); dims)
end
∇logsoftmax!(Δ, x; dims = 1) = Δ .= ∇logsoftmax(Δ, x; dims)
∇logsoftmax!(out, Δ, x; dims = 1) = out .= ∇logsoftmax(Δ, x; dims)


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

