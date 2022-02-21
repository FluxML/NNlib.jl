### Deprecated while v0.7 was latest

# Old 2-arg version recomputing forward
function ∇softmax(Δ, x; dims = 1)
    Base.depwarn("`∇softmax(Δ, x)` without `y = softmax(x)` argument is deprecatcated, as this is inefficient", :∇softmax)
    ∇softmax(Δ, x, softmax(x; dims); dims)
end
∇softmax!(Δ, x; dims = 1) = Δ .= ∇softmax(Δ, x; dims)
∇softmax!(out, Δ, x; dims = 1) = out .= ∇softmax(Δ, x; dims)

# Old 2-arg version recomputing forward
function ∇logsoftmax(Δ, x; dims = 1)
    Base.depwarn("`∇logsoftmax(Δ, x)` without `y = logsoftmax(x)` argument is deprecatcated", :∇logsoftmax)
    ∇logsoftmax(Δ, x, logsoftmax(x; dims); dims)
end
∇logsoftmax!(Δ, x; dims = 1) = Δ .= ∇logsoftmax(Δ, x; dims)
∇logsoftmax!(out, Δ, x; dims = 1) = out .= ∇logsoftmax(Δ, x; dims)


### Deprecated while v0.8 was latest

function ∇softmax!(out::AbstractArray, Δ::AbstractArray, 
                    x::AbstractArray, y::AbstractArray; dims = 1)
    Base.depwarn("`∇softmax!` is deprecatcated", :∇softmax!)
    out .= Δ .* y
    out .= out .- y .* sum(out; dims)
end

function ∇logsoftmax!(out::AbstractArray, Δ::AbstractArray,
                    x::AbstractArray, y::AbstractArray; dims = 1) 
    Base.depwarn("`∇softmax!` is deprecatcated", :∇softmax!)
    out .= Δ .- sum(Δ; dims) .* exp.(y)
end