### Deprecated in v0.9.37

# The softmax gradient helpers lost their `_data` suffix: the `x`-free
# `∇softmax(dy, y)` / `∇logsoftmax(dy, y)` are now the public spelling.
function ∇softmax_data(dy::AbstractArray, y::AbstractArray; dims = 1)
    Base.depwarn("`∇softmax_data(dy, y)` is deprecated, use `∇softmax(dy, y)` instead.", :∇softmax_data)
    return ∇softmax(dy, y; dims)
end

function ∇logsoftmax_data(dy::AbstractArray, y::AbstractArray; dims = 1)
    Base.depwarn("`∇logsoftmax_data(dy, y)` is deprecated, use `∇logsoftmax(dy, y)` instead.", :∇logsoftmax_data)
    return ∇logsoftmax(dy, y; dims)
end
