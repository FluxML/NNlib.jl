
### v0.8 Deprecations
@deprecate ∇softmax(Δ, x; dims=1)   ∇softmax(Δ, x, softmax(x, dims=dims); dims=dims)
@deprecate ∇softmax!(Δ, x; dims=1)   ∇softmax!(Δ, Δ, x, softmax(x, dims=dims); dims=dims)
@deprecate ∇softmax!(out, Δ, x; dims=1)   ∇softmax!(out, Δ, x, softmax(x, dims=dims); dims=dims)

@deprecate ∇logsoftmax(Δ, x; dims=1)   ∇logsoftmax(Δ, x, logsoftmax(x, dims=dims); dims=dims)
@deprecate ∇logsoftmax!(Δ, x; dims=1)   ∇logsoftmax!(Δ, Δ, x, logsoftmax(x, dims=dims); dims=dims)
@deprecate ∇logsoftmax!(out, Δ, x; dims=1)   ∇logsoftmax!(out, Δ, x, logsoftmax(x, dims=dims); dims=dims)
