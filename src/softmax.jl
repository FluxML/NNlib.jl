
"""
    softmax(x; dims = 1)

[Softmax](https://en.wikipedia.org/wiki/Softmax_function) turns input array `x`
into probability distributions that sum to 1 along the dimensions specified by `dims`.
It is semantically equivalent to the following:

    softmax(x; dims = 1) = exp.(x) ./ sum(exp.(x), dims = dims)

with additional manipulations enhancing numerical stability.

For a matrix input `x` it will by default (`dims = 1`) treat it as a batch of vectors,
with each column independent. Keyword `dims = 2` will instead treat rows independently, and so on.

See also [`logsoftmax`](@ref).

# Examples

```jldoctest; filter = r"[+-]?([0-9]*[.])?[0-9]+(f[+-]*[0-9])?"
julia> softmax([1, 2, 3])
3-element Vector{Float64}:
 0.09003057317038046
 0.24472847105479764
 0.6652409557748218

julia> softmax([1 2 3; 2 2 2])  # dims=1
2×3 Matrix{Float64}:
 0.268941  0.5  0.731059
 0.731059  0.5  0.268941

julia> softmax([1 2 3; 2 2 2]; dims=2)
2×3 Matrix{Float64}:
 0.0900306  0.244728  0.665241
 0.333333   0.333333  0.333333
```

Note that, when used with Flux.jl, `softmax` must not be passed to layers like `Dense`
which accept an activation function. The activation is broadcasted over the result,
thus applies to individual numbers. But `softmax` always needs to see the whole column.

```julia-repl
julia> using Flux

julia> x = randn(Float32, 4, 4, 3, 13);

julia> model = Chain(Conv((4, 4), 3 => 8, tanh), Flux.flatten, Dense(8 => 7), softmax);

julia> model(x) |> size
(7, 13)

julia> Dense(4 => 7, softmax)(x)
ERROR: `softmax(x)` called with a number, but it expects an array. 
```
"""
softmax(x::AbstractArray{T}; dims = 1) where {T} = softmax!(similar(x, float(T)), x; dims)

softmax!(x::AbstractArray; dims = 1) = softmax!(x, x; dims)

function softmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
    max_ = fast_maximum(x; dims)
    if all(isfinite, max_)
        @fastmath out .= exp.(x .- max_)
    else
        _zero, _one, _inf = T(0), T(1), T(Inf)
        @fastmath @. out = ifelse(isequal(max_,_inf), ifelse(isequal(x,_inf), _one, _zero), exp(x - max_))
    end
    tmp = dims isa Colon ? sum(out) : sum!(max_, out)
    out ./= tmp
end

function ∇softmax_data(dy::AbstractArray{T}, y::AbstractArray{S}; dims = 1) where {T,S}
    dx = if within_gradient(y)
        tmp = dy .* y
        tmp .- y .* sum(tmp; dims)
    else
        # This path is faster, only safe for 1st derivatives though.
        # Was previously `∇softmax!(dx, dy, x, y; dims)` to allow CUDA overloads,
        # but that was slow: https://github.com/FluxML/NNlibCUDA.jl/issues/30
        out = similar(y, promote_type(T,S))  # sure to be mutable
        out .= dy .* y
        out .= out .- y .* sum(out; dims)
    end
end

function rrule(::typeof(softmax), x; dims = 1)
    y = softmax(x; dims)
    softmax_pullback(dy) = (NoTangent(), ∇softmax_data(unthunk(dy), y; dims))
    return y, softmax_pullback
end

fast_maximum(x::AbstractArray{T}; dims) where {T} = @fastmath reduce(max, x; dims, init = float(T)(-Inf))

"""
    logsoftmax(x; dims = 1)

Computes the log of softmax in a more numerically stable
way than directly taking `log.(softmax(xs))`. Commonly used in
computing cross entropy loss.

It is semantically equivalent to the following:

    logsoftmax(x; dims = 1) = x .- log.(sum(exp.(x), dims = dims))

See also [`softmax`](@ref).
"""
logsoftmax(x::AbstractArray{T}; dims = 1) where {T} = logsoftmax!(similar(x, float(T)), x; dims)

logsoftmax!(x::AbstractArray; dims = 1) = logsoftmax!(x, x; dims)

function logsoftmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
    max_ = fast_maximum(x; dims)
    if all(isfinite, max_)
        out .= x .- max_
    else
        _zero, _minf, _inf = T(0), T(-Inf), T(Inf)
        @. out = ifelse(isequal(max_,_inf), ifelse(isequal(x,_inf), _zero, _minf), x - max_)
    end
    @fastmath log_ = log.(sum(exp, out; dims))
    out .-= log_
end

function ∇logsoftmax_data(dy::AbstractArray, y::AbstractArray; dims = 1)
    # This was previously `∇logsoftmax!(dx, dy, x, y; dims)` to allow CUDA overloads, but that was slow.
    dx = dy .- sum(dy; dims) .* exp.(y)
end
    
function rrule(::typeof(logsoftmax), x; dims = 1)
    y = logsoftmax(x; dims)
    logsoftmax_pullback(dy) = (NoTangent(), ∇logsoftmax_data(unthunk(dy), y; dims))
    return y, logsoftmax_pullback
end

"""
    logsumexp(x; dims = :)

Computes `log.(sum(exp.(x); dims))` in a numerically stable way.
Without `dims` keyword this returns a scalar.

See also [`logsoftmax`](@ref).
"""
function logsumexp(x::AbstractArray; dims = :)
    max_ = fast_maximum(x; dims)
    @fastmath max_ .+ log.(sum(exp.(x .- max_); dims))
end

function rrule(::typeof(logsumexp), x; dims = :)
    # The gradient is `softmax`, but both compute `tmp` so it's worth saving.
    max_ = fast_maximum(x; dims)
    @fastmath tmp = exp.(x .- max_)
    @fastmath y = max_ .+ log.(sum(tmp; dims))
    logsumexp_pullback(dy) = (NoTangent(), unthunk(dy) .* tmp ./ sum(tmp; dims))
    return y, logsumexp_pullback
end

# Informative error message if any of the softmax variants is called with a number
for f in (:softmax, :logsoftmax, :softmax!, :logsoftmax!, :logsumexp)
    @eval $(f)(x::Number, args...) = 
      error("`", $(string(f)), "(x)` called with a number, but it expects an array. Usually this is because a layer like `Dense(3,4,softmax)` is broadcasting it like an activation function; `softmax` needs to be outside the layer.")
end
