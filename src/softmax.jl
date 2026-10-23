
"""
    softmax(x; dims = 1)

[Softmax](https://en.wikipedia.org/wiki/Softmax_function) turns input array `x`
into probability distributions that sum to 1 along the dimensions specified by `dims`.
It is semantically equivalent to the following:

    softmax(x; dims = 1) = exp.(x) ./ sum(exp.(x), dims = dims)

with additional manipulations enhancing numerical stability.
For `Float16` and `BFloat16` inputs, the sums are computed in `Float32`.

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

"""
    softmax!(out, x; dims = 1)
    softmax!(x; dims = 1)

In-place version of [`softmax`](@ref), writing the result into `out` (or `x` itself).
"""
softmax!(x::AbstractArray; dims = 1) = softmax!(x, x; dims)

softmax!(out::AbstractArray, x::AbstractArray; dims = 1) = _softmax!(out, x; dims)

# Float16 and BFloat16 are summed in Float32: in their own type, the sums lose most of
# their digits, and Float16 overflows past 65504. `fast_maximum` returns Float32 for them,
# so `x .- max_` is computed in Float32 too. Results keep the input eltype.
_acc(x) = x
_acc(x::HalfFloat) = Float32(x)

# Generic implementation, can be overidden by CUDA and other backends.
function _softmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
    max_ = fast_maximum(x; dims)
    if all(isfinite, max_)
        out .= _exp.(x .- max_)
    else
        _zero, _one, _inf = _acc(T(0)), _acc(T(1)), _acc(T(Inf))
        @. out = ifelse(isequal(max_,_inf), ifelse(isequal(x,_inf), _one, _zero), _exp(x - max_))
    end
    out ./= sum(_acc, out; dims)
end

"""
    ∇softmax(dy, y; dims = 1)

Gradient of [`softmax`](@ref) with respect to its input, given the output `y = softmax(x; dims)`
and the upstream cotangent `dy`. Does not depend on `x`.
"""
function ∇softmax(dy::AbstractArray{T}, y::AbstractArray{S}; dims = 1) where {T,S}
    dx = similar(y, promote_type(T,S))  # sure to be mutable
    ∇softmax!(dx, dy, y; dims)
    return dx
end

"""
    ∇softmax!(dx, dy, y; dims = 1)

In-place version of [`∇softmax`](@ref), writing the gradient into `dx`.
"""
∇softmax!(dx::AbstractArray, dy::AbstractArray, y::AbstractArray; dims = 1) = _∇softmax!(dx, dy, y; dims)

function _∇softmax!(dx::AbstractArray, dy::AbstractArray{T}, y::AbstractArray{S}; dims = 1) where {T,S}
    dx .= dy .* y
    dx .= dx .- y .* sum(_acc, dx; dims)
end

function rrule(::typeof(softmax), x; dims = 1)
    y = softmax(x; dims)
    softmax_pullback(dy) = (NoTangent(), ∇softmax(unthunk(dy), y; dims))
    return y, softmax_pullback
end

# `∇softmax` mutates, and may call a backend kernel, so nested reverse-mode AD
# differentiates it with this rule. It is linear and self-adjoint in `dy`.
# Half precision is computed in Float32 (see `_acc`) and rounded back to `T`.
function rrule(::typeof(∇softmax), dy, y; dims = 1)
    dx = ∇softmax(dy, y; dims)
    function ∇softmax_pullback(Δ)
        Δ = unthunk(Δ)
        ∂dy = ∇softmax(Δ, y; dims)
        T = promote_type(eltype(Δ), eltype(dy), eltype(y))
        ∂y = T.(Δ .* (dy .- sum(dy .* _acc.(y); dims)) .- dy .* sum(Δ .* _acc.(y); dims))
        return (NoTangent(), ∂dy, ∂y)
    end
    return dx, ∇softmax_pullback
end

# `mapreduce(_acc, ...)`, not `reduce(...; init)`: along `dims = 1`, Base reduces each column in the
# eltype of `x`, and BFloat16 has no fast `max_fast`. Its generic `max` branches, ~100× slower.
fast_maximum(x::AbstractArray{T}; dims) where {T} = @fastmath mapreduce(_acc, max, x; dims, init = _acc(float(T)(-Inf)))
# GPUArrays 12 reductions have no fast kernel for `max_fast`, 2–5× slower than `max`
# (JuliaGPU/GPUArrays.jl#833).
fast_maximum(x::AbstractGPUArray{T}; dims) where {T} = reduce(max, x; dims, init = _acc(float(T)(-Inf)))

"""
    logsoftmax(x; dims = 1)

Computes the log of softmax in a more numerically stable
way than directly taking `log.(softmax(xs))`. Commonly used in
computing cross entropy loss.

It is semantically equivalent to the following:

    logsoftmax(x; dims = 1) = x .- log.(sum(exp.(x), dims = dims))

For `Float16` and `BFloat16` inputs, the sum is computed in `Float32`.

See also [`softmax`](@ref).
"""
logsoftmax(x::AbstractArray{T}; dims = 1) where {T} = logsoftmax!(similar(x, float(T)), x; dims)

"""
    logsoftmax!(out, x; dims = 1)
    logsoftmax!(x; dims = 1)

In-place version of [`logsoftmax`](@ref), writing the result into `out` (or `x` itself).
"""
logsoftmax!(x::AbstractArray; dims = 1) = logsoftmax!(x, x; dims)

logsoftmax!(out::AbstractArray, x::AbstractArray; dims = 1) = _logsoftmax!(out, x; dims)
 
function _logsoftmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
    max_ = fast_maximum(x; dims)
    if all(isfinite, max_)
        out .= x .- max_
    else
        _zero, _minf, _inf = _acc(T(0)), _acc(T(-Inf)), _acc(T(Inf))
        @. out = ifelse(isequal(max_,_inf), ifelse(isequal(x,_inf), _zero, _minf), x - max_)
    end
    log_ = log.(sum(@inline(y -> _exp(_acc(y))), out; dims))  # not vectorized without `@inline`
    out .-= log_
end

"""
    ∇logsoftmax(dy, y; dims = 1)

Gradient of [`logsoftmax`](@ref) with respect to its input, given the output `y = logsoftmax(x; dims)`
and the upstream cotangent `dy`. Does not depend on `x`.
"""
function ∇logsoftmax(dy::AbstractArray{T}, y::AbstractArray{S}; dims = 1) where {T,S}
    dx = similar(y, promote_type(T,S))  # sure to be mutable
    ∇logsoftmax!(dx, dy, y; dims)
    return dx
end

"""
    ∇logsoftmax!(dx, dy, y; dims = 1)

In-place version of [`∇logsoftmax`](@ref), writing the gradient into `dx`.
"""
∇logsoftmax!(dx::AbstractArray, dy::AbstractArray, y::AbstractArray; dims = 1) = _∇logsoftmax!(dx, dy, y; dims)

function _∇logsoftmax!(dx::AbstractArray, dy::AbstractArray, y::AbstractArray; dims = 1)
    dx .= dy .- sum(_acc, dy; dims) .* _exp.(_acc.(y))
end

function rrule(::typeof(logsoftmax), x; dims = 1)
    y = logsoftmax(x; dims)
    logsoftmax_pullback(dy) = (NoTangent(), ∇logsoftmax(unthunk(dy), y; dims))
    return y, logsoftmax_pullback
end

# Second-order rule, as for `∇softmax`.
function rrule(::typeof(∇logsoftmax), dy, y; dims = 1)
    dx = ∇logsoftmax(dy, y; dims)
    function ∇logsoftmax_pullback(Δ)
        Δ = unthunk(Δ)
        T = promote_type(eltype(Δ), eltype(y))
        S = promote_type(T, eltype(dy))
        p = _exp.(_acc.(y))
        ∂dy = T.(Δ .- sum(Δ .* p; dims))
        ∂y = S.(.-sum(_acc, dy; dims) .* p .* Δ)
        return (NoTangent(), ∂dy, ∂y)
    end
    return dx, ∇logsoftmax_pullback
end

"""
    logsumexp(x; dims = :)

Computes `log.(sum(exp.(x); dims))` in a numerically stable way.
Without `dims` keyword this returns a scalar.
For `Float16` and `BFloat16` inputs, it is computed and returned in `Float32`.

See also [`logsoftmax`](@ref).
"""
function logsumexp(x::AbstractArray; dims = :)
    max_ = fast_maximum(x; dims)
    return max_ .+ log.(sum(_exp.(x .- max_); dims))
end

function rrule(::typeof(logsumexp), x::AbstractArray{T}; dims = :) where {T}
    # The gradient is `softmax`, but both compute `tmp` and its sum so they are worth saving.
    max_ = fast_maximum(x; dims)
    tmp = _exp.(x .- max_)
    s = sum(tmp; dims)
    y = max_ .+ log.(s)
    logsumexp_pullback(dy) = (NoTangent(), float(T).(unthunk(dy) .* tmp ./ s))
    return y, logsumexp_pullback
end

# Informative error message if any of the softmax variants is called with a number
for f in (:softmax, :logsoftmax, :softmax!, :logsoftmax!, :logsumexp)
    @eval $(f)(x::Number, args...) = 
      error("`", $(string(f)), "(x)` called with a number, but it expects an array. Usually this is because a layer like `Dense(3,4,softmax)` is broadcasting it like an activation function; `softmax` needs to be outside the layer.")
end
