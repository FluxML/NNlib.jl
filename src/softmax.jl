
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

```julia
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
        @fastmath @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 1, 0), exp(x - max_))
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
    fast_exp(x)

For `x::Float32`, this is a much faster (about 20x)
but much less accurate (about 0.1%) version of `exp`.
All other real numbers call `@fastmath exp(x)`.

Handles `Inf` but not `NaN`:
```
julia> xs = Tuple([0, 1, Inf32, -Inf32, NaN32]);

julia> fast_exp.(xs)
(1.0017247f0, 2.717878f0, Inf32, 0.0f0, Inf32)

julia> exp.(xs)
(1.0f0, 2.7182817f0, Inf32, 0.0f0, NaN32)
```
"""
@inline function fast_exp(x::Float32)
    t = x * 1.442695041f0
    i = unsafe_trunc(Int32, t) - signbit(t)
    f = t - i
    f2 = evalpoly(f, (1.00172476f0, 0.657636276f0, 0.3371894346f0))
    y = reinterpret(Float32, reinterpret(Int32, f2) + (i << 23))
    ifelse(x < -87.33655f0, 0.0f0, ifelse(x < 88.72283f0, y, Inf32))
end
# Adapted from code by njuffa which claims /* max. rel. error <= 1.73e-3 on [-87,88] */
# https://stackoverflow.com/questions/10552280/fast-exp-calculation-possible-to-improve-accuracy-without-losing-too-much-perfo/10792321#10792321

# Direct translation to Float16, similar accuracy, twice as fast?
@inline function fast_exp(x::Float16)
    t = x * Float16(1.442)
    i = unsafe_trunc(Int16, t) - signbit(t)
    f = t - i
    f2 = evalpoly(f, (Float16(1.002), Float16(0.6577), Float16(0.3372)))
    y = reinterpret(Float16, reinterpret(Int16, f2) + (i << 10))
    ifelse(x < Float16(-9.7), Float16(-0.0), ifelse(x < Float16(11.09), y, Inf16))
end

fast_exp(x::Real) = @fastmath exp(x)

#=

julia> let x = randn(Float32, 1000)
             y = similar(x)
             @btime $y .= exp.($x)
             @btime @fastmath $y .= exp.($x)
             @btime @turbo $y .= exp.($x)
             @btime $y .= NNlib.fast_exp.($x)
         end;
    min 3.938 μs, mean 3.984 μs (0 allocations)
    min 3.422 μs, mean 3.450 μs (0 allocations)
    min 459.812 ns, mean 462.233 ns (0 allocations)
    min 249.777 ns, mean 251.146 ns (0 allocations)
  
    14.190 μs (0 allocations: 0 bytes)  # another computer
    12.435 μs (0 allocations: 0 bytes)
    1.311 μs (0 allocations: 0 bytes)
    553.774 ns (0 allocations: 0 bytes)

julia> let x = CUDA.randn(Float32, 100, 100_000)
             y = similar(x)
             @btime CUDA.@sync $y .= exp.($x)
             @btime CUDA.@sync @fastmath $y .= exp.($x)
             @btime CUDA.@sync $y .= NNlib.fast_exp.($x)
         end;
  124.673 μs (27 allocations: 1.36 KiB)
  124.202 μs (27 allocations: 1.36 KiB)
  124.066 μs (27 allocations: 1.36 KiB)

=#

export fast_softmax

"""
    fast_softmax(x; dims=1)

For `x::AbstractArray{Float32}`, this is a faster but less accurate `softmax`.

Mean error 0.01% on `x = randn(Float32, ...)`,
about 4 decimal digits worse than `softmax`.
About 5x faster.

# Example
```
julia> [fast_softmax([-Inf32,1,2,3]) softmax([-Inf32,1,2,3])]  # OK with -Inf
4×2 Matrix{Float32}:
 0.0        0.0
 0.0898185  0.0900306
 0.244652   0.244728
 0.66553    0.665241

julia> [fast_softmax([1,Inf32]) softmax([1,Inf32])]  # does not handle +Inf
2×2 Matrix{Float32}:
   0.0  0.0
 NaN    1.0
```
"""
fast_softmax(x::AbstractArray{T}; dims = 1) where {T} = fast_softmax!(similar(x, float(T)), x; dims)
function fast_softmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
    max_ = fast_maximum(x; dims)
    out .= fast_exp.(x .- max_)
    tmp = dims isa Colon ? sum(out) : sum!(max_, out)
    return out ./= tmp
end

function rrule(::typeof(fast_softmax), x; dims = 1)
    y = fast_softmax(x; dims)
    softmax_pullback(dy) = (NoTangent(), ∇softmax_data(unthunk(dy), y; dims))
    return y, softmax_pullback
end

#=

julia> let x = randn(Float32, 100, 1000)  # CPU
           y = similar(x)
           @btime softmax!($y, $x)
           @btime NNlib.fast_softmax!($y, $x)
       end;
  min 647.000 μs, mean 657.488 μs (1 allocation, 4.06 KiB)
  min 133.917 μs, mean 139.647 μs (1 allocation, 4.06 KiB)

  1.646 ms (1 allocation: 4.06 KiB)  # another computer
  322.792 μs (1 allocation: 4.06 KiB)

julia> let x = CUDA.rand(Float32, 100, 1000)  # same (small) size
           y = similar(x)
           @btime CUDA.@sync softmax!($y, $x)
           @btime CUDA.@sync NNlib.fast_softmax!($y, $x)  # faster because it skips a launch
       end;
  151.148 μs (262 allocations: 12.94 KiB)
  78.955 μs (153 allocations: 7.50 KiB)

# removing all(isfinite, max_) check, the full-precision softmax! is as fast:
  79.720 μs (153 allocations: 7.50 KiB)
  80.410 μs (153 allocations: 7.50 KiB)

julia> let x = CUDA.rand(Float32, 100, 10_000)  # 10 times bigger
           y = similar(x)
           @btime CUDA.@sync softmax!($y, $x)
           @btime CUDA.@sync NNlib.fast_softmax!($y, $x)
       end;
  205.560 μs (262 allocations: 12.94 KiB)
  150.375 μs (153 allocations: 7.50 KiB)

# removing all(isfinite, max_) check:
  149.104 μs (153 allocations: 7.50 KiB)
  149.570 μs (153 allocations: 7.50 KiB)

julia> let x = CUDA.rand(Float32, 100, 100_000)  # 100 times bigger
           y = similar(x)
           @btime CUDA.@sync softmax!($y, $x)
           @btime CUDA.@sync NNlib.fast_softmax!($y, $x)
       end;
  1.673 ms (309 allocations: 15.66 KiB)  # difference is noise I think
  1.729 ms (200 allocations: 10.27 KiB)

# removing all(isfinite, max_) check:
  1.740 ms (200 allocations: 10.27 KiB)
  1.708 ms (200 allocations: 10.27 KiB)

=#


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
        @. out = ifelse(isequal(max_,Inf), ifelse(isequal(x,Inf), 0, -Inf), x - max_)
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
