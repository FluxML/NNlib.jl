
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

"""
    softmax!(out, x; dims = 1)
    softmax!(x; dims = 1)

In-place version of [`softmax`](@ref), writing the result into `out` (or `x` itself).
"""
softmax!(x::AbstractArray; dims = 1) = softmax!(x, x; dims)

softmax!(out::AbstractArray, x::AbstractArray; dims = 1) = _softmax!(out, x; dims)

# Generic implementation, can be overidden by CUDA and other backends.
function _softmax!(out::AbstractArray{T}, x::AbstractArray; dims = 1) where {T}
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
    dx .= dx .- y .* sum(dx; dims)
end

function rrule(::typeof(softmax), x; dims = 1)
    y = softmax(x; dims)
    softmax_pullback(dy) = (NoTangent(), ∇softmax(unthunk(dy), y; dims))
    return y, softmax_pullback
end

# `∇softmax` mutates, and may call a backend kernel, so nested reverse-mode AD
# differentiates it with this rule. It is linear and self-adjoint in `dy`.
function rrule(::typeof(∇softmax), dy, y; dims = 1)
    dx = ∇softmax(dy, y; dims)
    function ∇softmax_pullback(Δ)
        Δ = unthunk(Δ)
        ∂dy = ∇softmax(Δ, y; dims)
        ∂y = Δ .* (dy .- sum(dy .* y; dims)) .- dy .* sum(Δ .* y; dims)
        return (NoTangent(), ∂dy, ∂y)
    end
    return dx, ∇softmax_pullback
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
        _zero, _minf, _inf = T(0), T(-Inf), T(Inf)
        @. out = ifelse(isequal(max_,_inf), ifelse(isequal(x,_inf), _zero, _minf), x - max_)
    end
    @fastmath log_ = log.(sum(exp, out; dims))
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
    dx .= dy .- sum(dy; dims) .* exp.(y)
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
        p = exp.(y)
        ∂dy = Δ .- sum(Δ .* p; dims)
        ∂y = .-sum(dy; dims) .* p .* Δ
        return (NoTangent(), ∂dy, ∂y)
    end
    return dx, ∇logsoftmax_pullback
end

"""
    logsumexp(x; dims = :)

Computes `log.(sum(exp.(x); dims))` in a numerically stable way.
Without `dims` keyword this returns a scalar.

See also [`logsoftmax`](@ref).
"""
function logsumexp(x::AbstractArray; dims = :)
    max_ = fast_maximum(x; dims)
    return @fastmath max_ .+ log.(sum(exp.(x .- max_); dims))
end

function rrule(::typeof(logsumexp), x; dims = :)
    # The gradient is `softmax`, but both compute `tmp` so it's worth saving.
    max_ = fast_maximum(x; dims)
    @fastmath tmp = exp.(x .- max_)
    @fastmath y = max_ .+ log.(sum(tmp; dims))
    logsumexp_pullback(dy) = (NoTangent(), unthunk(dy) .* tmp ./ sum(tmp; dims))
    return y, logsumexp_pullback
end

## CPU implementation
#
# `softmax!` and `logsoftmax!` of `Array`s of `Float32` or `Float64` take the loops below
# instead of the broadcasts of the generic implementation. They call `_exp`, so that the
# compiler vectorizes them, and they are divided among tasks.
#
# The array is seen as `(pre, n, post)`, with the dimensions to reduce merged into the
# second one. For `pre == 1` a slice is contiguous in memory, and the loops run along it.
# Otherwise they run along the first dimension, over a tile of rows of a slice, with the
# maximum and the sum of every row kept in a workspace.

const SoftmaxFloat = Union{Float32,Float64}

softmax!(out::Array{T}, x::Array{T}; dims = 1) where {T<:SoftmaxFloat} =
    softmax_cpu!(softmax, out, x, dims)
logsoftmax!(out::Array{T}, x::Array{T}; dims = 1) where {T<:SoftmaxFloat} =
    softmax_cpu!(logsoftmax, out, x, dims)

# Spawning tasks costs a few microseconds, and tens of them when the threads are asleep:
# no tasks below 2^17 elements, and at most one for every 2^15 of them. The size is not
# tested in one condition with `should_use_spawn()`: Enzyme then aborts while compiling
# the callers, on an LLVM "inlinable function call ... must have a !dbg location".
function softmax_ntasks(len::Int)
    n = len < 2^17 ? 1 : len >> 15
    return should_use_spawn() ? min(Threads.nthreads(:default), n) : 1
end

# Rows of a tile. The loops are faster on many rows than on few, but the tiles are what
# the tasks share: at least two for each of them.
softmax_tile(pre::Int, post::Int, ntasks::Int) = min(pre, clamp(pre * post ÷ 2ntasks, 64, 512))

# The sizes `(pre, n, post)` for the dimensions `dims`, or `nothing` for what is left to
# the generic implementation: dimensions that are not contiguous, an empty array, arrays
# of different sizes, and arrays that share memory without being the same one.
function softmax_shape(out::Array, x::Array, dims)
    r = softmax_dims(dims, ndims(x))
    (r === nothing || isempty(x) || size(out) != size(x)) && return nothing
    (out === x || !Base.mightalias(out, x)) || return nothing
    pre = n = post = 1
    for d in 1:ndims(x)
        d < r[1] ? (pre *= size(x, d)) : d <= r[2] ? (n *= size(x, d)) : (post *= size(x, d))
    end
    return (pre, n, post)
end

softmax_dims(::Colon, N::Int) = N > 0 ? (1, N) : nothing
softmax_dims(d::Integer, N::Int) = 1 <= d <= N ? (Int(d), Int(d)) : nothing
function softmax_dims(dims::Tuple{Integer,Vararg{Integer}}, N::Int)
    d1, d2 = extrema(dims)
    contiguous = 1 <= d1 && d2 <= N && allunique(dims) && length(dims) == d2 - d1 + 1
    return contiguous ? (Int(d1), Int(d2)) : nothing
end
softmax_dims(dims, N::Int) = nothing

softmax_generic!(::typeof(softmax), out, x, dims) = _softmax!(out, x; dims)
softmax_generic!(::typeof(logsoftmax), out, x, dims) = _logsoftmax!(out, x; dims)

function softmax_cpu!(f::F, out::Array{T}, x::Array{T}, dims) where {F,T}
    shape = softmax_shape(out, x, dims)
    shape === nothing && return softmax_generic!(f, out, x, dims)
    pre, n, post = shape
    maxtasks = softmax_ntasks(length(x))
    # The items the tasks share: the slices if they are contiguous, tiles of rows if not.
    tile = pre == 1 ? 1 : softmax_tile(pre, post, maxtasks)
    nitems = cld(pre, tile) * post
    ntasks = min(maxtasks, nitems)
    if ntasks == 1
        softmax_items!(f, out, x, softmax_workspace(T, pre, tile), pre, n, tile, 1:nitems)
        return out
    end
    # Every task has a range of items of its own, which it takes a chunk at a time. Once
    # done it takes chunks from the ranges of the others, so that a task on a slower or a
    # busy core does not hold the others back.
    bounds = [(t * nitems) ÷ ntasks for t in 0:ntasks]
    nexts = [Threads.Atomic{Int}(bounds[t] + 1) for t in 1:ntasks]
    chunk = max(1, nitems ÷ 16ntasks)
    function run(t)
        ws = softmax_workspace(T, pre, tile)
        for u in t:t+ntasks-1
            r = (u - 1) % ntasks + 1
            while true
                i0 = Threads.atomic_add!(nexts[r], chunk)
                i0 > bounds[r+1] && break
                softmax_items!(f, out, x, ws, pre, n, tile, i0:min(i0 + chunk - 1, bounds[r+1]))
            end
        end
        return nothing
    end
    # the calling task takes the first range: the others may take a while to start
    tasks = [Threads.@spawn(run(t)) for t in 2:ntasks]
    run(1)
    foreach(wait, tasks)
    return out
end

# The maxima and the sums of the rows of a tile.
softmax_workspace(::Type{T}, pre::Int, tile::Int) where {T} = Vector{T}(undef, pre == 1 ? 0 : 2tile)

@inline function softmax_items!(f::F, out::Array{T}, x::Array{T}, ws::Vector{T}, pre::Int, n::Int,
                                tile::Int, items::UnitRange{Int}) where {F,T}
    if pre == 1
        for k in items
            softmax_slice!(f, out, x, (k - 1) * n, n)
        end
    else
        ntiles = cld(pre, tile)
        for item in items
            k, it = divrem(item - 1, ntiles)
            softmax_block!(f, out, x, ws, k * pre * n + it * tile, min(tile, pre - it * tile), pre, n)
        end
    end
    return nothing
end

# `x - m` for the maximum `m` of the slice of `x`. If `m` is infinite, the infinite entries
# share the probability and the others have none.
@inline softmax_shift(x::T, m::T) where {T} = x - m
@inline softmax_shift_inf(x::T, m::T) where {T} =
    ifelse((m == T(Inf)) & (x == T(Inf)), zero(T), x - m)

# `f` of the slice `x[off+1:off+n]`.
@inline function softmax_slice!(f::F, out::Array{T}, x::Array{T}, off::Int, n::Int) where {F,T}
    # The maximum, with the entries brought into [-floatmax, floatmax], NaN included: then
    # none is infinite or NaN, as `@fastmath` takes for granted, and the loop is vectorized.
    hi = floatmax(T)
    m = -hi
    @inbounds @simd for i in off+1:off+n
        v = x[i]
        v = ifelse(v <= hi, v, hi)
        v = ifelse(v >= -hi, v, -hi)
        m = @fastmath max(m, v)
    end
    if abs(m) < hi
        softmax_slice!(f, softmax_shift, out, x, off, n, m)
    else
        # an infinite maximum, a NaN or no finite entry: the maximum again, one entry at a time
        m = T(-Inf)
        @inbounds for i in off+1:off+n
            m = ifelse(x[i] > m, x[i], m)
        end
        softmax_slice!(f, softmax_shift_inf, out, x, off, n, m)
    end
    return nothing
end

@inline function softmax_slice!(f::F, shift::S, out::Array{T}, x::Array{T}, off::Int, n::Int,
                                m::T) where {F,S,T}
    s = zero(T)
    if f === softmax
        @inbounds @simd ivdep for i in off+1:off+n
            e = _exp(shift(x[i], m))
            out[i] = e
            s += e
        end
        r = inv(s)
        @inbounds @simd ivdep for i in off+1:off+n
            out[i] *= r
        end
    else
        @inbounds @simd for i in off+1:off+n
            s += _exp(shift(x[i], m))
        end
        l = log(s)
        @inbounds @simd ivdep for i in off+1:off+n
            out[i] = shift(x[i], m) - l
        end
    end
    return nothing
end

# `f` along the second dimension of `nt` rows of a slice, the first of which starts at
# `x[off+1]`. `ws[1:nt]` receives the maxima of the rows and `ws[nt+1:2nt]` their sums.
@inline function softmax_block!(f::F, out::Array{T}, x::Array{T}, ws::Vector{T}, off::Int, nt::Int,
                                pre::Int, n::Int) where {F,T}
    finite = true
    @inbounds begin
        @simd ivdep for t in 1:nt
            ws[t] = x[off+t]
        end
        for j in 1:n-1
            o = off + j * pre
            @simd ivdep for t in 1:nt
                ws[t] = ifelse(x[o+t] > ws[t], x[o+t], ws[t])
            end
        end
        @simd for t in 1:nt
            finite &= isfinite(ws[t])
        end
    end
    if finite
        softmax_block!(f, softmax_shift, out, x, ws, off, nt, pre, n)
    else
        softmax_block!(f, softmax_shift_inf, out, x, ws, off, nt, pre, n)
    end
    return nothing
end

@inline function softmax_block!(f::F, shift::S, out::Array{T}, x::Array{T}, ws::Vector{T}, off::Int,
                                nt::Int, pre::Int, n::Int) where {F,S,T}
    @inbounds begin
        @simd ivdep for t in 1:nt
            ws[nt+t] = zero(T)
        end
        for j in 0:n-1
            o = off + j * pre
            if f === softmax
                @simd ivdep for t in 1:nt
                    e = _exp(shift(x[o+t], ws[t]))
                    out[o+t] = e
                    ws[nt+t] += e
                end
            else
                @simd ivdep for t in 1:nt
                    ws[nt+t] += _exp(shift(x[o+t], ws[t]))
                end
            end
        end
        @simd ivdep for t in 1:nt
            ws[nt+t] = f === softmax ? inv(ws[nt+t]) : log(ws[nt+t])
        end
        for j in 0:n-1
            o = off + j * pre
            if f === softmax
                @simd ivdep for t in 1:nt
                    out[o+t] *= ws[nt+t]
                end
            else
                @simd ivdep for t in 1:nt
                    out[o+t] = shift(x[o+t], ws[t]) - ws[nt+t]
                end
            end
        end
    end
    return nothing
end

# Informative error message if any of the softmax variants is called with a number
for f in (:softmax, :logsoftmax, :softmax!, :logsoftmax!, :logsumexp)
    @eval $(f)(x::Number, args...) = 
      error("`", $(string(f)), "(x)` called with a number, but it expects an array. Usually this is because a layer like `Dense(3,4,softmax)` is broadcasting it like an activation function; `softmax` needs to be outside the layer.")
end
