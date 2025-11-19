"""
    within_gradient(x) --> Bool

Returns `false` except when used inside a `gradient` call, when it returns `true`.
Useful for Flux regularisation layers which behave differently during training and inference.

This should work with any ChainRules-based differentiation package, in which case `x` is ignored.
But Tracker.jl overloads `with_gradient(x::TrackedArray)`, thus for widest use you should
pass it an array whose gradient is of interest.
There is also an overload for ForwardDiff.jl's `Dual` types (and arrays of them).

# Examples
```julia-repl
julia> using ForwardDiff, Zygote, NNlib

julia> f_good(x) = if NNlib.within_gradient(x)
                     @show 10x
                   else
                     x
                   end;

julia> Zygote.withgradient(f_good, 1.0)
10x = 10.0
(val = 10.0, grad = (10.0,))

julia> ForwardDiff.derivative(f_good, 1.0)
10x = Dual{ForwardDiff.Tag{typeof(f_good), Float64}}(10.0,10.0)
10.0

julia> f_bad(x, y) = if any(NNlib.within_gradient, (x, y))
                       @show x * y
                     else
                       x / y
                     end;

julia> Zygote.withgradient(f_bad, 2.0, 3.0)
(val = 0.6666666666666666, grad = (0.3333333333333333, -0.2222222222222222))

julia> ForwardDiff.derivative(x -> f_bad(x, 3.0), 2.0)
x * y = Dual{ForwardDiff.Tag{var"#9#10", Float64}}(6.0,3.0)
3.0
```

What goes wrong in `f_bad` is that Zygote knows `any` to be non-differentiable,
and thus completely ignores its contents. This is not a perfect mechanism,
and the only style recommended is precisely that of `f_good` above.
"""
within_gradient(x) = false

ChainRulesCore.rrule(::typeof(within_gradient), x) = true, _ -> (NoTangent(), NoTangent())


"""
    safe_div(x, y)

Returns `x/y` unless `y==0`, in which case it just returns `x`.
(Used internally by `scatter`.)
"""
safe_div(x, y) = ifelse(iszero(y), x, x/y)

"""
    maximum_dims(dims)

Given an array of `CartesianIndex{N}` or `NTuple{N,Int}`,
returns a tuple containing the maximum of all the 1st entries,
all the 2nd entries, and so on up to `N`.

Given an array of integers, returns `(maximum(dims),)`.

(These arguments are what [`scatter`](@ref NNlib.scatter) understands.)
"""
maximum_dims(dims::AbstractArray{<:Integer}) = (maximum(dims), )
maximum_dims(dims::AbstractArray{NTuple{N, T}}) where {N,T} = ntuple(i -> maximum(x->x[i], dims), N)
maximum_dims(dims::AbstractArray{CartesianIndex{N}}) where {N} = ntuple(i -> maximum(x->x[i], dims), N)

function reverse_indices!(rev::AbstractArray, idx::AbstractArray{<:Tuple})
    for (ind, val) in pairs(Array(idx))
        push!(rev[val...], ind)
    end
    # if CUDA supports `unique`, a more efficient version:
    # cidx in CartesianIndices(idx)
    # for i = unique(idx)
    #     rev[i] = cidx[idx .== i]
    # end
    rev
end

function reverse_indices!(rev::AbstractArray, idx::AbstractArray)
    for (ind, val) in pairs(Array(idx))
        push!(rev[val], ind)
    end
    rev
end

"""
    reverse_indices(idx)

Return the reverse indices of `idx`. The indices of `idx` will be values, and values of `idx` will be index.

# Arguments

- `idx`: The indices to be reversed. Accepts array or cuarray of integer, tuple or `CartesianIndex`.
"""
function reverse_indices(idx::AbstractArray{<:Any,N}) where N
    max_dims = maximum_dims(idx)
    T = CartesianIndex{N}
    rev = Array{Vector{T}}(undef, max_dims...)
    for i in eachindex(rev)
        rev[i] = T[]
    end
    return reverse_indices!(rev, idx)
end

unsqueeze(x) = reshape(x, 1, size(x)...)


"""
    _fast_broadcast!(f, x, y, z...)

This does `x .= f.(x, y, z...)`, but works around
an issue with broadcasting that prevents SIMD in such cases.
Can perhaps be removed once https://github.com/JuliaLang/julia/issues/43153 is fixed.

Has an `rrule` to avoid mutation within derivatives.

!!! warning
    Not intended for general use.
    Uses `@inbounds` but does not check sizes!
    Assumes that `f` has no derivative!
"""
function _fast_broadcast!(f::F, x::Array, yz...) where {F<:Function}
    bc = Broadcast.instantiate(Broadcast.broadcasted(f, x, yz...))
    @simd ivdep for I in eachindex(bc)
        @inbounds x[I] = bc[I]
    end
    return x
end
function _fast_broadcast!(f::F, x::AbstractArray, yz...) where {F<:Function}
    # CUDA does not suffer from this bug
    broadcast!(f, x, x, yz...)
end

function rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(_fast_broadcast!), f::F, x::AbstractArray, ys...)  where {F<:Function}
    rrule_via_ad(cfg, broadcast, f, x, ys...)
end


