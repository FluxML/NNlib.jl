export scatter_add!, scatter_sub!, scatter_max!, scatter_min!, scatter_mul!, scatter_div!,
    scatter_mean!

const IntOrTuple = Union{Integer,Tuple}
const ops = [:add, :sub, :mul, :div, :max, :min, :mean]
const name2op = Dict(:add => :+, :sub => :-, :mul => :*, :div => :/)

"""
    scatter_add!(ys, us, xs)

Scatter addition operation.

    ys[xs[k]...] = ys[xs[k]...] .+ us[k...]

# Arguments
- `ys`: the destination for `us` to aggregate to. This argument will be mutated.
- `us`: the source data for aggregating.
- `xs`: the mapping for aggregation from source (index) to destination (value).
The index of `xs` is corresponding to the index of `us` and the value of `xs` is
corresponding to the index of `ys`. The value of `xs` can be `Int` or `Tuple` type.

The dimension of `us` must equal to dimension of `xs`. `ys`, `us` and `xs` must be
supported array type and be the same type.`Array`, `StaticArray` and `CuArray`
are currently supported.
"""
function scatter_add! end

"""
    scatter_sub!(ys, us, xs)

Scatter subtraction operation.

    ys[xs[k]...] = ys[xs[k]...] .- us[k...]

# Arguments
- `ys`: the destination for `us` to aggregate to. This argument will be mutated.
- `us`: the source data for aggregating.
- `xs`: the mapping for aggregation from source (index) to destination (value).
The index of `xs` is corresponding to the index of `us` and the value of `xs` is
corresponding to the index of `ys`. The value of `xs` can be `Int` or `Tuple` type.

The dimension of `us` must equal to dimension of `xs`. `ys`, `us` and `xs` must be
supported array type and be the same type.`Array`, `StaticArray` and `CuArray`
are currently supported.
"""
function scatter_sub! end

"""
    scatter_mul!(ys, us, xs)

Scatter multiplication operation.

    ys[xs[k]...] = ys[xs[k]...] .* us[k...]

# Arguments
- `ys`: the destination for `us` to aggregate to. This argument will be mutated.
- `us`: the source data for aggregating.
- `xs`: the mapping for aggregation from source (index) to destination (value).
The index of `xs` is corresponding to the index of `us` and the value of `xs` is
corresponding to the index of `ys`. The value of `xs` can be `Int` or `Tuple` type.

The dimension of `us` must equal to dimension of `xs`. `ys`, `us` and `xs` must be
supported array type and be the same type.`Array`, `StaticArray` and `CuArray`
are currently supported.
"""
function scatter_mul! end

"""
    scatter_div!(ys, us, xs)

Scatter division operation.

    ys[xs[k]...] = ys[xs[k]...] ./ us[k...]

# Arguments
- `ys`: the destination for `us` to aggregate to. This argument will be mutated.
- `us`: the source data for aggregating.
- `xs`: the mapping for aggregation from source (index) to destination (value).
The index of `xs` is corresponding to the index of `us` and the value of `xs` is
corresponding to the index of `ys`. The value of `xs` can be `Int` or `Tuple` type.

The dimension of `us` must equal to dimension of `xs`. `ys`, `us` and `xs` must be
supported array type and be the same type.`Array`, `StaticArray` and `CuArray`
are currently supported.
"""
function scatter_div! end

for op = [:add, :sub, :mul, :div]
    fn = Symbol("scatter_$(op)!")
    @eval function $fn(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
        @simd for k = 1:length(xs)
            k = CartesianIndices(xs)[k]
            ys_v = view(ys, xs[k]...)
            us_v = view(us, k)
            @inbounds ys_v .= $(name2op[op]).(ys_v, us_v)
        end
        ys
    end
end

"""
    scatter_max!(ys, us, xs)

Scatter maximum operation.

    ys[xs[k]...] = max.(ys[xs[k]...], us[k...])

# Arguments
- `ys`: the destination for `us` to aggregate to. This argument will be mutated.
- `us`: the source data for aggregating.
- `xs`: the mapping for aggregation from source (index) to destination (value).
The index of `xs` is corresponding to the index of `us` and the value of `xs` is
corresponding to the index of `ys`. The value of `xs` can be `Int` or `Tuple` type.

The dimension of `us` must equal to dimension of `xs`. `ys`, `us` and `xs` must be
supported array type and be the same type.`Array`, `StaticArray` and `CuArray`
are currently supported.
"""
function scatter_max!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    @simd for k = 1:length(xs)
        k = CartesianIndices(xs)[k]
        ys_v = view(ys, xs[k]...)
        us_v = view(us, k)
        @inbounds ys_v .= max.(ys_v, us_v)
    end
    ys
end

"""
    scatter_min!(ys, us, xs)

Scatter minimum operation.

    ys[xs[k]...] = min.(ys[xs[k]...], us[k...])

# Arguments
- `ys`: the destination for `us` to aggregate to. This argument will be mutated.
- `us`: the source data for aggregating.
- `xs`: the mapping for aggregation from source (index) to destination (value).
The index of `xs` is corresponding to the index of `us` and the value of `xs` is
corresponding to the index of `ys`. The value of `xs` can be `Int` or `Tuple` type.

The dimension of `us` must equal to dimension of `xs`. `ys`, `us` and `xs` must be
supported array type and be the same type.`Array`, `StaticArray` and `CuArray`
are currently supported.
"""
function scatter_min!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    @simd for k = 1:length(xs)
        k = CartesianIndices(xs)[k]
        ys_v = view(ys, xs[k]...)
        us_v = view(us, k)
        @inbounds ys_v .= min.(ys_v, us_v)
    end
    ys
end

"""
    scatter_mean!(ys, us, xs)

Scatter mean operation.

    ys[xs[k]...] = mean.(ys[xs[k]...], us[k...])

# Arguments
- `ys`: the destination for `us` to aggregate to. This argument will be mutated.
- `us`: the source data for aggregating.
- `xs`: the mapping for aggregation from source (index) to destination (value).
The index of `xs` is corresponding to the index of `us` and the value of `xs` is
corresponding to the index of `ys`. The value of `xs` can be `Int` or `Tuple` type.

The dimension of `us` must equal to dimension of `xs`. `ys`, `us` and `xs` must be
supported array type and be the same type.`Array`, `StaticArray` and `CuArray`
are currently supported.
"""
function scatter_mean!(ys::Array{T}, us::Array{T}, xs::Array{<:IntOrTuple}) where {T<:Real}
    Ns = zero(ys)
    ys_ = zero(ys)
    scatter_add!(Ns, one.(us), xs)
    scatter_add!(ys_, us, xs)
    ys .+= save_div.(ys_, Ns)
    return ys
end


# Support different types of array

for op = ops
    fn = Symbol("scatter_$(op)!")
    @eval function $fn(ys::Array{T}, us::Array{S}, xs::Array{<:IntOrTuple}) where {T<:Real,S<:Real}
        PT = promote_type(T, S)
        $fn(PT.(ys), PT.(us), xs)
    end
end
