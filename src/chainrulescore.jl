using ChainRulesCore, ZygoteRules
using ArrayLayouts: MemoryLayout, AbstractColumnMajor

const Numeric = Union{AbstractArray{<:T}, T} where {T<:Number}

@scalar_rule(selu(x), dselu(x))
@scalar_rule(elu(x, α), (delu(x, α), DoesNotExist()))
@scalar_rule(σ(x::Real), Ω * (1 - Ω))

function dselu(x)
    λ = oftype(x/1, 1.0507009873554804934193349852946)
    α = oftype(x/1, 1.6732632423543772848170429916717)
    return λ * ifelse(x > 0, one(x), α * exp(x))
end
delu(x, α) = ifelse(x ≥ 0, one(x), α * exp(x))

# This is a performance hack specifically for Zygote, because it doesn't handle fused
# broadcasts well
for (f, df) in [
    (:relu, :(x .> 0)),
    (:selu, :(dselu.(x))),
    (:elu, :(delu.(x))),
    (:σ, :(conj.(Ω .* (1 .- Ω)))),
]
    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval @adjoint function Base.Broadcast.broadcasted(::typeof($f), x::Numeric)
        Ω = $f.(x)
        $pullback(Δ) = (nothing, Δ .* $df)
        return Ω, $pullback
    end
end

for softmax in [:softmax, :logsoftmax]
    local ∇softmax = Symbol(:∇, softmax)
    pullback = Symbol(softmax, :_pullback)

    @eval function ChainRulesCore.rrule(::typeof($softmax), xs; dims=1)
        $pullback(Δ) = (NO_FIELDS, @thunk($∇softmax(Δ, xs, dims=dims)))
        return $softmax(xs; dims=dims), $pullback
    end
end

for Dims in [:DenseConvDims, :DepthwiseConvDims, :PoolDims]
    pullback = Symbol(Dims, :_pullback)
    @eval function ChainRulesCore.rrule(::Type{$Dims}, args...; kwargs...)
        $pullback(Δ) = (NO_FIELDS, ntuple(_ -> DoesNotExist(), length(args))...)
        return $Dims(args...; kwargs...), $pullback
    end
end

colmajor(x) = colmajor(MemoryLayout(typeof(x)), x)
colmajor(_, x) = convert(Array, x)
colmajor(::AbstractColumnMajor, x) = x

for conv in [:conv, :depthwiseconv]
    local ∇conv_data, ∇conv_filter = Symbol.(:∇, conv, [:_data, :_filter])
    conv_pullback, ∇conv_data_pullback = Symbol.([conv, ∇conv_data], :_pullback)

    @eval function ChainRulesCore.rrule(::typeof($conv), x, w, cdims; kw...)
        function $conv_pullback(Δ)
            Δ = colmajor(Δ)
            return (
                NO_FIELDS,
                @thunk($∇conv_data(Δ, w, cdims, kw...)),
                @thunk($∇conv_filter(x, Δ, cdims, kw...)),
                DoesNotExist(),
            )
        end
        return $conv(x, w, cdims; kw...), $conv_pullback
    end

    @eval function ChainRulesCore.rrule(::typeof($∇conv_data), x, w, cdims; kw...)
        function $∇conv_data_pullback(Δ)
            Δ = colmajor(Δ)
            return (
                NO_FIELDS,
                @thunk($conv(Δ, w, cdims, kw...)),
                @thunk($∇conv_filter(Δ, x, cdims, kw...)),
                DoesNotExist(),
            )
        end
        return $∇conv_data(x, w, cdims; kw...), $∇conv_data_pullback
    end
end

for pool in [:maxpool, :meanpool]
    ∇pool = Symbol(:∇, pool)
    pullback = Symbol(pool, :_pullback)
    @eval function ChainRulesCore.rrule(::typeof($pool), x, pdims::PoolDims; kw...)
        Ω = $pool(x, pdims; kw...)
        $pullback(Δ) = (NO_FIELDS, @thunk($∇pool(Δ, Ω, x, pdims; kw...)), DoesNotExist())
        return Ω, $pullback
    end
end

function ChainRulesCore.rrule(
    ::typeof(batched_mul),
    A::AbstractArray{S,3},
    B::AbstractArray{T,3},
) where {S,T}
    function batched_mul_pullback(Δ)
        return (
            NO_FIELDS,
            @thunk(batched_mul(Δ, batched_adjoint(B))),
            @thunk(batched_mul(batched_adjoint(A), Δ)),
        )
    end
    batched_mul(A, B), batched_mul_pullback
end
