export conv_bias_act, conv_bias_act!

"""
    conv_bias_act(x, w, cdims, b, σ; kw...)

This should be equivalent to `σ.(conv(x, w, cdims) .+ reshape(b, ...))`, 
but faster because it will:
* try hard to re-use memory, when safe to do so,
* call fused CUDNN operation, when possible

Keyword arguments are passed to the CUDA version.


cudnnConvolutionForward! only supports relu and identity. 
So let's only call conv_bias_act! with those, ever,
even though it has some fallback code.
"""
function conv_bias_act(x::AbstractArray{xT,N}, w::AbstractArray{wT,N}, 
                cdims::ConvDims, b, σ; kwargs...) where {xT, wT, N}

    y = similar(x, promote_type(xT, wT), output_size(cdims)..., channels_out(cdims), size(x,N))
    if σ === identity || σ === relu
        # On the GPU, cudnnConvolutionForward! is fast but only supports relu and identity.
        # Best path forwards:
        return conv_bias_act!(y, x, w, cdims, b, σ; kwargs...)
    else
        # In other cases, still want to call it, but then handle the activation ourselves:
        conv_bias_act!(y, x, w, cdims, b, identity; kwargs...)
        # For nice activation functions including `relu` and `tanh`, this will over-write `y`
        # because it is not needed for the gradient. But in the general case, it will broadacst
        # making `z = σ.(y)`, and the gradient will need both `y` and `z`.
        return bias_act!(σ, y)
    end
    # But on the CPU, this is sub-optimal? Maybe not if we're careful.
end

# Best case, with a nice activation function, we can do much of the gradient calculation in-place:
for (act, grad) in INPLACE_ACTS

    @eval function rrule(::typeof(conv_bias_act), x, w, cdims, b::B, σ; kw...) where {B}
        Ω = conv_bias_act(x, w, cdims, b, σ; kw...)
        size_b = size(b)
        # TODO pull this out?
        function conv_bias_act_pullback(Δ_raw)
            Δ_out = colmajor(unthunk(Δ_raw)) # gradient outside the activation function
            Δ = @. Δ_out * $grad             # inside it -- we can overwrite Ω
            if eltype(B) == Bool
                Δb = NoTangent()
            else
                # Δb = sum!(similar(B, size_b), Δ)
                dims = filter(d -> get(size_b, d, 1)==1, ntuple(identity, ndims(Δ)))
                Δb = reshape(sum(Δ; dims = dims), size_b)
            end
            return (
                NoTangent(), # func
                @thunk(∇conv_data(Δ, w, cdims; kw...)),
                @thunk(∇conv_filter(x, Δ, cdims; kw...)),
                NoTangent(), # cdims
                Δb,
                NoTangent(), # σ
            )
        end
        return Ω, conv_bias_act_pullback
    end

end

# Generic case: `bias_act!(σ, y) = σ.(y)` is handling the gradient of the activation,
# we only need to handle the gradient of the convolution inside.
function rrule(::typeof(conv_bias_act!), y, x, w, cdims, b::B, σ::typeof(identity); kw...) where {B}
    Ω = conv_bias_act!(y, x, w, cdims, b, σ; kw...)
    size_b = size(b)
    function conv_bias_act!_pullback(Δ_raw)
        Δ = colmajor(unthunk(Δ_raw)) 
        if eltype(B) == Bool
            Δb = NoTangent()
        else
            # Δb = sum!(similar(B, size_b), Δ)
            dims = filter(d -> get(size_b, d, 1)==1, ntuple(ideneity, ndims(Δ)))
            Δb = reshape(sum(Δ; dims = dims), size_b)
        end
        return (
            NoTangent(), # func
            NoTangent(), # y
            @thunk(∇conv_data(Δ, w, cdims; kw...)),
            @thunk(∇conv_filter(x, Δ, cdims; kw...)),
            NoTangent(), # cdims
            Δb,
            NoTangent(), # σ
        )
    end
    return Ω, conv_bias_act_pullback
end


# CPU actor. 
function conv_bias_act!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5}, w::AbstractArray{wT,5}, 
                cdims::ConvDims, b, σ; alpha=1, beta=0, algo=nothing) where {yT, xT, wT}
    # The GPU version accepts alpha, beta. Silently ignoring them may mean giving wrong answers.
    (alpha == 1 && beta == 0) || throw(ArgumentError("CPU version of `conv_bias_act!` accepts only `alpha=1, beta=0` right now"))
    y = conv!(y, x, w, cdims)
    if σ != identity || b isa AbstractArray
        return bias_act!(σ, y, b)
    else
        # nothing to do
        return y
    end
end

# Reshape all cases to 5-dim arrays:
for N in (3, 4)
    @eval begin
        function conv_bias_act!(
                        y::AbstractArray{yT,$N}, x::AbstractArray{xT,$N},
                        w::AbstractArray{wT,$N}, cdims::ConvDims,
                        b, σ; kwargs...) where {yT, xT, wT}
            conv_bias_act!(
                insert_singleton_spatial_dimension(y, $(5 - N)),
                insert_singleton_spatial_dimension(x, $(5 - N)),
                insert_singleton_spatial_dimension(w, $(5 - N)),
                insert_singleton_spatial_dimension(cdims, $(5 - N)),
                # This has a fall-through method for non-array b::Bool etc:
                insert_singleton_spatial_dimension(b, $(5 - N)),
                σ; kwargs...)
            # We explicitly return `y` here, because the backend call
            # itself may return a reshaped view, which we don't want.
            return y
        end
    end
end
