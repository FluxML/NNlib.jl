
export conv_bias_act

"""
    conv_bias_act(x, w, cdims, b, σ; kw...)

This should be equivalent to `σ.(conv(x, w, cdims) .+ reshape(b, ...))`, 
but faster because it will:
* try hard to re-use memory, when safe to do so,
* call fused CUDNN operation, when possible.

Keyword arguments are passed to the CUDA version.
"""
function conv_bias_act(x::AbstractArray{xT,N}, w::AbstractArray{wT,N}, 
                cdims::ConvDims, b, σ; kwargs...) where {xT, wT, N}
    y = similar(x, promote_type(xT, wT), output_size(cdims)..., channels_out(cdims), size(x,N))
    if σ === identity || σ === relu
        # On the GPU, cudnnConvolutionForward! is fast but only supports relu and identity.
        return conv_bias_act!(y, x, w, cdims, b, σ; kwargs...)
    else
        # In other cases, still want to call it, but then handle the activation ourselves:
        conv_bias_act!(y, x, w, cdims, b, identity; kwargs...)
        # For nice activation functions including `tanh`, this will over-write `y`
        # because it is not needed for the gradient. Otherwise, it's `z = σ.(y)`:
        return bias_act!(σ, y)
    end
end

# CPU actor:
function conv_bias_act!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5}, w::AbstractArray{wT,5}, 
                cdims::ConvDims, b, σ; alpha=1, beta=0, algo=nothing) where {yT, xT, wT}
    # The GPU version accepts alpha, beta. Silently ignoring them may mean giving wrong answers.
    (alpha == 1 && beta == 0) || throw(ArgumentError("CPU version of `conv_bias_act!` accepts only `alpha=1, beta=0` right now"))
    y = conv!(y, x, w, cdims)
    if σ != identity || b isa AbstractArray
        return bias_act!(σ, y, b)
    elseif iszero(b)
        # nothing to do
        return y
    else
        throw(ArgumentError("The bias argument to `conv_bias_act!` must either be zero or an array."))
    end
end

# Gradient rule, only handles exactly two activation functions!
function rrule(::typeof(conv_bias_act!), y, x, w, cdims, b::B, σ::F; kw...) where {B, F}
    Ω = conv_bias_act!(y, x, w, cdims, b, σ; kw...)
    if eltype(B) != Bool
        b_dims = ntuple(d -> size(b, d)==1 ? d : ndims(x)+1, ndims(x))
        proj_b = ProjectTo(b)
    end
    function conv_bias_act!_pullback(Δ_raw)
        if σ === identity
            Δ = colmajor(unthunk(Δ_raw))
        elseif σ === relu
            Δ = colmajor(unthunk(Δ_raw) .* (Ω .> 0))
        else
            throw(ArgumentError("`conv_bias_act!` should only be called with `identity` or `relu`. `conv_bias_act` ensures this."))
        end
        if eltype(B) == Bool
            Δb = NoTangent()
        else
            Δb = proj_b(sum(Δ; dims = b_dims))
        end
        return (
            NoTangent(), # func
            NoTangent(), # y
            @thunk(ProjectTo(x)(∇conv_data(Δ, w, cdims; kw...))),
            @thunk(ProjectTo(w)(∇conv_filter(x, Δ, cdims; kw...))),
            NoTangent(), # cdims
            Δb,
            NoTangent(), # σ
        )
    end
    return Ω, conv_bias_act!_pullback
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
