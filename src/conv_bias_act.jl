function conv_bias_act(x::AbstractArray{xT,N}, w::AbstractArray{wT,N},
                cdims::ConvDims, b::AbstractArray{bT,N}, σ=identity; kwargs...) where {xT, wT, bT, N}
    y = similar(x, promote_type(xT, wT, bT), output_size(cdims)..., channels_out(cdims), size(x,N))
    conv_bias_act!(y, x, w, cdims, b, σ; kwargs...)
    return y
end

function conv_bias_act!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5}, w::AbstractArray{wT,5},
                cdims::ConvDims, b::AbstractArray{bT,5}, σ=identity; kwargs...) where {yT, xT, wT, bT}
    conv!(y, x, w, cdims)
    y .= σ.(y .+ b)
    return y
end

for N in (3, 4)
    @eval begin
        function $(Symbol("conv_bias_act!"))(
                        y::AbstractArray{yT,$N}, x::AbstractArray{xT,$N},
                        w::AbstractArray{wT,$N}, cdims::ConvDims,
                        b::AbstractArray{bT,$N}, σ=identity;
                        kwargs...) where {yT, xT, wT, bT}
            $(Symbol("conv_bias_act!"))(
                insert_singleton_spatial_dimension(y, $(5 - N)),
                insert_singleton_spatial_dimension(x, $(5 - N)),
                insert_singleton_spatial_dimension(w, $(5 - N)),
                insert_singleton_spatial_dimension(cdims, $(5 - N)),
                insert_singleton_spatial_dimension(b, $(5 - N)),
                σ;
                kwargs...
            )

            # We explicitly return `y` here, because the backend call
            # itself may return a reshaped view, which we don't want.
            return y
        end
    end
end
