export upsample, upsample!, ∇upsample, ∇upsample!

for N in (3, 4)
    @eval begin
        function upsample!(y::AbstractArray{T,$N}, x::AbstractArray{T,$N},
                           udims::UpsampleDims) where {T}
            upsample!(
                insert_singleton_spatial_dimension(y, $(5 - N)),
                insert_singleton_spatial_dimension(x, $(5 - N)),
                insert_singleton_spatial_dimension(udims, $(5 - N))
            )

            # We explicitly return `y` here, because the backend call
            # itself may return a reshaped view, which we don't want.
            return y
        end

        function ∇upsample!(dx::AbstractArray{T,$N}, dy::AbstractArray{T,$N},
                            x::AbstractArray{T,$N}, udims::UpsampleDims) where {T}
            ∇upsample!(
                insert_singleton_spatial_dimension(dx, $(5 - N)),
                insert_singleton_spatial_dimension(dy, $(5 - N)),
                insert_singleton_spatial_dimension(x, $(5 - N)),
                insert_singleton_spatial_dimension(udims, $(5 - N))
            )

            # We explicitly return `dx` here, because the backend call
            # itself may return a reshaped view, which we don't want.
            return dx
        end
    end
end

function upsample(x::AbstractArray{T,N}, udims::UpsampleDims) where {T,N}
    y = similar(x, output_size(udims)..., channels_out(udims), size(x, N))
    return upsample!(y, x, udims)
end

function ∇upsample(dy::AbstractArray{T,N}, x::AbstractArray{T},
                   udims::UpsampleDims) where {T,N}
    dx = similar(dy, input_size(udims)..., channels_in(udims), size(dy, N))
    return ∇upsample!(y, x, udims)
end
