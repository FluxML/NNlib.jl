export maxpool, maxpool!, meanpool, meanpool!, ∇maxpool, ∇maxpool!, ∇meanpool, ∇meanpool!

## Pooling API
#
#  We provide the following generic methods, for 3d, 4d, and 5d tensors, calculating 1d,
#  2d and 3d pooling, based on the rank of the input tensors, in both mutating and
#  non-mutating auto-allocating variants:
#   - Pooling:
#     - maxpool(x, pdims)
#     - maxpool!(y, x, pdims)
#     - meanpool(x, pdims)
#     - meanpool!(y, x, pdims)
#   - Pooling input backprop
#     - ∇maxpool(dy, y, x, pdims)
#     - ∇maxpool!(dx, dy, y, x, pdims)
#     - ∇meanpool(dy, y, x, pdims)
#     - ∇meanpool!(dx, dy, y, x pdims)
#
#   All methods require a `PoolDims` object to define the dimensions and optional
#   elements of the convolution (stride, dilation, etc...), which is easily constructable
#   through something like `PoolDims(x, w)`.


# First, we will define mappings from the generic API names to our accelerated backend
# implementations.  At the moment this is only the direct implementation, however this
# exists here so that other packages (NNPACK, MAGMA, etc...) can override this easily.
for (front_name, backend) in (
        # This maps from public, front-facing name, to internal backend name
        :maxpool  => :direct,
        :meanpool => :direct,
    )

    # We only define 3d pooling primitives, we reshape lower down to get 1d and 2d pooling
    @eval begin
        function $(Symbol("$(front_name)!"))(
                y::AbstractArray{T,5}, x::AbstractArray{T,5},
                pdims::PoolDims; kwargs...) where {T}
            $(Symbol("$(front_name)_$(backend)!"))(y, x, pdims; kwargs...)
        end
    end
end

# Do the same for backprops
for (front_name, backend) in (
        :∇maxpool  => :direct,
        :∇meanpool => :direct,
    )
    @eval begin
        function $(Symbol("$(front_name)!"))(
                        dx::AbstractArray{T,5}, dy::AbstractArray{T,5},
                        y::AbstractArray{T,5}, x::AbstractArray{T,5},
                        pdims::PoolDims; kwargs...) where {T}
            $(Symbol("$(front_name)_$(backend)!"))(dx, dy, y, x, pdims; kwargs...)
        end
    end
end


# Our strategy for pooling is to reshape to an array with three spatial dimensions, which
# makes things MUCH EASIER for us on the backend side, and is in general pretty fast,
# since we can specialize on sizes.
for front_name in (:maxpool, :meanpool)
    for backend in (Symbol(), :_direct)
        for N in (3, 4)
            @eval begin
                function $(Symbol("$(front_name)$(backend)!"))(
                                y::AbstractArray{T,$N}, x::AbstractArray{T,$N},
                                pdims::PoolDims; kwargs...) where {T}
                    $(Symbol("$(front_name)$(backend)!"))(
                        insert_singleton_spatial_dimension(y, $(5 - N)),
                        insert_singleton_spatial_dimension(x, $(5 - N)),
                        insert_singleton_spatial_dimension(pdims, $(5 - N));
                        kwargs...
                    )

                    # We explicitly return `y` here, because the backend call
                    # itself may return a reshaped view, which we don't want.
                    return y
                end

                # backprops too
                function $(Symbol("∇$(front_name)$(backend)!"))(
                                dx::AbstractArray{T,$N}, dy::AbstractArray{T,$N},
                                y::AbstractArray{T,$N}, x::AbstractArray{T,$N},
                                pdims::PoolDims; kwargs...) where {T}
                    $(Symbol("∇$(front_name)$(backend)!"))(
                        insert_singleton_spatial_dimension(dx, $(5 - N)),
                        insert_singleton_spatial_dimension(dy, $(5 - N)),
                        insert_singleton_spatial_dimension(y, $(5 - N)),
                        insert_singleton_spatial_dimension(x, $(5 - N)),
                        insert_singleton_spatial_dimension(pdims, $(5 - N));
                        kwargs...
                    )

                    # We explicitly return `dx` here, because the backend call
                    # itself may return a reshaped view, which we don't want.
                    return dx
                end
            end
        end
    end
end


# Finally, let's generate auto-allocating versions of all our functions, for all backends:
for backend in (Symbol(), :_direct, :_im2col)
    # First make auto-allocating versions of the basic pooling calls:
    for name in (:maxpool, :meanpool)
        @eval begin
            @timeit_debug to function $(Symbol("$(name)$(backend)"))(
                            x::AbstractArray{xT,N},
                            pdims::PoolDims; kwargs...) where {xT, N}
                y = similar(x, output_size(pdims)..., channels_out(pdims), size(x, N))
                fill!(y, xT(0))
                return $(Symbol("$(name)$(backend)!"))(y, x, pdims; kwargs...)
            end
            
            # Backprops too
            @timeit_debug to function $(Symbol("∇$(name)$(backend)"))(
                            dy::AbstractArray{T,N}, y::AbstractArray{T,N},
                            x::AbstractArray{T,N}, pdims::PoolDims;
                            kwargs...) where {T, N}
                dx = similar(x, input_size(pdims)..., channels_in(pdims), size(dy, N))
                fill!(dx, T(0))
                return $(Symbol("∇$(name)$(backend)!"))(dx, dy, y, x, pdims; kwargs...)
            end
        end
    end
end
