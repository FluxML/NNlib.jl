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
for backend in (Symbol(), :_direct, :_nnpack)
    # First make auto-allocating versions of the basic pooling calls:
    for name in (:maxpool, :meanpool)
        @eval begin
            function $(Symbol("$(name)$(backend)"))(
                            x::AbstractArray{xT,N},
                            pdims::PoolDims; kwargs...) where {xT, N}
                y = similar(x, output_size(pdims)..., channels_out(pdims), size(x, N))
                fill!(y, xT(0))
                return $(Symbol("$(name)$(backend)!"))(y, x, pdims; kwargs...)
            end

            # Backprops too
            function $(Symbol("∇$(name)$(backend)"))(
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

## Use NNPACK if it is available and operation is supported.
## The corresponding gradient is not available in NNPACK
## Commented out due to #210
# if is_nnpack_available()
#     function maxpool(x::Array{Float32, 4}, pdims::PoolDims{2, K, S, P, (1, 1)}; kwargs...) where {T, K, S, P}
#         func = nnpack_supported_operation(pdims) ? maxpool_nnpack : maxpool_direct
#         return func(x, pdims; kwargs...)
#     end
# end

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)


"""
    maxpool(x, k::NTuple; pad=0, stride=k)

Perform max pool operation with window size `k` on input tensor `x`.
"""
function maxpool(x, k::NTuple{N, Integer}; pad=0, stride=k) where N
    pad = expand(Val(N), pad)
    stride = expand(Val(N), stride)
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return maxpool(x, pdims)
end


"""
    meanpool(x, k::NTuple; pad=0, stride=k)

Perform mean pool operation with window size `k` on input tensor `x`.
"""
function meanpool(x, k::NTuple{N, Integer}; pad=0, stride=k) where N
    pad = expand(Val(N), pad)
    stride = expand(Val(N), stride)
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return meanpool(x, pdims)
end


for pool in [:maxpool, :meanpool]
    ∇pool = Symbol(:∇, pool)
    pullback = Symbol(pool, :_pullback)
    @eval function rrule(::typeof($pool), x, pdims::PoolDims; kw...)
        Ω = $pool(x, pdims; kw...)
        $pullback(Δ) = (NO_FIELDS, $∇pool(Δ, Ω, x, pdims; kw...), DoesNotExist())
        return Ω, $pullback
    end
end