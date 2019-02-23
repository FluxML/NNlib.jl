export conv, conv!, ∇conv_data, ∇conv_data!, ∇conv_filter, ∇conv_filter!

## Convolution API
#
#  We provide the following generic methods, for 3d, 4d, and 5d tensors, calculating 1d,
#  2d and 3d convolutions, based on the rank of the input tensors, in both mutating and
#  non-mutating auto-allocating variants:
#   - Convolution:
#     - conv(x, w, cdims)
#     - conv!(y, x, w, cdims)
#   - Convolution data backpropagation
#     - ∇conv_data(dy, w, cdims)
#     - ∇conv_data!(dx, dy, w, cdims)
#   - Convolution filter backpropagation
#     - ∇conv_filter(x, dy, cdims)
#     - ∇conv_filter!(dw, x, dy, cdims)
#
#   All methods require a `ConvDims` object to define the dimensions and optional
#   elements of the convolution (padding, stride, dilation, kernel-flipping, etc...),
#   which is easily constructable through something like `DenseConvDims(x, w)`.  All
#   methods take in the `ConvDims` of the associated normal, forward-pass convolution,
#   that is, the following is legal:
#
#       cdims = ConvDims(x, w; stride=2, dilation=(3,2))
#       dx = ∇conv_data(conv(x, w, cdims), w, cdims)



# First, we will define mappings from the generic API names to our accelerated backend
# implementations. For homogenous-datatype 1, 2 and 3d convolutions, we default to using
# im2col + GEMM.  Do so in a loop, here:
for (front_name, backend) in (
        # This maps from public, front-facing name, to internal backend name
        :conv                   => :im2col,
        :∇conv_data             => :im2col,
        :∇conv_filter           => :im2col,
        :depthwiseconv          => :im2col,
        :∇depthwiseconv_data    => :im2col,
        :∇depthwiseconv_filter  => :im2col,
    )

    # These are the GEMM types we will accelerate with `im2col`
    G = Union{[x[2] for x in gemm_datatype_mappings]...}

    # We only define 3d conv primitives, we reshape lower down to get 1d and 2d convolution
    @eval begin
        # im2col-accelerated function forwarding definition
        function $(Symbol("$(front_name)!"))(
                out::AbstractArray{T,5}, in1::AbstractArray{T,5},
                in2::AbstractArray{T,5}, cdims::ConvDims; kwargs...) where {T <: $G}
            $(Symbol("$(front_name)_$(backend)!"))(out, in1, in2, cdims; kwargs...)
        end
    end
end

# Our strategy for 1d and 2d convolution is to reshape from to 3d convolutions, which
# makes things MUCH EASIER for us on the backend side, and is in general pretty fast,
# since we can specialize on sizes.
for front_name in (:conv, :∇conv_data, :∇conv_filter,
                   :depthwiseconv, :∇depthwiseconv_data, :∇depthwiseconv_filter)
    for backend in (Symbol(), :_direct, :_im2col)
        for N in (3, 4)
            @eval begin
                function $(Symbol("$(front_name)$(backend)!"))(
                        y::AbstractArray{yT,$N}, x::AbstractArray{xT,$N},
                        w::AbstractArray{wT,$N}, cdims::ConvDims;
                        kwargs...) where {yT, xT, wT}
                    $(Symbol("$(front_name)$(backend)!"))(
                        insert_singleton_spatial_dimension(y, $(5 - N)),
                        insert_singleton_spatial_dimension(x, $(5 - N)),
                        insert_singleton_spatial_dimension(w, $(5 - N)),
                        insert_singleton_spatial_dimension(cdims, $(5 - N));
                        kwargs...
                    )

                    # We explicitly return `y` here, because the backend call
                    # itself may return a reshaped view, which we don't want.
                    return y
                end
            end
        end
    end
end

# We always support a fallback, non-accelerated path, where we use the direct, but
# slow, implementations.  These should not typically be used, hence the `@debug`,
# but let's ggo ahead and define them first:
for front_name in (:conv, :∇conv_data, :∇conv_filter,
                   :depthwiseconv, :∇depthwiseconv_data, :∇depthwiseconv_filter)
    @eval begin
        function $(Symbol("$(front_name)!"))(out::AbstractArray, in1::AbstractArray,
                                            in2::AbstractArray, cdims::ConvDims; kwargs...)
            @debug "Slow fallback implementation invoked for $(front_name)!  You probably don't want this; check your datatypes."
            $(Symbol("$(front_name)_direct!"))(out, in1, in2, cdims; kwargs...)
        end
    end
end

# Finally, let's generate auto-allocating versions of all our functions, for all backends:
for backend in (Symbol(), :_direct, :_im2col)
    # First make auto-allocating versions of the conv()-like calls:
    for name in (:conv, :depthwiseconv)
        @eval begin
            function $(Symbol("$(name)$(backend)"))(
                    x::AbstractArray{xT,N}, w::AbstractArray{wT,N},
                    cdims::ConvDims; kwargs...) where {xT, wT, N}
                yT = promote_type(xT, wT)
                # Annoyingly, we must allocate with `zeros()` because if we were to use
                # the faster `similar()`, it may have NaNs within it, which will poison
                # the output because we support accumulation (even with `beta = 0` the
                # NaNs poison us as NaN * 0 == NaN).  This is a bit of a shame, but it's
                # not really that bad as if you're truly interested in performance, you
                # should be allocating your own `y` and calling the non-allocating
                # variant of this method anyway.                
                y = zeros(yT, output_size(cdims)..., channels_out(cdims), size(x, N))
                return $(Symbol("$(name)$(backend)!"))(y, x, w, cdims; kwargs...)
            end
        end
    end

    for name in (:∇conv_data, :∇depthwiseconv_data)
        @eval begin
            function $(Symbol("$(name)$(backend)"))(
                    dy::AbstractArray{yT,N}, w::AbstractArray{wT,N},
                    cdims::cdT; kwargs...) where {yT, wT, N, cdT <: ConvDims}
                # Again, allocate with zeros
                dx = zeros(yT, input_size(cdims)..., channels_in(cdims), size(dy, N))
                return $(Symbol("$(name)$(backend)!"))(dx, dy, w, cdims; kwargs...)
            end
        end
    end

    # We do the conv/depthwiseconv filter backprops separately, as the shape calculation
    # for `w` is slightly different for depthwise than for normal dense convolution.
    @eval begin
        function $(Symbol("∇conv_filter$(backend)"))(
                x::AbstractArray{xT,N}, dy::AbstractArray{yT,N},
                cdims::cdT; kwargs...) where {xT, yT, N, cdT <: ConvDims}
            # Again, allocate with zeros
            dw = zeros(yT, kernel_size(cdims)..., channels_in(cdims),
                                                   channels_out(cdims))
            return $(Symbol("∇conv_filter$(backend)!"))(dw, x, dy, cdims; kwargs...)
        end
    end

    @eval begin
        function $(Symbol("∇depthwiseconv_filter$(backend)"))(
                x::AbstractArray{xT,N}, dy::AbstractArray{yT,N},
                cdims::cdT; kwargs...) where {xT, yT, N, cdT <: ConvDims}
            # Again, allocate with zeros
            dw = zeros(yT, kernel_size(cdims)..., channel_multiplier(cdims),
                                                  channels_in(cdims))
            return $(Symbol("∇depthwiseconv_filter$(backend)!"))(dw, x, dy, cdims;
                                                                 kwargs...)
        end
    end
end
