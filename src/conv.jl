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
# implementations. For homogeneous-datatype 1, 2 and 3d convolutions, we default to using
# im2col + GEMM.  Do so in a loop, here:
for (front_name, backend) in (
        # This maps from public, front-facing name, to internal backend name
        :conv                   => :im2col,
        :∇conv_data             => :im2col,
        :∇conv_filter           => :im2col,
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

# Our strategy for 1d and 2d convolution is to reshape to 3d convolutions, which
# makes things MUCH EASIER for us on the backend side, and is in general pretty fast,
# since we can specialize on sizes.
for front_name in (:conv, :∇conv_data, :∇conv_filter)
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
for front_name in (:conv, :∇conv_data, :∇conv_filter)
    @eval begin
        function $(Symbol("$(front_name)!"))(
                        y::AbstractArray{yT,N}, in1::AbstractArray{T1,N},
                        in2::AbstractArray{T2,N}, cdims::ConvDims;
                        kwargs...) where {yT, T1, T2, N}
            @debug string("Slow fallback implementation invoked for $(front_name)!  ",
                          "You probably don't want this; check your datatypes.")
            $(Symbol("$(front_name)_direct!"))(y, in1, in2, cdims; kwargs...)
        end
    end
end

# Finally, let's generate auto-allocating versions of all our functions, for all backends.
# We `@timeit` these methods separately, as we want to know how much time is spent in
# allocation.  :P
for backend in (Symbol(), :_direct, :_im2col)
    # First make auto-allocating versions of the conv()-like calls:
    for name in (:conv,)
        @eval begin
            function $(Symbol("$(name)$(backend)"))(
                            x::AbstractArray{xT,N}, w::AbstractArray{wT,N},
                            cdims::ConvDims; kwargs...) where {xT, wT, N}
                y = similar(x, promote_type(xT, wT), output_size(cdims)...,
                               channels_out(cdims), size(x,N))
                return $(Symbol("$(name)$(backend)!"))(y, x, w, cdims; kwargs...)
            end
        end
    end

    for name in (:∇conv_data,)
        @eval begin
            function $(Symbol("$(name)$(backend)"))(
                            dy::AbstractArray{yT,N}, w::AbstractArray{wT,N},
                            cdims::ConvDims; kwargs...) where {yT, wT, N}
                dx = similar(dy, input_size(cdims)..., channels_in(cdims),
                                                        size(dy, N))
                return $(Symbol("$(name)$(backend)!"))(dx, dy, w, cdims; kwargs...)
            end
        end
    end

    # This filter back prop covers dense/depthwise/groupwise conv filter backprops, as groupcount alone 
    # is a deciding factor from cudnn's perspective. For backends im2col and direct needs to be handled.
    @eval begin
        function $(Symbol("∇conv_filter$(backend)"))(
                        x::AbstractArray{xT,N}, dy::AbstractArray{yT,N},
                        cdims::ConvDims; kwargs...) where {xT, yT, N}
            dw = similar(dy, kernel_size(cdims)..., div(channels_in(cdims),group_count(cdims)),
                                                    channels_out(cdims))
            return $(Symbol("∇conv_filter$(backend)!"))(dw, x, dy, cdims; kwargs...)
        end
    end
end


# Use NNPACK if it is available and the operation is supported
if is_nnpack_available()
    function conv(x::Array{xT, 4}, w::Array{wT, 4},
                  cdims::DenseConvDims{2, K, C_in, C_out, (1, 1), P, (1, 1), F};
                  kwargs...) where {xT, wT, K, C_in, C_out, S, P, F}
        return conv_nnpack(x, w, cdims; kwargs...)
    end
end

function conv(x, w::AbstractArray{T, N}; stride = 1, pad = 0, dilation = 1, flipped = false) where {T, N}
    stride = expand(Val(N-2), stride)
    pad = expand(Val(N-2), pad)
    dilation = expand(Val(N-2), dilation)
    cdims = DenseConvDims(x, w; stride = stride, padding = pad, dilation = dilation, flipkernel = flipped)
    return conv(x, w, cdims)
end

function depthwiseconv(x, w::AbstractArray{T, N}; stride = 1, pad = 0, dilation = 1, flipped = false, groupcount) where {T, N}
    stride = expand(Val(N-2), stride)
    pad = expand(Val(N-2), pad)
    dilation = expand(Val(N-2), dilation)
    cdims = DenseConvDims(x, w; stride = stride, padding = pad, dilation = dilation, flipkernel = flipped, groupcount=groupcount)
    return depthwiseconv(x, w, cdims)
end
