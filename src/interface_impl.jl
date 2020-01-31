## This file creates the mappings from `conv!()` -> `conv_nnpack!()`, as well as
#  convenience functions such as `conv_nnpack()` -> `conv_nnpack!()`, the auto-allocating
#  variants, the reshaping variants, etc...

       # convolution
export conv, conv!, ∇conv_data, ∇conv_data!, ∇conv_filter, ∇conv_filter!,
       # depthwise convolution
       depthwiseconv, depthwiseconv!, ∇depthwiseconv_data, ∇depthwiseconv_data!,
       ∇depthwiseconv_filter, ∇depthwiseconv_filter!,
       # pooling
       maxpool, maxpool!, meanpool, meanpool!, ∇maxpool, ∇maxpool!, ∇meanpool, ∇meanpool!


# We're going to use some typenames an awful lot
const AA = AbstractArray

# Luckily, because the dispatch signatures for each of these backends are distinct,
# we can just layer the applicable backends and multiple dispatch will take care of
# calling the correct version for us.
for (front_name, backends) in conv_backends
    # _nnpack() methods are generally preferrable, but they're pretty strict
    # in what they can do; they can't deal with anything other than Float32,
    # they only do conv2d, and they can't deal with stride or dilation.
    if :nnpack in backends
        # Only conv2d, no conv1d or conv3d.  Also no stride or dilation
        nnpack_cdims = DenseConvDims{
            # only conv2d, no conv1d or conv3d
            2,
            # Any kernel size, channels in or out, whatever, it's good.
            K, C_in, C_out,
            # No stride
            (1, 1),
            # Padding can be whatevs
            P,
            # No dilation
            (1, 1),
            # Flipping is fine
            F,
        } where {K, C_in, C_out, P, F}

        # Map from the front name to the back name
        fname = Symbol("$(front_name)!")
        bname = Symbol("$(front_name)_nnpack!")
        @eval begin
            function $(fname)(out::AA{Float32, 4}, in1::AA{Float32, 4},
                              in2::AA{Float32, 4}, cdims::$(nnpack_cdims);
                              kwargs...)
                $(bname)(out, in1, in2, cdims; kwargs...)
                return out
            end
        end
    end

    # _im2col() methods are a little less strict.  They can deal with any kind
    # of DenseConvDims, but they are still limited to the basic types that BLAS
    # can deal with, which is {Complex,}Float{32,64}.  For BLAS to work, all
    # types must also be the same:
    if :im2col in backends
        # These are the types that our BLAS can handle
        BLAS_TYPES = Union{[x[2] for x in Im2col.gemm_datatype_mappings]...}

        # Map from the front name to the back name
        fname = Symbol("$(front_name)!")
        bname = Symbol("$(front_name)_im2col!")
        @eval begin
            function $(fname)(out::AA{T}, in1::AA{T},
                              in2::AA{T}, cdims::ConvDims;
                              kwargs...) where {T <: $(BLAS_TYPES)}
                $(bname)(out, in1, in2, cdims; kwargs...)
                return out
            end

            # We add here some "expanders" that convert conv1d/2d inputs up to
            # the conv3d input shape that our backend is expecting:
            function $(bname)(out::AA{T,3}, in1::AA{T,3}, in2::AA{T,3},
                              cdims::ConvDims; kwargs...) where {T <: $(BLAS_TYPES)}
                out, in1, in2, cdims = expand_dimensions(Val(5), out, in1, in2, cdims)
                $(bname)(out, in1, in2, cdims; kwargs...)
                return out
            end
            function $(bname)(out::AA{T,4}, in1::AA{T,4}, in2::AA{T,4},
                              cdims::ConvDims; kwargs...) where {T <: $(BLAS_TYPES)}
                out, in1, in2, cdims = expand_dimensions(Val(5), out, in1, in2, cdims)
                $(bname)(out, in1, in2, cdims; kwargs...)
                return out
            end
        end
    end

    # _direct() can take in anything, but it can be much slower, so it's best
    # to take advantage of the accelerated definitions above.  We still do the
    # expansion of dimensions here, to make sure this works for conv3d cases.
    if :direct in backends
        fname = Symbol("$(front_name)!")
        bname = Symbol("$(front_name)_direct!")
        @eval begin
            function $(fname)(out::AA, in1::AA,
                              in2::AA, cdims::ConvDims;
                              kwargs...)
                @debug string("Slow fallback implementation invoked for ", $front_name, "!  ",
                              "You probably don't want this; check your datatypes.")
                $(bname)(out, in1, in2, cdims; kwargs...)
                return out
            end

            # We add here some "expanders" that convert conv1d/2d inputs up to
            # the conv3d input shape that our backend is expecting:
            function $(bname)(out::AA{<:Any,3}, in1::AA{<:Any,3}, in2::AA{<:Any,3},
                              cdims::ConvDims; kwargs...)
                out, in1, in2, cdims = expand_dimensions(Val(5), out, in1, in2, cdims)
                $(bname)(out, in1, in2, cdims; kwargs...)
                return out
            end
            function $(bname)(out::AA{<:Any,4}, in1::AA{<:Any,4}, in2::AA{<:Any,4},
                              cdims::ConvDims; kwargs...)
                out, in1, in2, cdims = expand_dimensions(Val(5), out, in1, in2, cdims)
                $(bname)(out, in1, in2, cdims; kwargs...)
                return out
            end
        end
    end
end

"""
    expand_dimensions(M, x, args...)

Inserts singleton dimensions into `x`, and any further arguments until they
are `M`-dimensional.  It is an error for the input tensors to be of greater
dimensionality than `M`.
"""
function expand_dimensions(::Val{M}, x::AbstractArray{<:Any, N}, args...) where {N, M}
    if M == N
        return (x, args...)
    end
    if N > M
        error("Cannot expand_dimensions() to a smaller dimensionality!")
    end
    return (
        insert_singleton_spatial_dimension(x, M - N),
        insert_singleton_spatial_dimension.(args, M - N)...,
    )
end

# Finally, let's generate auto-allocating versions of all our functions.
# These are the ones that don't have the `!` at the end.  Note that we do not
# type-specialize these like the non-allocating versions.
for backend in (Symbol(), :_direct, :_im2col, :_nnpack)
    # First, conv() forward passes
    for name in (:conv, :depthwiseconv)
        fname = Symbol("$(name)$(backend)")
        bname = Symbol("$(name)$(backend)!")
        @eval begin
            function $(fname)(x::AA, w::AA, cdims::ConvDims; kwargs...)
                y = similar(x,
                    promote_type(eltype(x), eltype(w)),
                    output_size(cdims)...,
                    channels_out(cdims),
                    size(x, ndims(x)),
                )
                $(bname)(y, x, w, cdims; kwargs...)
                return y
            end
        end
    end

    # Next, backward passes for `_data()`
    for name in (:∇conv_data, :∇depthwiseconv_data)
        fname = Symbol("$(name)$(backend)")
        bname = Symbol("$(name)$(backend)!")
        @eval begin
            function $(fname)(dy::AA, w::AA, cdims::ConvDims; kwargs...)
                dx = similar(dy,
                    input_size(cdims)...,
                    channels_in(cdims),
                    size(dy, ndims(dy)),
                )
                $(bname)(dx, dy, w, cdims; kwargs...)
                return dx
            end
        end
    end

    # We do the filter backprops separately, as the shape calculation for `w`
    # is slightly different for depthwise than for normal dense convolution.
    fname = Symbol("∇conv_filter$(backend)")
    bname = Symbol("∇conv_filter$(backend)!")
    @eval begin
        function $(fname)(x::AA, dy::AA, cdims::ConvDims; kwargs...)
            dw = similar(dy,
                kernel_size(cdims)...,
                channels_in(cdims),
                channels_out(cdims),
            )
            $(bname)(dw, x, dy, cdims; kwargs...)
            return dw
        end
    end

    fname = Symbol("∇depthwiseconv_filter$(backend)")
    bname = Symbol("∇depthwiseconv_filter$(backend)!")
    @eval begin
        function $(fname)(x::AA, dy::AA, cdims::ConvDims; kwargs...)
            dw = similar(dy,
                kernel_size(cdims)...,
                channel_multiplier(cdims),
                channels_in(cdims),
            )
            $(bname)(dw, x, dy, cdims; kwargs...)
            return dw
        end
    end
end

## Pooling
for (front_name, backends) in pooling_backends
    if :nnpack in backends
        nnpack_pdims = PoolDims{
            # only conv2d, no conv1d or conv3d
            2,
            # Any kernel size, channels in or out, whatever, it's good.
            K,
            # No stride
            S,
            # Padding can be whatevs
            P,
            # No dilation
            (1, 1),
        } where {K, S, P}
        # Map from the front name to the back name
        fname = Symbol("$(front_name)!")
        @eval begin
            function $(fname)(out::AA{Float32, 4}, in::AA{Float32, 4},
                              pdims::$(nnpack_pdims); kwargs...)
                # Check to see if this is a supported operation
                if nnpack_supported_pooling(pdims)
                    $(Symbol("$(front_name)_nnpack!"))(out, in, pdims; kwargs...)
                else
                    # If it's not suported, then bail out to _direct()
                    $(Symbol("$(front_name)_direct!"))(out, in, pdims; kwargs...)
                end
                return out
            end
        end
    end

    if :direct in backends
        fname = Symbol("$(front_name)!")
        bname = Symbol("$(front_name)_direct!")
        @eval begin
            function $(fname)(out::AA, in::AA, pdims::PoolDims; kwargs...)
                $(bname)(out, in, pdims; kwargs...)
                return out
            end

            # Add reshapers
            function $(bname)(out::AA{T,3}, in::AA{T,3},
                              pdims::PoolDims; kwargs...) where {T}
                outx, inx, pdimsx = expand_dimensions(Val(5), out, in, pdims)
                $(bname)(outx, inx, pdimsx; kwargs...)
                return out
            end
            function $(bname)(out::AA{T,4}, in::AA{T,4},
                              pdims::PoolDims; kwargs...) where {T}
                outx, inx, pdimsx = expand_dimensions(Val(5), out, in, pdims)
                $(bname)(outx, inx, pdimsx; kwargs...)
                return out
            end
        end
    end
end

# We only have direct backprop for pooling
for (front_name, backend) in (
        :∇maxpool  => :direct,
        :∇meanpool => :direct,
    )
    fname = Symbol("$(front_name)!")
    bname = Symbol("$(front_name)_direct!")
    @eval begin
        function $(fname)(dx::AA{T}, dy::AA{T},
                          y::AA{T}, x::AA{T},
                          pdims::PoolDims; kwargs...) where {T}
            $(bname)(dx, dy, y, x, pdims; kwargs...)
            return dx
        end
        function $(bname)(dx::AA{T,3}, dy::AA{T,3},
                          y::AA{T,3}, x::AA{T,3},
                          pdims::PoolDims; kwargs...) where {T}
            dxx, dyx, yx, xx, pdimsx = expand_dimensions(Val(5), dx, dy, y, x, pdims)
            $(bname)(dxx, dyx, yx, xx, pdimsx; kwargs...)
            return dx
        end
        function $(bname)(dx::AA{T,4}, dy::AA{T,4},
                          y::AA{T,4}, x::AA{T,4},
                          pdims::PoolDims; kwargs...) where {T}
            dxx, dyx, yx, xx, pdimsx = expand_dimensions(Val(5), dx, dy, y, x, pdims)
            $(bname)(dxx, dyx, yx, xx, pdimsx; kwargs...)
            return dx
        end
    end
end

# Finally, let's generate auto-allocating versions of all our functions, for all backends:
for backend in (Symbol(), :_direct),
    name in (:maxpool, :meanpool)

    fname = Symbol("$(name)$(backend)")
    bname = Symbol("$(name)$(backend)!")
    f_backname = Symbol("∇$(name)$(backend)")
    b_backname = Symbol("∇$(name)$(backend)!")
    @eval begin
        function $(fname)(x::AA, pdims::PoolDims; kwargs...)
            y = similar(x, output_size(pdims)..., channels_out(pdims), size(x, ndims(x)))
            fill!(y, zero(eltype(x)))
            $(bname)(y, x, pdims; kwargs...)
            return y
        end

        # Backprops too
        function $(f_backname)(dy::AA, y::AA, x::AA, pdims::PoolDims; kwargs...)
            dx = similar(x, input_size(pdims)..., channels_in(pdims), size(dy, ndims(dy)))
            fill!(dx, zero(eltype(x)))
            $(b_backname)(dx, dy, y, x, pdims; kwargs...)
            return dx
        end
    end
end

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

# Simplified conv() adapters that construct the `DenseConvDims` for you.
function conv(x, w::AbstractArray{T, N}; stride = 1, pad = 0, dilation = 1, flipped = false) where {T, N}
    stride = expand(Val(N-2), stride)
    pad = expand(Val(N-2), pad)
    dilation = expand(Val(N-2), dilation)
    cdims = DenseConvDims(x, w; stride = stride, padding = pad, dilation = dilation, flipkernel = flipped)
    return conv(x, w, cdims)
end

function depthwiseconv(x, w::AbstractArray{T, N}; stride = 1, pad = 0, dilation = 1, flipped = false) where {T, N}
    stride = expand(Val(N-2), stride)
    pad = expand(Val(N-2), pad)
    dilation = expand(Val(N-2), dilation)
    cdims = DepthwiseConvDims(x, w; stride = stride, padding = pad, dilation = dilation, flipkernel = flipped)
    return depthwiseconv(x, w, cdims)
end

function maxpool(x, k::NTuple{N, Integer}; pad = 0, stride = k) where N
    pad = expand(Val(N), pad)
    stride = expand(Val(N), stride)
    pdims = PoolDims(x, k; padding = pad, stride = stride)
    return maxpool(x, pdims)
end

function meanpool(x, k::NTuple{N, Integer}; pad = 0, stride = k) where N
    pad = expand(Val(N), pad)
    stride = expand(Val(N), stride)
    pdims = PoolDims(x, k; padding = pad, stride = stride)
    return meanpool(x, pdims)
end
