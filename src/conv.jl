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

#   The computational flow, starting from the user facing functions,
#   goes through the following steps:
#
#   STEP 1:
#       use ConvDims objects (only for `conv` and `depthwiseconv`)
#   STEP 2:
#        define autoallocating version (frontend and implementations)
#   STEP 3:
#        reshape to 3d convolutions (frontend and implementions)
#   STEP 4:
#        choose implementation

# TODO: should we also add
#   STEP X:
#        use homogeneus datatypes
# to handle etherogeneus inputs now handled by conv_direct?


########## STEP 1 ############
"""
    conv(x, w; stride = 1, pad = 0, dilation = 1, flipped = false, groups = 1)

Apply convolution filter `w` to input `x`. `x` and `w` are 3d/4d/5d tensors
in 1d/2d/3d convolutions respectively. They may have real or complex element types.

The dimensions are arranged as follows, with the spatial dimensions first:
- `x`: `(spatial..., channels_in, batch)`
- `w`: `(spatial..., channels_in ÷ groups, channels_out)`
- output: `(spatial_out..., channels_out, batch)`

The output spatial size for each dimension is
`(spatial + 2*pad - (dilation*(spatial_w - 1) + 1)) ÷ stride + 1`.

# Keyword arguments
- `stride`: stride of the filter (`Int` or tuple of one `Int` per spatial dimension).
- `pad`: zero padding. An `Int` or tuple applies symmetric padding; a tuple of
  `2*ndims` entries gives `(left, right)` padding per spatial dimension.
- `dilation`: spacing between filter elements (atrous convolution).
- `flipped`: if `false` (default), perform cross-correlation (the usual deep-learning
  convention); if `true`, perform a true mathematical convolution by flipping `w`.
- `groups`: number of groups for grouped convolution.

# Examples
```jldoctest
julia> x = rand(Float32, 28, 28, 3, 16);  # (width, height, channels_in, batch)

julia> w = rand(Float32, 5, 5, 3, 8);     # (width, height, channels_in, channels_out)

julia> size(conv(x, w))                    # 28 - 5 + 1 = 24
(24, 24, 8, 16)

julia> size(conv(x, w; stride=2, pad=1))
(13, 13, 8, 16)
```
"""
function conv(x, w::AbstractArray{T, N}; stride = 1, pad = 0, dilation = 1, flipped = false, groups = 1) where {T, N}
    stride = expand(Val(N - 2), stride)
    padding = expand(Val(N - 2), pad)
    dilation = expand(Val(N - 2), dilation)
    cdims = DenseConvDims(
        size(x), size(w); stride, padding, dilation, flipkernel=flipped, groups)
    return conv(x, w, cdims)
end

"""
    depthwiseconv(x, w; stride=1, pad=0, dilation=1, flipped=false)

Depthwise convolution operation with filter `w` on input `x`. `x` and `w`
are 3d/4d/5d tensors in 1d/2d/3d convolutions respectively. The filter `w`
has shape `(spatial..., channel_multiplier, channels_in)`.

This is equivalent to a grouped [`conv`](@ref) with `groups = channels_in`, and is
implemented in terms of it.
"""
function depthwiseconv(x, w::AbstractArray{T, N}; stride=1, pad=0, dilation=1, flipped=false) where {T, N}
    stride = expand(Val(N-2), stride)
    pad = expand(Val(N-2), pad)
    dilation = expand(Val(N-2), dilation)
    cdims = DepthwiseConvDims(x, w; stride=stride, padding=pad, dilation=dilation, flipkernel=flipped)
    return depthwiseconv(x, w, cdims)
end
##############################


########### STEP 2 ###################
# Let's generate auto-allocating versions of all our functions, for all backends.
# We `@timeit` these methods separately, as we want to know how much time is spent in
# allocation.  :P
for backend in (Symbol(), :_direct, :_im2col)
    # First make auto-allocating versions of the conv()-like calls:
    @eval begin
        function $(Symbol("conv$(backend)"))(
                        x::AbstractArray{xT,N}, w::AbstractArray{wT,N},
                        cdims::ConvDims; kwargs...) where {xT, wT, N}
            y = similar(x, promote_type(xT, wT), output_size(cdims)...,
                           channels_out(cdims), size(x,N))
            return $(Symbol("conv$(backend)!"))(y, x, w, cdims; kwargs...)
        end
    end

    @eval begin
        function $(Symbol("∇conv_data$(backend)"))(
                        dy::AbstractArray{yT,N}, w::AbstractArray{wT,N},
                        cdims::C; kwargs...) where {yT, wT, N, C <: ConvDims}
            dx = similar(dy, input_size(cdims)..., channels_in(cdims), size(dy, N))
            return $(Symbol("∇conv_data$(backend)!"))(dx, dy, w, cdims; kwargs...)
        end
    end

    @eval begin
        function $(Symbol("∇conv_filter$(backend)"))(
                        x::AbstractArray{xT,N}, dy::AbstractArray{yT,N},
                        cdims::ConvDims; kwargs...) where {xT, yT, N}
            dw = similar(dy, kernel_size(cdims)..., channels_in(cdims) ÷ groupcount(cdims),
                                                    channels_out(cdims))
            return $(Symbol("∇conv_filter$(backend)!"))(dw, x, dy, cdims; kwargs...)
        end
    end
end
##########################################


########## STEP 3 ############

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

#######################################


########### STEP 4 ############

# First, we will define mappings from the generic API names to our accelerated backend
# implementations. For homogeneous-datatype 1, 2 and 3d convolutions, we default to using
# im2col + GEMM.
# But we always support a fallback, non-accelerated path, where we use the direct, but
# slow, implementations. These should not typically be used, hence the `@warn`,

# These are the GEMM types we will accelerate with `im2col`
const G = Union{[x[2] for x in gemm_datatype_mappings]...}

for (front_name, backend, signature) in (
    # This maps from public, front-facing name, to internal backend name, given the function signature and the where clause
    # (frontend, backend, (out Array signature, in1 Array signature, in2 Array signature, (parametric Types)))
    (:conv, :im2col, ((:T, 5), (:T, 5), (:T, 5), :C, (:(T <: G), :(C <: ConvDims)))),
    (:conv, :direct, ((:yT, :N), (:T1, :N), (:T2, :N), :C, (:yT, :T1, :T2, :N, :(C <: ConvDims)))),
)
    # We only define 3d conv primitives, we reshape lower down to get 1d and 2d convolution
    @eval begin

        function $(Symbol("$(front_name)!"))(
                        out::AbstractArray{$(signature[1][1]), $(signature[1][2])},
                        in1::AbstractArray{$(signature[2][1]), $(signature[1][2])},
                        in2::AbstractArray{$(signature[3][1]), $(signature[1][2])},
                        cdims::$(signature[4]);
                        kwargs...) where {$(signature[5]...)}
            if $(string(backend)) == "direct" && yT == Float64  # warn for Float32 + accidental Float64, but don't print warning for ForwardDiff.Dual
                @warn string("Slow fallback implementation invoked for ", $(string(front_name)), "!  ",
                        "You probably don't want this; check your datatypes.") yT T1 T2 maxlog=1
            end

            x_cs = Iterators.partition(1:size(in1, 4),
                                    channels_in(cdims) ÷ groupcount(cdims))
            w_cs = Iterators.partition(1:size(in2, 5),
                                    channels_out(cdims) ÷ groupcount(cdims))
            cdims2 = basetype(C)(cdims,
                                G = 1,
                                C_in = channels_in(cdims) ÷ groupcount(cdims),
                                C_out = channels_out(cdims) ÷ groupcount(cdims))

            function conv_group(xc, wc)
                x = @view in1[ntuple(i -> i == 4 ? xc : Colon(), 5)...]
                w = @view in2[ntuple(i -> i == 5 ? wc : Colon(), 5)...]
                y = @view out[ntuple(i -> i == 4 ? wc : Colon(), 5)...]
                $(Symbol("$(front_name)_$(backend)!"))(y, x, w, cdims2; kwargs...)
            end

            if should_use_spawn() && length(x_cs) > 1
                Threads.@sync for (xc, wc) in zip(x_cs, w_cs)
                    Threads.@spawn conv_group(xc, wc)
                end
            else
                for (xc, wc) in zip(x_cs, w_cs)
                    conv_group(xc, wc)
                end
            end

            return out
        end
    end
end

# im2col-accelerated function forwarding definition
for (front_name, backend, signature) in (
    # This maps from public, front-facing name, to internal backend name, given the function signature and the where clause
    # (frontend, backend, (out Array signature, in1 Array signature, in2 Array signature, (parametric Types)))
    (:∇conv_data, :im2col, ((:T, 5), (:T, 5), (:T, 5), :C, (:(T <: G), :(C <: ConvDims)))),
    (:∇conv_data, :direct, ((:yT, :N), (:T1, :N), (:T2, :N), :C, (:yT, :T1, :T2, :N, :(C <: ConvDims)))),
)
    # We only define 3d conv primitives, we reshape lower down to get 1d and 2d convolution
    @eval begin
        function $(Symbol("$(front_name)!"))(
                        out::AbstractArray{$(signature[1][1]), $(signature[1][2])},
                        in1::AbstractArray{$(signature[2][1]), $(signature[1][2])},
                        in2::AbstractArray{$(signature[3][1]), $(signature[1][2])},
                        cdims::$(signature[4]);
                        kwargs...) where {$(signature[5]...)}
            if $(string(backend)) == "direct" && yT == Float64  # warn for Float32 + accidental Float64, but don't print warning for ForwardDiff.Dual
                @warn string("Slow fallback implementation invoked for ", $(string(front_name)), "!  ",
                        "You probably don't want this; check your datatypes.") yT T1 T2 maxlog=1
            end


            dx_cs = Iterators.partition(1:size(out, 4),
                                        channels_in(cdims) ÷ groupcount(cdims))
            w_cs = Iterators.partition(1:size(in2, 5),
                                    channels_out(cdims) ÷ groupcount(cdims))
            dy_cs = Iterators.partition(1:size(in1, 4),
                                        channels_out(cdims) ÷ groupcount(cdims))
            cdims2 = basetype(C)(cdims,
                                G = 1,
                                C_in = channels_in(cdims) ÷ groupcount(cdims),
                                C_out = channels_out(cdims) ÷ groupcount(cdims))

            function ∇conv_data_group(xc, yc, wc)
                dxv = @view out[ntuple(i -> i == 4 ? xc : Colon(), 5)...]
                dyv = @view in1[ntuple(i -> i == 4 ? yc : Colon(), 5)...]
                wv = @view in2[ntuple(i -> i == 5  ? wc : Colon(), 5)...]
                $(Symbol("$(front_name)_$(backend)!"))(dxv, dyv, wv, cdims2; kwargs...)
            end

            if should_use_spawn() && length(dx_cs) > 1
                Threads.@sync for (xc, yc, wc) in zip(dx_cs, dy_cs, w_cs)
                    Threads.@spawn ∇conv_data_group(xc, yc, wc)
                end
            else
                for (xc, yc, wc) in zip(dx_cs, dy_cs, w_cs)
                    ∇conv_data_group(xc, yc, wc)
                end
            end

            return out
        end
    end
end

for (front_name, backend, signature) in (
    # This maps from public, front-facing name, to internal backend name, given the function signature and the where clause
    # (frontend, backend, (out Array signature, in1 Array signature, in2 Array signature, (parametric Types)))
    (:∇conv_filter, :im2col, ((:T, 5), (:T, 5), (:T, 5), :C, (:(T <: G), :(C <: ConvDims)))),
    (:∇conv_filter, :direct, ((:yT, :N), (:T1, :N), (:T2, :N), :C, (:yT, :T1, :T2, :N, :(C <: ConvDims)))),
)
    # We only define 3d conv primitives, we reshape lower down to get 1d and 2d convolution
    @eval begin
        function $(Symbol("$(front_name)!"))(
                        out::AbstractArray{$(signature[1][1]), $(signature[1][2])},
                        in1::AbstractArray{$(signature[2][1]), $(signature[1][2])},
                        in2::AbstractArray{$(signature[3][1]), $(signature[1][2])},
                        cdims::$(signature[4]);
                        kwargs...) where {$(signature[5]...)}
            if $(string(backend)) == "direct" && yT == Float64  # warn for Float32 + accidental Float64, but don't print warning for ForwardDiff.Dual
                @warn string("Slow fallback implementation invoked for ", $(string(front_name)), "!  ",
                        "You probably don't want this; check your datatypes.") yT T1 T2 maxlog=1
            end

            dw_cs = Iterators.partition(1:size(out, 5),
                                        channels_out(cdims) ÷ groupcount(cdims))
            dy_cs = Iterators.partition(1:size(in2, 4),
                                        channels_out(cdims) ÷ groupcount(cdims))
            x_cs = Iterators.partition(1:size(in1, 4),
                                    channels_in(cdims) ÷ groupcount(cdims))
            cdims2 = basetype(C)(cdims,
                                G = 1,
                                C_in = channels_in(cdims) ÷ groupcount(cdims),
                                C_out = channels_out(cdims) ÷ groupcount(cdims))

            function ∇conv_filter_group(wc, xc, yc)
                x = @view in1[ntuple(i -> i == 4 ? xc : Colon(), 5)...]
                dy = @view in2[ntuple(i -> i == 4 ? yc : Colon(), 5)...]
                dw = @view out[ntuple(i -> i == 5 ? wc : Colon(), 5)...]
                $(Symbol("$(front_name)_$(backend)!"))(dw, x, dy, cdims2; kwargs...)
            end

            if should_use_spawn() && length(dw_cs) > 1
                Threads.@sync for (wc, xc, yc) in zip(dw_cs, x_cs, dy_cs)
                    Threads.@spawn ∇conv_filter_group(wc, xc, yc)
                end
            else
                for (wc, xc, yc) in zip(dw_cs, x_cs, dy_cs)
                    ∇conv_filter_group(wc, xc, yc)
                end
            end

            return out
        end
    end
end


###########  DEPTHWISE CONVOLUTION  ###########
#
# A depthwise convolution is exactly a grouped convolution with `groups == channels_in`.
# Rather than maintaining a separate set of specialized kernels (which benchmarked slower
# than the generic grouped-convolution path, see FluxML/NNlib.jl#402), we route every
# depthwise entry point through the dense grouped-convolution implementation.
#
# The only bookkeeping is the weight layout. A depthwise weight has shape
# `(spatial..., channel_multiplier, channels_in)`; the equivalent grouped weight has shape
# `(spatial..., 1, channel_multiplier * channels_in)`. Because Julia arrays are
# column-major, the depthwise `(m, i)` channel pair flattens to grouped output channel
# `(i - 1) * channel_multiplier + m`, which is exactly the channel order a grouped
# convolution expects -- so the conversion is a zero-copy `reshape`.

# Equivalent grouped `DenseConvDims` for a `DepthwiseConvDims`.
function DenseConvDims(cdims::DepthwiseConvDims)
    return DenseConvDims(
        input_size(cdims), kernel_size(cdims),
        channels_in(cdims), channels_out(cdims), channels_in(cdims),
        stride(cdims), padding(cdims), dilation(cdims), flipkernel(cdims))
end

# Reshape a depthwise weight `(spatial..., C_mult, C_in)` to the grouped layout
# `(spatial..., 1, C_mult * C_in)`. Zero-copy.
@inline _grouped_weight(w::AbstractArray{T,N}) where {T,N} =
    reshape(w, ntuple(i -> size(w, i), N - 2)..., 1, size(w, N - 1) * size(w, N))

function depthwiseconv(x, w, cdims::DepthwiseConvDims; kwargs...)
    return conv(x, _grouped_weight(w), DenseConvDims(cdims); kwargs...)
end

function depthwiseconv!(y, x, w, cdims::DepthwiseConvDims; kwargs...)
    conv!(y, x, _grouped_weight(w), DenseConvDims(cdims); kwargs...)
    return y
end

function ∇depthwiseconv_data(dy, w, cdims::DepthwiseConvDims; kwargs...)
    return ∇conv_data(dy, _grouped_weight(w), DenseConvDims(cdims); kwargs...)
end

function ∇depthwiseconv_data!(dx, dy, w, cdims::DepthwiseConvDims; kwargs...)
    ∇conv_data!(dx, dy, _grouped_weight(w), DenseConvDims(cdims); kwargs...)
    return dx
end

function ∇depthwiseconv_filter(x, dy, cdims::DepthwiseConvDims; kwargs...)
    dw = ∇conv_filter(x, dy, DenseConvDims(cdims); kwargs...)
    return reshape(dw, kernel_size(cdims)..., channel_multiplier(cdims), channels_in(cdims))
end

function ∇depthwiseconv_filter!(dw, x, dy, cdims::DepthwiseConvDims; kwargs...)
    dwg = reshape(dw, kernel_size(cdims)..., 1, channels_out(cdims))
    ∇conv_filter!(dwg, x, dy, DenseConvDims(cdims); kwargs...)
    return dw
end

for Dims in [:DenseConvDims, :DepthwiseConvDims, :PoolDims]
    @eval @non_differentiable $Dims(::Any...)
end

colmajor(x) = (is_strided(x) && Base.stride(x, 1) == 1) ? x : collect(x)

for conv in [:conv, :depthwiseconv]
    local ∇conv_data, ∇conv_filter = Symbol.(:∇, conv, [:_data, :_filter])
    conv_pullback, ∇conv_data_pullback = Symbol.([conv, ∇conv_data], :_pullback)

    @eval function rrule(::typeof($conv), x, w, cdims; kw...)
        function $conv_pullback(Δraw)
            Δ = colmajor(unthunk(Δraw))
            return (
                NoTangent(),
                @thunk($∇conv_data(Δ, w, cdims, kw...)),
                @thunk($∇conv_filter(x, Δ, cdims, kw...)),
                NoTangent(),
            )
        end
        return $conv(x, w, cdims; kw...), $conv_pullback
    end

    @eval function rrule(::typeof($∇conv_data), x, w, cdims; kw...)
        function $∇conv_data_pullback(Δraw)
            Δ = colmajor(unthunk(Δraw))
            return (
                NoTangent(),
                @thunk($conv(Δ, w, cdims, kw...)),
                @thunk($∇conv_filter(Δ, x, cdims, kw...)),
                NoTangent(),
            )
        end
        return $∇conv_data(x, w, cdims; kw...), $∇conv_data_pullback
    end
end

function rrule(::typeof(∇conv_filter), x, dy, cdims; kw...)
    function ∇conv_filter_pullback(Δ)
        Δ1 = colmajor(unthunk(Δ))
        return (
            NoTangent(),
            @thunk(∇conv_data(dy, Δ1, cdims, kw...)),
            @thunk(conv(x, Δ1, cdims, kw...)),
            NoTangent(),
        )
    end
    return ∇conv_filter(x, dy, cdims; kw...), ∇conv_filter_pullback
end
