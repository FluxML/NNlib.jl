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
#     - lpnormpool(x, pdims)
#     - lpnormpool!(y, x, pdims)
#   - Pooling input backprop
#     - ∇maxpool(dy, y, x, pdims)
#     - ∇maxpool!(dx, dy, y, x, pdims)
#     - ∇meanpool(dy, y, x, pdims)
#     - ∇meanpool!(dx, dy, y, x pdims)
#     - ∇lpnormpool(dy, y, x, pdims)
#     - ∇lpnormpool!(dx, dy, y, x pdims)
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
        :lpnormpool => :direct,
    )

    # We only define 3d pooling primitives, we reshape lower down to get 1d and 2d pooling
    @eval begin
        function $(Symbol("$(front_name)!"))(
                y::AbstractArray{<:Any,5}, x::AbstractArray{<:Any,5},
                pdims::PoolDims; kwargs...)
            $(Symbol("$(front_name)_$(backend)!"))(y, x, pdims; kwargs...)
        end
    end
end

# Do the same for backprops
for (front_name, backend) in (
        :∇maxpool  => :direct,
        :∇meanpool => :direct,
        :∇lpnormpool => :direct,
    )
    @eval begin
        function $(Symbol("$(front_name)!"))(
                        dx::AbstractArray{<:Any,5}, dy::AbstractArray{<:Any,5},
                        y::AbstractArray{<:Any,5}, x::AbstractArray{<:Any,5},
                        pdims::PoolDims; kwargs...)
            $(Symbol("$(front_name)_$(backend)!"))(dx, dy, y, x, pdims; kwargs...)
        end
    end
end


# Our strategy for pooling is to reshape to an array with three spatial dimensions, which
# makes things MUCH EASIER for us on the backend side, and is in general pretty fast,
# since we can specialize on sizes.
for front_name in (:maxpool, :meanpool, :lpnormpool)
    for backend in (Symbol(), :_direct)
        for N in (3, 4)
            @eval begin
                function $(Symbol("$(front_name)$(backend)!"))(
                                y::AbstractArray{<:Any,$N}, x::AbstractArray{<:Any,$N},
                                pdims::PoolDims; kwargs...)
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
                                dx::AbstractArray{<:Any,$N}, dy::AbstractArray{<:Any,$N},
                                y::AbstractArray{<:Any,$N}, x::AbstractArray{<:Any,$N},
                                pdims::PoolDims; kwargs...)
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
for backend in (Symbol(), :_direct)
    # First make auto-allocating versions of the basic pooling calls:
    for name in (:maxpool, :meanpool, :lpnormpool)
        @eval begin
            function $(Symbol("$(name)$(backend)"))(
                            x::AbstractArray{<:Any,N},
                            pdims::PoolDims; kwargs...) where {N}
                y = similar(x, output_size(pdims)..., channels_out(pdims), size(x, N))
                fill!(y, 0)
                return $(Symbol("$(name)$(backend)!"))(y, x, pdims; kwargs...)
            end

            # Backprops too
            function $(Symbol("∇$(name)$(backend)"))(
                            dy::AbstractArray{<:Any,N}, y::AbstractArray{<:Any,N},
                            x::AbstractArray{<:Any,N}, pdims::PoolDims;
                            kwargs...) where {N}
                dx = similar(x, input_size(pdims)..., channels_in(pdims), size(dy, N))
                fill!(dx, 0)
                return $(Symbol("∇$(name)$(backend)!"))(dx, dy, y, x, pdims; kwargs...)
            end
        end
    end
end

# Complex mean pooling (https://github.com/FluxML/NNlib.jl/issues/610).
#
# Mean pooling is linear, so it extends to complex inputs by pooling the real and
# imaginary parts independently and recombining. Doing this once here, in terms of
# the real-valued `meanpool`/`∇meanpool`, gives complex support to every backend
# that pools real arrays (CPU, cuDNN, MIOpen, ...) for free — no per-backend method.
# Differentiation flows through the same split: the `meanpool` `rrule` below is
# restricted to real `x`, so for complex `x` the AD engine differentiates the two
# real `meanpool` calls (each picking up its backend's own rule) and the `complex.`
# recombination.
#
# `maxpool`/`lpnormpool` are not extended: `max` and the p-norm are undefined for
# complex numbers, so they keep erroring (as they already do on the CPU path).
function meanpool(x::AbstractArray{<:Complex,N}, pdims::PoolDims; kwargs...) where {N}
    return complex.(meanpool(real(x), pdims; kwargs...),
                    meanpool(imag(x), pdims; kwargs...))
end

function ∇meanpool(dy::AbstractArray{<:Complex,N}, y::AbstractArray{<:Complex,N},
                   x::AbstractArray{<:Complex,N}, pdims::PoolDims; kwargs...) where {N}
    return complex.(∇meanpool(real(dy), real(y), real(x), pdims; kwargs...),
                    ∇meanpool(imag(dy), imag(y), imag(x), pdims; kwargs...))
end

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)


"""
    maxpool(x, k::NTuple{N, Integer}; pad=0, stride=k)

Perform max pool operation with window size `k` on input tensor `x`.

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`
* `pad`: See [`pad_zeros`](@ref) for details.
* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.
"""
function maxpool(x, k::NTuple{N, Integer}; pad=0, stride=k) where N
    pad = expand(Val(N), pad)
    stride = expand(Val(N), stride)
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return maxpool(x, pdims)
end


"""
    meanpool(x, k::NTuple{N, Integer}; pad=0, stride=k, count_include_pad=true)

Perform mean pool operation with window size `k` on input tensor `x`.

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`
* `pad`: See [`pad_zeros`](@ref) for details.
* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.
* `count_include_pad`: When `true` (the default), padding (zero) elements are included in the
  averaging denominator, so each output element divides by the full window size `prod(k)`.
  When `false`, only the in-bounds (non-padding) elements are counted, matching cuDNN's
  `CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING` and PyTorch's `count_include_pad=false`.
  Has no effect when `pad == 0`.
"""
function meanpool(x, k::NTuple{N, Integer}; pad=0, stride=k, count_include_pad=true) where N
    pad = expand(Val(N), pad)
    stride = expand(Val(N), stride)
    pdims = PoolDims(x, k; padding=pad, stride=stride)
    return meanpool(x, pdims; count_include_pad)
end


"""
    lpnormpool(x, p::Real, k::NTuple{N, Integer}; pad=0, stride=k)

Perform Lp pool operation with value of the Lp norm `p` and window size `k` on input tensor `x`, also known as LPPool in pytorch.
This pooling operator from [Learned-Norm Pooling for Deep Feedforward and Recurrent Neural Networks](https://arxiv.org/abs/1311.1780).

Arguments:

* `x` and `k`: Expects `ndim(x) ∈ 3:5`, and always `length(k) == ndim(x) - 2`
* `p` is restricted to `0 < p < Inf`.
* `pad`: See [`pad_zeros`](@ref) for details.
* `stride`: Either a tuple with the same length as `k`, or one integer for all directions. Default is `k`.

For all elements `x` in a size `k` window, lpnormpool computes `(∑ᵢ xᵢ^p)^(1 / p)` as an element of the output.

Thus `lpnormpool(x, 1, k) ./ prod(k) ≈ meanpool(x, k)` and `lpnormpool(x, 2, k).^2 ./ prod(k) ≈ meanpool(x.^2, k)`.
"""
function lpnormpool(x, p::Real, k::NTuple{N, Integer}; pad=0, stride=k) where {N}
    pow = p isa Integer ? p : convert(float(eltype(x)), p)
    (isinf(pow) || pow < 0) && error("p value of Lp norm pool expects `0 < p < Inf`, but p is $(pow) now.")
    pdims = PoolDims(x, k; padding=expand(Val(N), pad), stride=expand(Val(N), stride))
    return lpnormpool(x, pdims; p=pow)
end


# Restricted to real `x`: complex `meanpool` has no direct rule and is instead
# differentiated through its real/imaginary split (see the complex `meanpool`
# above), while complex `maxpool`/`lpnormpool` are unsupported.
for pool in [:maxpool, :meanpool, :lpnormpool]
    ∇pool = Symbol(:∇, pool)
    pullback = Symbol(pool, :_pullback)
    @eval function rrule(::typeof($pool), x::AbstractArray{<:Real}, pdims::PoolDims; kw...)
        Ω = $pool(x, pdims; kw...)
        $pullback(Δ) = (NoTangent(), $∇pool(unthunk(Δ), Ω, x, pdims; kw...), NoTangent())
        return Ω, $pullback
    end
end


function topk(x::AbstractArray{T,N}, k; rev=false, dims=nothing) where {T,N}
    if dims === nothing
        y = vec(x)
        perm = partialsortperm(y, 1:k; rev)
        return y[perm], linear_to_cartesian(x, perm)
    else
        @assert dims isa Int
        sz1 = size(x)[1:dims-1]
        sz2 = size(x)[dims+1:end]
        slice1 = CartesianIndices(sz1)
        slice2 = CartesianIndices(sz2)
        perm = similar(x, Int, (sz1..., k, sz2...))
        y = similar(x, (sz1..., k, sz2...))
        for I1 in slice1
            for I2 in slice2
                xI = x[I1,:,I2]
                permI = partialsortperm(x[I1,:,I2], 1:k; rev)
                perm[I1,:,I2] .= permI
                y[I1,:,I2] .= xI[permI]
            end
        end
        return y, perm
    end
end

function linear_to_cartesian(x, i)
    CartesianIndices(x)[i]
end
