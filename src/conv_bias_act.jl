export conv_bias_act, conv_bias_act!

"""
    conv_bias_act(x, w, cdims, b, σ; kw...)

This should be equivalent to `σ.(conv(x, w, cdims) .+ reshape(b, ...))`, 
but faster because it will:
* try hard to re-use memory, when safe to do so,
* call fused CUDNN operation, when possible?

Keyword arguments are passed to the CUDA version.

TODO:
* conv_bias_act! is overloaded here:
https://github.com/FluxML/NNlibCUDA.jl/blob/master/src/cudnn/conv.jl#L37
but fusion of σ other than identity, relu, needs more thought...
should probably go to `bias_act!(σ, conv_bias_act(identity, x, w, ...))`



cudnnConvolutionForward! only supports relu and identity. 
So let's only call conv_bias_act! with those, ever,
even though it has some fallback code.




* slightly weird signature, worth changing?
* simplify so that conv! does not need a gradient?
"""
function conv_bias_act(x::AbstractArray{xT,N}, w::AbstractArray{wT,N}, 
                cdims::ConvDims, b, σ; kwargs...) where {xT, wT, N}

    y = similar(x, promote_type(xT, wT), output_size(cdims)..., channels_out(cdims), size(x,N))
    if σ === identity || σ === relu
        # On the GPU, cudnnConvolutionForward! is fast but only supports relu and identity.
        # Best path forwards:
        return conv_bias_act!(y, x, w, cdims, b, σ; kwargs...)
    else
        # In other cases, still want to call it, but then handle the activation ourselves:
        conv_bias_act!(y, x, w, cdims, b, identity; kwargs...)
        # For nice activation functions including `relu` and `tanh`, this will over-write `y`
        # because it is not needed for the gradient. But in the general case, it will broadacst
        # making `z = σ.(y)`, and the gradient will need both `y` and `z`.
        return bias_act!(σ, y)
    end
    # But on the CPU, this is sub-optimal? Maybe not if we're careful.
end

# Best case, with a nice activation function, we can do much of the gradient calculation in-place:
for (act, grad) in INPLACE_ACTS

    @eval function rrule(::typeof(conv_bias_act), x, w, cdims, b::B, σ; kw...) where {B}
        Ω = conv_bias_act(x, w, cdims, b, σ; kw...)
        size_b = size(b)
        # TODO pull this out?
        function conv_bias_act_pullback(Δ_raw)
            Δ_out = colmajor(unthunk(Δ_raw)) # gradient outside the activation function
            Δ = @. Δ_out * $grad             # inside it -- we can overwrite Ω
            if eltype(B) == Bool
                Δb = NoTangent()
            else
                # Δb = sum!(similar(B, size_b), Δ)
                dims = filter(d -> get(size_b, d, 1)==1, ntuple(identity, ndims(Δ)))
                Δb = reshape(sum(Δ; dims = dims), size_b)
            end
            return (
                NoTangent(), # func
                @thunk(∇conv_data(Δ, w, cdims; kw...)),
                @thunk(∇conv_filter(x, Δ, cdims; kw...)),
                NoTangent(), # cdims
                Δb,
                NoTangent(), # σ
            )
        end
        return Ω, conv_bias_act_pullback
    end

end

# Generic case: `bias_act!(σ, y) = σ.(y)` is handling the gradient of the activation,
# we only need to handle the gradient of the convolution inside.
function rrule(::typeof(conv_bias_act!), y, x, w, cdims, b::B, σ::typeof(identity); kw...) where {B}
    Ω = conv_bias_act!(y, x, w, cdims, b, σ; kw...)
    size_b = size(b)
    function conv_bias_act!_pullback(Δ_raw)
        Δ = colmajor(unthunk(Δ_raw)) 
        if eltype(B) == Bool
            Δb = NoTangent()
        else
            # Δb = sum!(similar(B, size_b), Δ)
            dims = filter(d -> get(size_b, d, 1)==1, ntuple(ideneity, ndims(Δ)))
            Δb = reshape(sum(Δ; dims = dims), size_b)
        end
        return (
            NoTangent(), # func
            NoTangent(), # y
            @thunk(∇conv_data(Δ, w, cdims; kw...)),
            @thunk(∇conv_filter(x, Δ, cdims; kw...)),
            NoTangent(), # cdims
            Δb,
            NoTangent(), # σ
        )
    end
    return Ω, conv_bias_act_pullback
end


# CPU actor. 
function conv_bias_act!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5}, w::AbstractArray{wT,5}, 
                cdims::ConvDims, b, σ; kwargs...) where {yT, xT, wT}
    y = conv!(y, x, w, cdims)
    if σ != identity || b isa AbstractArray
        return bias_act!(σ, y, b)
    else
        # nothing to do
        return y
    end
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


#=

] add https://github.com/mcabbott/NNlib.jl#activate

# w = rand(3, 4, 5);
# b = zeros(5)
# c1 = Conv(weight, bias, sigmoid) # Conv((3,), 4 => 5, σ)
# Conv((5,5), 1=>7, relu)

x = randn(Float32, 28, 28, 1, 32);
w = randn(Float32, 5, 5, 1, 7);
b = zeros(Float32, 7);
cdims = DenseConvDims(x, w)

y = conv(x, w, cdims); summary(y) # 24×24×7×32


julia> @btime relu.(conv($x, $w, $cdims) .+ reshape($b,1,1,:));
  191.541 μs (52 allocations: 1.21 MiB)
  844.718 μs (52 allocations: 1.21 MiB)  # cyclops

julia> @btime conv_bias_act($x, $w, $cdims, reshape($b,1,1,:), relu);
  232.500 μs (52 allocations: 732.06 KiB)
  808.792 μs (52 allocations: 732.09 KiB)  # cyclops

using NNlib, NNlibCUDA, CUDA, BenchmarkTools
CUDA.device!(0); @time cu(ones(3)) .+ 1
cx = cu(x);
cw = cu(w);
cb = cu(b);

julia> @btime CUDA.@sync relu.(conv($cx, $cw, $cdims) .+ reshape($cb,1,1,:));
  86.972 μs (75 allocations: 2.84 KiB)

julia> @btime CUDA.@sync  conv_bias_act($cx, $cw, $cdims, reshape($cb,1,1,:,1), relu);
  85.438 μs (64 allocations: 2.31 KiB)

julia> CUDA.@time relu.(conv(cx, cw, cdims) .+ reshape(cb,1,1,:));
  0.000680 seconds (89 CPU allocations: 3.141 KiB) (2 GPU allocations: 1008.000 KiB, 6.50% gc time of which 68.63% spent allocating)

julia> CUDA.@time conv_bias_act(cx, cw, cdims, reshape(cb,1,1,:,1), relu);
  0.000417 seconds (65 CPU allocations: 2.344 KiB) (1 GPU allocation: 504.000 KiB, 7.61% gc time of which 81.12% spent allocating)




# Bigger numbers, over-writing:
x = randn(Float32, 100, 100, 3, 32*2);
w = randn(Float32, 5, 5, 3, 32);
b = zeros(Float32, 32);
cdims = DenseConvDims(x, w)
cx = cu(x);
cw = cu(w);
cb = cu(b);

myone(relu, cx, cw, cdims, cb) = bias_act!(relu, conv(cx, cw, cdims), reshape(cb,1,1,:));

julia> @btime CUDA.@sync relu.(conv($cx, $cw, $cdims) .+ reshape($cb,1,1,:));
  1.290 ms (235 allocations: 5.42 KiB)

julia> @btime CUDA.@sync  conv_bias_act($cx, $cw, $cdims, reshape($cb,1,1,:,1), relu);
  474.030 μs (115 allocations: 3.14 KiB)

julia> @btime CUDA.@sync bias_act!(relu, conv($cx, $cw, $cdims), reshape($cb,1,1,:));
  1.054 ms (375 allocations: 7.48 KiB)

julia> CUDA.@time relu.(conv(cx, cw, cdims) .+ reshape(cb,1,1,:));
  0.352912 seconds (382 CPU allocations: 8.375 KiB, 90.98% gc time) (3 GPU allocations: 144.063 MiB, 99.66% gc time of which 0.03% spent allocating)

julia> CUDA.@time conv_bias_act(cx, cw, cdims, reshape(cb,1,1,:,1), relu);
  0.333903 seconds (225 CPU allocations: 4.906 KiB, 96.56% gc time) (2 GPU allocations: 72.063 MiB, 99.80% gc time of which 0.01% spent allocating)

julia> CUDA.@time bias_act!(relu, conv(cx, cw, cdims), reshape(cb,1,1,:)); # wtf?
  0.001556 seconds (345 CPU allocations: 7.516 KiB) (2 GPU allocations: 72.063 MiB, 2.35% gc time of which 75.07% spent allocating)

julia> CUDA.@time myone(relu, cx, cw, cdims, cb); 
  0.003185 seconds (344 CPU allocations: 7.484 KiB) (2 GPU allocations: 72.063 MiB, 46.00% gc time of which 97.58% spent allocating)



# If you read this selectively, you learn that:
# This fused thing really is twice as fast
# my trick saves memory but not time
# Gradients... to come!



# https://github.com/FluxML/NNlib.jl/issues/329

using Flux, CUDA
using CUDA.CUDNN: scalingParameter, CUDNN_CONVOLUTION, convdims, 
                  cudnnConvolutionDescriptor, cudnnConvolutionBwdDataAlgoPerf,
                  cudnnConvolutionForward!, cudnnConvolutionBwdFilterAlgoPerf,
                  cudnnConvolutionBackwardData, cudnnConvolutionBackwardFilter,
                  cudnnConvolutionBackwardBias, CUDNN_ACTIVATION_IDENTITY,
                  CUDNN_ACTIVATION_RELU

const CUDNNFloat = Union{Float16,Float32,Float64}

# From https://github.com/FluxML/Flux.jl/pull/1302
function (c::Conv)(x::AbstractArray)
    σ, b = c.σ, reshape(c.bias, ntuple(_->1, length(c.stride))..., :, 1)
    cdims = DenseConvDims(x, c.weight; stride=c.stride, padding=c.pad, dilation=c.dilation)
    # σ.(conv(x, c.weight, cdims) .+ b)
    conv_bias_act(x, c.weight, cdims, b, σ)
end

# From https://github.com/FluxML/Flux.jl/pull/1302
NNlib.conv_bias_act(x, w, cdims::DenseConvDims, b::Flux.Zeros, σ) = σ.(conv(x, w, cdims))
function NNlib.conv_bias_act(x::CuArray, w::CuArray{T}, cdims::DenseConvDims, b::Flux.Zeros, σ) where T
  bz = gpu(collect(b))
  conv_bias_act(x, w, cdims, bz, σ)
end

# https://github.com/FluxML/NNlibCUDA.jl/blob/master/src/cudnn/conv.jl#L51
function NNlib.conv_bias_act!(y::DenseCuArray{T}, x::DenseCuArray{T}, w::DenseCuArray{T}, 
                            cdims::DenseConvDims, bias::DenseCuArray{T}, σ=identity;
                            z::DenseCuArray{T}=y, alpha=1, beta=0, algo=-1) where T<:CUDNNFloat
    # if cudnnversion() < v"6"
    #     all(x -> x == 1, dilation(cdims)) || error("Only dilation = 1 is supported in cuDNN version < 6")
    # end
    if algo != -1
        @warn "The algo option has been deprecated, the fastest algo is computed automatically" maxlog=1
    end    
    d = cudnnConvolutionDescriptor(cdims, x)
    # only relu and identity are supported by cudnnConvolutionForward!
    activation = (σ == NNlib.relu ? CUDA.CUDNN.CUDNN_ACTIVATION_RELU : CUDA.CUDNN.CUDNN_ACTIVATION_IDENTITY)
    cudnnConvolutionForward!(y, w, x, d; z, bias, activation, alpha, beta)
    if activation === CUDA.CUDNN.CUDNN_ACTIVATION_IDENTITY && σ ∉ (nothing, identity)
        y = σ.(y)
    end
    return y
end

function oned_test()
    # Sequential MNIST size
    x = randn(Float32, 782, 1, 32)
    c = Conv((3,), 1=>2, relu)
    out = c(x)
    g = gradient(Flux.params(c)) do 
        sum(abs2, c(x))
    end
    return g
end

=#