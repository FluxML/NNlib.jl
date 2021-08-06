export conv_bias_act, conv_bias_act!

"""
    conv_bias_act(x, w, cdims, b, σ)

This should be equivalent to `σ.(conv(x, w, cdims) .+ reshape(b, ...))`, 
but faster because it will:
* try hard to re-use memory, when safe to do so,
* call fused CUDNN operation, when possible?

TODO:
* conv_bias_act! is overloaded here:
https://github.com/FluxML/NNlibCUDA.jl/blob/master/src/cudnn/conv.jl#L37
but fusion of σ other than identity, relu, needs more thought...
should probably go to `bias_act!(σ, conv_bias_act(identity, x, w, ...))`
* slightly weird signature, worth changing?
* simplify so that conv! does not need a gradient?
"""
function conv_bias_act(x::AbstractArray{xT,N}, w::AbstractArray{wT,N}, 
                cdims::ConvDims, b, σ=identity; kwargs...) where {xT, wT, N}
    y = similar(x, promote_type(xT, wT), output_size(cdims)..., channels_out(cdims), size(x,N))
    return conv_bias_act!(y, x, w, cdims, b, σ; kwargs...)
end

# The fast path is handled by bias_act!, including fusing the gradient as best it can.
# There is a gradient definition for conv!, with NoTangent() for 1st arg.
function conv_bias_act!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5}, w::AbstractArray{wT,5}, 
                cdims::ConvDims, b, σ; kwargs...) where {yT, xT, wT}
    conv!(y, x, w, cdims)
    bias_act!(σ, y, b)
end

# # Fast case, for whitelisted activation functions:
# for (act, grad) in INPLACE_ACTS
#     # Forward
#     @eval function conv_bias_act!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5}, w::AbstractArray{wT,5}, 
#                     cdims::ConvDims, b, σ::typeof($act); kwargs...) where {yT, xT, wT}
#         conv!(y, x, w, cdims)
#         y .= σ.(y .+ b)
#         return y
#     end

#     # Gradient -- TODO
# end

# # Fallback case: we cannot in general assume that the gradient of `z = σ.(y)` does not need
# # both `y` and `z`, so we must keep them both around:
# function conv_bias_act!(y::AbstractArray{yT,5}, x::AbstractArray{xT,5}, w::AbstractArray{wT,5}, 
#                 cdims::ConvDims, b, σ; kwargs...) where {yT, xT, wT}
#     conv!(y, x, w, cdims)
#     return σ.(y .+ b)
# end

# # Fallback gradient
# function rrule(::RuleConfig, ::typeof(conv_bias_act!), y::AbstractArray{yT,5}, x::AbstractArray{xT,5}, w::AbstractArray{wT,5},
#             cdims::ConvDims, b::B, σ; kwargs...) where {yT, xT, wT, B}
#     conv!(y, x, w, cdims)
#     Ω, uncast = rrule_via_ad(config, broadcast, σ, y)  # broadcasting does not overwrite result of conv
#     function conv_bias_act!_pullback(Δ_raw)
#         dy = uncast(unthunk(Δ_raw))[3]
#         Δ = colmajor(dy)
#         if B <: AbstractArray
#             dims = ntuple(d -> d+1, ndims(Δ)-1)
#             sum(Δ; dims = dims)
#         else
#             db = NoTangent()
#         end
#         return (
#             NoTangent(), # function
#             NoTangent(), # y
#             @thunk($∇conv_data(Δ, w, cdims; kw...)),
#             @thunk($∇conv_filter(x, Δ, cdims; kw...)),
#             NoTangent(), # cdims
#             db,
#         )
#     end
#     return Ω, conv_bias_act!_pullback
# end

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

] add 

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

julia> @btime conv_bias_act($x, $w, $cdims, reshape($b,1,1,:), relu);
  232.500 μs (52 allocations: 732.06 KiB)


using NNlib, NNlibCUDA, CUDA, BenchmarkTools
cx = cu(x);
cw = cu(w);
cb = cu(b);

julia> @btime CUDA.@sync relu.(conv($cx, $cw, $cdims) .+ reshape($cb,1,1,:));

julia> @btime CUDA.@sync  conv_bias_act($cx, $cw, $cdims, reshape($cb,1,1,:), relu);



=#