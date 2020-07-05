include("impl.jl")

## NNPACK supports only Float32
for (front_name, backend) in (
        :conv          => :_nnpack,
        :∇conv_data    => :_nnpack,
        :∇conv_filter  => :_nnpack,
    )
    @eval begin
        function $(Symbol("$(front_name)$(backend)!"))(
                        out::Array{T1,4}, in1::Array{T2,4}, in2::Array{T3,4},
                        cdims::ConvDims; kwargs...) where {T1, T2, T3}
            @warn "Automatically converting input tensor to Float32. This will have performance implications" maxlog=1
            # Output must of the same type as in the function signature
            T1.($(Symbol("$(front_name)$(backend)!"))(Float32.(out), Float32.(in1),
                                                      Float32.(in2), cdims; kwargs...))
        end
    end
end

function maxpool_nnpack!(y::Array{T1, 4}, x::Array{T2, 4}, pdims::PoolDims;
                         kwargs...) where {T1, T2}
    @warn "Automatically converting input tensor to Float32. This will have performance implications" maxlog=1
    # We want the output to be of the same type as desired
    T1.(maxpool_nnpack!(Float32.(y), Float32.(x), pdims; kwargs...))
end

"""
    nnpack_supported_operation(cdims::ConvDims)
    nnpack_supported_operation(pdims::PoolDims)

Returns `true` if nnpack supports the convolution/pooling operation for the given parameters.
"""
function nnpack_supported_operation(pdims::PoolDims{2, K, S, P, (1, 1)}) where {K, S, P}
    val = input_size(pdims)[1:2] .+ (P[1] + P[2], P[3] + P[4]) .- K
    return val .% S == (0, 0) ? true : false
end

function nnpack_supported_operation(cdims::ConvDims{2, K, (1, 1), P, (1, 1)}) where {K, S, P}
    return true
end

# Return false for everything else
nnpack_supported_operation(dims) = false
