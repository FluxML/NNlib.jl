include("impl.jl")


for (front_name, backend) in (
        :conv          => :_nnpack,
        :∇conv_data    => :_nnpack,
        :∇conv_filter  => :_nnpack,
    )
    @eval begin
        @timeit_debug to function $(Symbol("$(front_name)$(backend)!"))(
                        out::Array{T1,4}, in1::Array{T2,4}, in2::Array{T3,4},
                        cdims::ConvDims; kwargs...) where {T1, T2, T3}
            @warn "Automatically converting $(size(in1)) input tensor to Float32" maxlog=1
            # Output must of the same type as in the function signature
            T1.($(Symbol("$(front_name)$(backend)!"))(Float32.(out), Float32.(in1),
                                                      Float32.(in2), cdims; kwargs...))
        end
    end
end


function maxpool_nnpack!(y::Array{T1, 4}, x::Array{T2, 4}, pdims::PoolDims;
                         kwargs...) where {T1, T2}
    @warn "Automatically converting $(size(x)) input tensor to Float32" maxlog=1
    # We want the output to be of the same type as desired
    T1.(maxpool_nnpack!(Float32.(y), Float32.(x), pdims; kwargs...))
end
