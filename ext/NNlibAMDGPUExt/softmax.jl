for fname in (:softmax, :logsoftmax)
    @eval function NNlib.$(fname)(x::ROCArray{Float32}; dims = 1)
        MIOpen.$(fname)(x; dims)
    end
    @eval function NNlib.$(fname)(x::ROCArray{Float16}; dims = 1)
        Float16.(MIOpen.$(fname)(Float32.(x); dims))
    end

    @eval function NNlib.$(Symbol("∇$(fname)"))(
        dy::ROCArray{T, N}, x::ROCArray{T, N}, y::ROCArray{T, N}; dims = 1,
    ) where {T <: MIOPENFloat, N}
        MIOpen.$(Symbol("∇$(fname)!"))(dy, y; dims)
    end
end
