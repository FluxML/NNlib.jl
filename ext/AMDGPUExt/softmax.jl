for fname in (:softmax, :logsoftmax)
    @eval function NNlib.$(fname)(x::ROCArray{T}; dims = 1) where T <: MIOPENFloat
        MIOpen.$(fname)(x; dims)
    end

    @eval function NNlib.$(Symbol("∇$(fname)"))(
        dy::ROCArray{T, N}, x::ROCArray{T, N}, y::ROCArray{T, N}; dims = 1,
    ) where {T <: MIOPENFloat, N}
        MIOpen.$(Symbol("∇$(fname)!"))(dy, y; dims)
    end
end
