for fname in (:softmax, :logsoftmax)
    @eval function NNlib.$(fname)(x::ROCArray{Float32}; dims = 1)
        MIOpen.$(fname)(x; dims)
    end

    @eval function NNlib.$(Symbol("∇$(fname)_data"))(
        dy::ROCArray{T, N}, y::ROCArray{T, N}; dims = 1,
    ) where {T <: MIOPENFloat, N}
        MIOpen.$(Symbol("∇$(fname)"))(dy, y; dims)
    end

    # FP16 variants. Cast to FP32 -> compute result -> cast back to FP16.
    # Eagerly free intermediate FP32 arrays.

    @eval function NNlib.$(fname)(x::ROCArray{Float16}; dims = 1)
        x_fp32 = Float32.(x)
        y_fp32 = NNlib.$(fname)(x_fp32; dims)
        y = Float16.(y_fp32)

        AMDGPU.synchronize()
        AMDGPU.unsafe_free!.((x_fp32, y_fp32))
        return y
    end

    @eval function ChainRulesCore.rrule(::typeof(NNlib.$(fname)), x::ROCArray{Float16}; dims = 1)
        x_fp32 = Float32.(x)
        y_fp32 = NNlib.$(fname)(x_fp32; dims)
        AMDGPU.synchronize()
        AMDGPU.unsafe_free!(x_fp32)

        y = Float16.(y_fp32)

        function _pullback(dy)
            dy_fp32 = Float32.(unthunk(dy))
            dx_fp32 = NNlib.$(Symbol("∇$(fname)_data"))(dy_fp32, y_fp32; dims)
            AMDGPU.synchronize()
            AMDGPU.unsafe_free!.((dy_fp32, y_fp32))

            dx = Float16.(dx_fp32)

            AMDGPU.synchronize()
            AMDGPU.unsafe_free!(dx_fp32)
            NoTangent(), dx
        end
        return y, _pullback
    end
end
