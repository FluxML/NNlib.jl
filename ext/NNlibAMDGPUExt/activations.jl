# We deliberately do NOT route activation broadcasts (relu, σ, tanh, ...) through
# MIOpen. Those overloads used to pirate `Base.materialize` (hurting latency via
# invalidations, #504) and, like cuDNN, MIOpen does not propagate NaNs (#509).
# AMDGPU's native broadcast is correct and, for these elementwise ops, just as fast.

Base.broadcasted(::typeof(identity), x::ROCArray{T}) where {T<:MIOPENFloat} = x
