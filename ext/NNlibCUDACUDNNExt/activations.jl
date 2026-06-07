
# Activation

# We deliberately do NOT route activation broadcasts (relu, σ, elu, tanh, ...)
# through cuDNN's `cudnnActivationForward!`. Those overloads used to pirate
# `Base.materialize`/`materialize!` (hurting latency via invalidations, #504) and,
# worse, cuDNN does not propagate NaNs by default, so e.g. `relu.(cu([NaN]))`
# returned `0` instead of `NaN` (#509). CUDA.jl's native broadcast is correct and,
# for these memory-bandwidth-bound elementwise ops, just as fast.

# CUDNN_ACTIVATION_IDENTITY does not work with cudnnActivationForward
# FIXME: put this optimization in GPUArrays' `copyto!` (like Base.Broadcast's `copyto!`)
Base.broadcasted(::typeof(identity), x::DenseCuArray{T}) where {T<:CUDNNFloat} = x
