module NNlibForwardDiffExt

using ForwardDiff: ForwardDiff
using NNlib: NNlib

NNlib.within_gradient(x::ForwardDiff.Dual) = true
NNlib.within_gradient(x::AbstractArray{<:ForwardDiff.Dual}) = true

# Strip the dual part before writing batch statistics back into (plain-typed)
# running-mean/var buffers (see NNlib's `_update_running_stats!`).
NNlib._value(x::ForwardDiff.Dual) = ForwardDiff.value(x)

end
