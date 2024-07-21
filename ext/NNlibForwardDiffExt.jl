module NNlibForwardDiffExt

using ForwardDiff: ForwardDiff
using NNlib: NNlib

NNlib.within_gradient(x::ForwardDiff.Dual) = true
NNlib.within_gradient(x::AbstractArray{<:ForwardDiff.Dual}) = true

end
