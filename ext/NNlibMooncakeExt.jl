module NNlibMooncakeExt

using Mooncake: Mooncake, MinimalCtx, @zero_adjoint, @zero_derivative
using NNlib: NNlib

# Mooncake's reverse pass restores every value that was overwritten on the forward pass.
# Traced like ordinary code, the in-place update of the running statistics is therefore
# undone, and a gradient call hands back the statistics it started with. As a primitive
# the update is an opaque call instead, so its writes are kept, as under Zygote and Enzyme.
#
# The statistics are updated once per evaluation of the function. `prepare_gradient_cache`
# and `prepare_pullback_cache` evaluate it once themselves, so they update them as well.
@zero_adjoint MinimalCtx Tuple{typeof(NNlib._update_running_stats!),Vararg}

# `CUDA.default_rng()` reads the task-local storage, which Mooncake cannot differentiate
# (chalk-lab/Mooncake.jl#1368). Same as the `@non_differentiable` for ChainRules.
@zero_derivative MinimalCtx Tuple{typeof(NNlib._rng_from_array),AbstractArray}

# Mooncake has no predicate to query, so `within_gradient` is replaced in the code it differentiates.
Mooncake.@mooncake_overlay NNlib.within_gradient(x) = true

# `_exp` builds a power of two from its bits, which Mooncake refuses to differentiate
# (a `bitcast` from an integer to a float). It has a rule for `exp` instead.
Mooncake.@mooncake_overlay NNlib._exp(x::Float32) = exp(x)
Mooncake.@mooncake_overlay NNlib._exp(x::Float64) = exp(x)

end
