module NNlibMooncakeExt

using NNlib, Mooncake

# Mooncake has no predicate to query, so `within_gradient` is replaced in the code it differentiates.
Mooncake.@mooncake_overlay NNlib.within_gradient(x) = true

end
