module NNlibReactantExt

using NNlib: NNlib
using Reactant: TracedRNumber

# Reactant traces an array of `T` as an array of `TracedRNumber{T}`. The normalization
# functions treat half precision as they do for `T` itself: they require Float32
# parameters, accumulate in Float32 and cast the result back to the input eltype.
# Otherwise Float32 parameters would promote the result to Float32.
NNlib._stats_type(::Type{TracedRNumber{T}}) where {T} = TracedRNumber{NNlib._stats_type(T)}

end
