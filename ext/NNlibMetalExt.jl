module NNlibMetalExt


using Metal: Metal, MtlDeviceArray, method_table, @device_override
using NNlib: NNlib

@device_override NNlib.tanh_fast(x) = Base.FastMath.tanh_fast(x)

# Atomix's Metal atomics only support a subset of reduction ops: they raise a
# compile-time error for float `max`/`min` and silently no-op for `*`/`/`. Route
# scatter atomics through `Metal.@atomic` instead, which falls back to a generic
# compare-and-swap loop for ops without a native atomic.
# See https://github.com/FluxML/NNlib.jl/issues/534.
@inline function NNlib._atomic_scatter!(dst::MtlDeviceArray, idx, op::OP, val) where OP
    @inbounds Metal.@atomic dst[idx...] = op(dst[idx...], val)
    return nothing
end

end
