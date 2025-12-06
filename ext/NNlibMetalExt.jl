module NNlibMetalExt


using Metal: method_table, @device_override
using NNlib: NNlib

@device_override NNlib.tanh_fast(x) = Base.FastMath.tanh_fast(x)

end
