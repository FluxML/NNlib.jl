module NNlibMooncakeCUDAExt

using NNlib, Mooncake
using CUDA.CUDACore: CuArray
using Base: IEEEFloat

import NNlib: batchnorm, ∇batchnorm
import Mooncake:
    MinimalCtx,
    rrule!!,
    @is_primitive,
    CoDual,
    NoRData,
    primal,
    tangent,
    arrayify,
    zero_fcodual,
    zero_rdata

# cudnnBNForward! (called inside batchnorm) contains a try/catch that Mooncake cannot
# trace. We intercept at the batchnorm level: run the primal via cuDNN as normal, then
# use NNlib.∇batchnorm in the pullback, which is safe because pullbacks run as plain Julia.
#
# Handles the main 4-D / 5-D cuDNN path. The 2-D case (which reshapes then delegates to 4-D)
# and the g=Nothing / b=Nothing case are traced through by Mooncake and reach this rule
# via Core.kwcall after the reshape.

@is_primitive(
    MinimalCtx,
    Tuple{
        typeof(Core.kwcall),
        NamedTuple,
        typeof(batchnorm),
        CuArray{T},
        CuArray{T},
        CuArray{T},
        Any,
        Any,
        Any,
    } where {T<:IEEEFloat},
)
function rrule!!(
    ::CoDual{typeof(Core.kwcall)},
    kw::CoDual{<:NamedTuple},
    ::CoDual{typeof(batchnorm)},
    g::CoDual{<:CuArray{T}},
    b::CoDual{<:CuArray{T}},
    x::CoDual{<:CuArray{T}},
    running_mean::CoDual,
    running_var::CoDual,
    momentum::CoDual,
) where {T<:IEEEFloat}
    pg, dg = arrayify(g)
    pb, db = arrayify(b)
    px, dx = arrayify(x)
    pkw = primal(kw)
    prm = primal(running_mean)
    prv = primal(running_var)
    pm = primal(momentum)

    y = batchnorm(pg, pb, px, prm, prv, pm; pkw...)
    dy_out = zero(y)
    zero_kw = zero_rdata(pkw)

    function batchnorm_pb!!(::NoRData)
        # ∇batchnorm forwards kwargs to cudnnBNBackward!, which defaults track_stats=true.
        # With track_stats=true and running_mean=nothing, cudnnBNBackward! passes nothing
        # directly to the cuDNN C API instead of CU_NULL, corrupting the backward pass.
        # Overriding track_stats=false when running stats are nothing makes cudnnBNBackward!
        # substitute CU_NULL, which is what cuDNN expects in this case.
        kw_back = prm === nothing ? merge(pkw, (track_stats=false,)) : pkw
        grads = ∇batchnorm(pg, pb, px, dy_out, prm, prv, pm; kw_back...)
        grads[1] !== nothing && (dg .+= grads[1])
        grads[2] !== nothing && (db .+= grads[2])
        dx .+= grads[3]
        return NoRData(),
        zero_kw,
        NoRData(),
        NoRData(),
        NoRData(),
        NoRData(),
        NoRData(),
        NoRData(),
        zero(pm)
    end

    return CoDual(y, dy_out), batchnorm_pb!!
end

end
