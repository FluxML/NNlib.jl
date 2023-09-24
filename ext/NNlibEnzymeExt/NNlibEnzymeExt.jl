module NNlibEnzymeExt

using NNlib
isdefined(Base, :get_extension) ? (import Enzyme) : (import ..Enzyme)

using EnzymeCore

function EnzymeCore.EnzymeRules.augmented_primal(config, func::Const{typeof(NNlib.conv!)}, ::Type{RT}, y::OutType, x, w, cdims; kwargs...) where {OutType, RT}

    @assert !(OutType <: Const)
    if OutType <: Duplicated || OutType <: DuplicatedNoNeed
        func.val(y.val, x.val, w.val, cdims.val; kwargs...)
    end

    primal = if EnzymeCore.EnzymeRules.needs_primal(config)
        y.val
    else
        nothing
    end
    shadow = if EnzymeCore.EnzymeRules.needs_shadow(config)
        y.dval
    else
        nothing
    end

    # Cache x if its overwritten and w is active (and thus required)
    cache_x = ( EnzymeCore.EnzymeRules.overwritten(config)[3] && !(typeof(w) <: Const) ) ? copy(x.val) : nothing

    # Cache w if its overwritten and x is active (and thus required)
    cache_w = ( EnzymeCore.EnzymeRules.overwritten(config)[4] && !(typeof(x) <: Const) ) ? copy(w.val) : nothing

    cache = (cache_x, cache_w)

    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeCore.EnzymeRules.reverse(config, func::Const{typeof(NNlib.conv!)}, ::Type{RT}, cache, y, x, w, cdims; kwargs...) where {RT}
    cache_x, cache_w = cache

    # Don't cache x if not overwritten and w is active (and thus required)
    if !(typeof(w) <: Const)
        if !EnzymeCore.EnzymeRules.overwritten(config)[3]
            cache_x = x.val
        end
    end

    # Don't cache w if not overwritten and x is active (and thus required)
    if !(typeof(x) <: Const)
        if !EnzymeCore.EnzymeRules.overwritten(config)[4]
            cache_w = w.val
        end
    end

    dys = y.dval
    dxs = (typeof(x) <: Const) ? dys : x.dval
    dws = (typeof(w) <: Const) ? dys : w.dval

    if EnzymeCore.EnzymeRules.width(config) == 1
        dys = (dys,)
        dxs = (dxs,)
        dws = (dws,)
    end

    for (dy, dx, dw) in zip(dys, dxs, dws)
        if !(typeof(x) <: Const) && dx !== x
            # dx += grad wrt x
            NNlib.∇conv_data!(dx, dy, cache_w, cdims.val; alpha=eltype(dw)(1), beta=eltype(dw)(1), kwargs...)
        end
        if !(typeof(w) <: Const) && dw !== w
            # dw += grad wrt w
            NNlib.∇conv_filter!(dw, cache_x, dy, cdims.val; alpha=eltype(dw)(1), beta=eltype(dw)(1), kwargs...)
        end
        dy .= 0
    end

    return (nothing, nothing, nothing, nothing)
end

end