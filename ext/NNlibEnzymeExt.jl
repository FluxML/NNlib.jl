module NNlibEnzymeExt

using NNlib
isdefined(Base, :get_extension) ? (import Enzyme) : (import ..Enzyme)

using Enzyme

using EnzymeCore

function EnzymeCore.EnzymeRules.augmented_primal(config, func::Const{typeof(NNlib.conv!)}, ::Type{RT}, y::OutType, x, w, cdims::Const; kwargs...)
) where {OutType, RT}

    @assert !(OutType <: Const)
    if OutType <: Duplicated || OutType <: DuplicatedNoNeed
        func.val(y.val, x.val, y.val, cdims.val; kwargs...)
    end

    dres = if EnzymeRules.width(config) == 1
        func.val(prob.dval, alg.val; kwargs...)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            func.val(prob.dval[i], alg.val; kwargs...)
        end
    end

    primal = if EnzymeRules.needs_primal(config)
        y.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        y.dval
    else
        nothing
    end

    # Cache x if its overwritten and w is active (and thus required)
    cache_x = ( EnzymeRules.overwritten(config)[3] && !(typeof(w) <: Const) ) ? copy(x.val) : nothing

    # Cache w if its overwritten and x is active (and thus required)
    cache_w = ( EnzymeRules.overwritten(config)[4] && !(typeof(x) <: Const) ) ? copy(w.val) : nothing

    cache = (cache_x, cache_w)

    return EnzymeCore.EnzymeRules.AugmentedReturn(y.val, y.dval, cache)
end

function EnzymeCore.EnzymeRules.reverse(config, func::Const{typeof(NNlib.conv!)}, ::Type{RT}, cache, y, x, w, cdims::Const; kwargs...) where {RT}
    cache_x, cache_w = cache

    # Don't cache x if not overwritten and w is active (and thus required)
    if !(typeof(w) <: Const)
        if !EnzymeRules.overwritten(config)[3]
            cache_x = x.val
        end
    end

    # Don't cache w if not overwritten and x is active (and thus required)
    if !(typeof(x) <: Const)
        if !EnzymeRules.overwritten(config)[4]
            cache_w = w.val
        end
    end

    dys = y.dval
    dxs = (typeof(x) <: Const) ? nothing : x.dval
    dws = (typeof(w) <: Const) ? nothing : w.dval

    if EnzymeRules.width(config) == 1
        dys = (dys,)
        dxs = (dxs,)
        dws = (dws,)
    end

    for (dy, dx, dw) in (dys, dxs, dws)
        if !(typeof(x) <: Const)
            # dx += grad wrt x
            NNlib.∇conv_data!(dx, dy, cache_w, cdims; alpha=1, beta=1, kwargs...)
        end
        if !(typeof(y) <: Const)
            # dw += grad wrt w
            NNlib.∇conv_filter!(dw, cache_x, dy, cdims; alpha=1, beta=1, kwargs...)
        end
    end

    return (nothing, nothing, nothing, nothing)
end