import EnzymeCore

function EnzymeCore.EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.conv!)}, ::Type{RT}, y::OutType, x, w, cdims; kwargs...) where {OutType, RT}

    @assert !(OutType <: EnzymeCore.Const)
    if OutType <: EnzymeCore.Duplicated || OutType <: EnzymeCore.DuplicatedNoNeed
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
    cache_x = ( EnzymeCore.EnzymeRules.overwritten(config)[3] && !(typeof(w) <: EnzymeCore.Const) ) ? copy(x.val) : nothing

    # Cache w if its overwritten and x is active (and thus required)
    cache_w = ( EnzymeCore.EnzymeRules.overwritten(config)[4] && !(typeof(x) <: EnzymeCore.Const) ) ? copy(w.val) : nothing

    cache = (cache_x, cache_w)

    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeCore.EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.conv!)}, ::Type{RT}, cache, y, x, w, cdims; kwargs...) where {RT}
    cache_x, cache_w = cache

    # Don't cache x if not overwritten and w is active (and thus required)
    if !(typeof(w) <: EnzymeCore.Const)
        if !EnzymeCore.EnzymeRules.overwritten(config)[3]
            cache_x = x.val
        end
    end

    # Don't cache w if not overwritten and x is active (and thus required)
    if !(typeof(x) <: EnzymeCore.Const)
        if !EnzymeCore.EnzymeRules.overwritten(config)[4]
            cache_w = w.val
        end
    end

    dys = y.dval
    dxs = (typeof(x) <: EnzymeCore.Const) ? dys : x.dval
    dws = (typeof(w) <: EnzymeCore.Const) ? dys : w.dval

    if EnzymeCore.EnzymeRules.width(config) == 1
        dys = (dys,)
        dxs = (dxs,)
        dws = (dws,)
    end

    for (dy, dx, dw) in zip(dys, dxs, dws)
        if !(typeof(x) <: EnzymeCore.Const) && dx !== x.val
            # dx += grad wrt x.val
            NNlib.∇conv_data!(dx, dy, cache_w, cdims.val; alpha=eltype(dw)(1), beta=eltype(dw)(1), kwargs...)
        end
        if !(typeof(w) <: EnzymeCore.Const) && dw !== w.val
            # dw += grad wrt w.val
            NNlib.∇conv_filter!(dw, cache_x, dy, cdims.val; alpha=eltype(dw)(1), beta=eltype(dw)(1), kwargs...)
        end
        dy .= 0
    end

    return (nothing, nothing, nothing, nothing)
end


function EnzymeCore.EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.gather!)}, ::Type{RT}, dst::OutType, src, idx::EnzymeCore.Const) where {OutType, RT}

    @assert !(OutType <: EnzymeCore.Const)
    if OutType <: EnzymeCore.Duplicated || OutType <: EnzymeCore.DuplicatedNoNeed
        func.val(dst.val, src.val, idx.val)
    end

    primal = if EnzymeCore.EnzymeRules.needs_primal(config)
        dst.val
    else
        nothing
    end
    shadow = if EnzymeCore.EnzymeRules.needs_shadow(config)
        dst.dval
    else
        nothing
    end

    # Cache idx if its overwritten
    cache_idx = ( EnzymeCore.EnzymeRules.overwritten(config)[4] && !(typeof(src) <: EnzymeCore.Const) ) ? copy(idx.val) : nothing

    return EnzymeCore.EnzymeRules.AugmentedReturn(primal, shadow, cache_idx)
end

function EnzymeCore.EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.gather!)}, ::Type{RT}, cache_idx, dst::OutType, src, idx::EnzymeCore.Const) where {OutType, RT}

    # Don't cache idx if not overwritten
    if !(typeof(src) <: EnzymeCore.Const)
        if !EnzymeCore.EnzymeRules.overwritten(config)[4]
            cache_idx = idx.val
        end
    end

    ddsts = dst.dval
    dsrcs = src.dval

    if EnzymeCore.EnzymeRules.width(config) == 1
        ddsts = (ddsts,)
        dsrcs = (dsrcs,)
    end

    for (ddst, dsrc) in zip(ddsts, dsrcs)
        if !(typeof(src) <: EnzymeCore.Const) && ddst !== dst.val
            src_size = size(src.val)
            NNlib.∇gather_src(ddst, src_size, cache_idx)
        end
        if !(typeof(w) <: EnzymeCore.Const) && dw !== w
            ddst .= 0
        end
    end

    return (nothing, nothing, nothing, nothing)
end