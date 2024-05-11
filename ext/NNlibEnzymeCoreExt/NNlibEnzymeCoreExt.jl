module NNlibEnzymeCoreExt

using NNlib
import EnzymeCore
using Random

using EnzymeCore.EnzymeRules

for (name, dataname, filtername) in (
                                     (typeof(NNlib.conv!), NNlib.∇conv_data!, NNlib.∇conv_filter!),
                                     (typeof(NNlib.depthwiseconv!), NNlib.∇depthwiseconv_data!, NNlib.∇depthwiseconv_filter!),
                                     (typeof(NNlib.∇conv_data!), NNlib.conv!, NNlib.∇conv_filter!),
                                     (typeof(NNlib.∇conv_filter!), NNlib.∇conv_data!, NNlib.conv!),
                                    )
    @eval begin

		function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{$name}, ::Type{RT},
		                                                y::EnzymeCore.Annotation{<:AbstractArray{yT, N}},
		                                                x::EnzymeCore.Annotation{<:AbstractArray{xT, N}},
		                                                w::EnzymeCore.Annotation{<:AbstractArray{wT, N}},
		                                                cdims; kwargs...) where {RT, yT, xT, wT, N}

		    if typeof(y) <: EnzymeCore.Duplicated || typeof(y) <: EnzymeCore.BatchDuplicated
		        func.val(y.val, x.val, w.val, cdims.val; kwargs...)
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
		    cache_x = ( EnzymeRules.overwritten(config)[3]
		                && !(typeof(w) <: EnzymeCore.Const)
		                && !(typeof(y) <: EnzymeCore.Const)
		                ) ? copy(x.val) : nothing

		    # Cache w if its overwritten and x is active (and thus required)
		    cache_w = ( EnzymeRules.overwritten(config)[4]
		                && !(typeof(x) <: EnzymeCore.Const)
		                && !(typeof(y) <: EnzymeCore.Const)
		                ) ? copy(w.val) : nothing

		    cache = (cache_x, cache_w)

		    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
		end

		function EnzymeRules.reverse(config, func::EnzymeCore.Const{$name}, ::Type{RT}, cache,
		                                                y::EnzymeCore.Annotation{<:AbstractArray{yT, N}},
		                                                x::EnzymeCore.Annotation{<:AbstractArray{xT, N}},
		                                                w::EnzymeCore.Annotation{<:AbstractArray{wT, N}},
		                                                cdims; kwargs...) where {RT, yT, xT, wT, N}
		    cache_x, cache_w = cache

		    # Don't cache x if not overwritten and w is active (and thus required)
		    if !(typeof(w) <: EnzymeCore.Const) && !(typeof(y) <: EnzymeCore.Const)
		        if !EnzymeRules.overwritten(config)[3]
		            cache_x = x.val
		        end
		    end

		    # Don't cache w if not overwritten and x is active (and thus required)
		    if !(typeof(x) <: EnzymeCore.Const) && !(typeof(y) <: EnzymeCore.Const)
		        if !EnzymeRules.overwritten(config)[4]
		            cache_w = w.val
		        end
		    end

		    dys = y.dval
		    dxs = (typeof(x) <: EnzymeCore.Const) ? dys : x.dval
		    dws = (typeof(w) <: EnzymeCore.Const) ? dys : w.dval

		    if EnzymeRules.width(config) == 1
		        dys = (dys,)
		        dxs = (dxs,)
		        dws = (dws,)
		    end

		    for (dy, dx, dw) in zip(dys, dxs, dws)
		        if !(typeof(y) <: EnzymeCore.Const) && dy !== y.val

		            if !(typeof(x) <: EnzymeCore.Const) && dx !== x.val
		                # dx += grad wrt x.val
		                $dataname(dx, $(name != typeof(NNlib.∇conv_filter!) ? :dy : :cache_w), $(name != typeof(NNlib.∇conv_filter!) ? :cache_w : :dy), cdims.val; alpha=xT(1), beta=xT(1), kwargs...)
		            end
		            if !(typeof(w) <: EnzymeCore.Const) && dw !== w.val
		                # dw += grad wrt w.val
                        $filtername(dw, $(name != typeof(NNlib.∇conv_data!) ? :cache_x : :dy), $(name != typeof(NNlib.∇conv_data!) ? :dy : :cache_x), cdims.val; alpha=wT(1), beta=wT(1), kwargs...)
		            end
		            
		            dy .= 0
		        end
		    end

		    return (nothing, nothing, nothing, nothing)
		end

end
end

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.gather!)}, ::Type{RT}, dst::OutType, src, idx::EnzymeCore.Const) where {OutType, RT}

    if OutType <: EnzymeCore.Duplicated || OutType <: EnzymeCore.BatchDuplicated
        func.val(dst.val, src.val, idx.val)
    end

    primal = if EnzymeRules.needs_primal(config)
        dst.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        dst.dval
    else
        nothing
    end

    # Cache idx if its overwritten
    cache_idx = ( EnzymeRules.overwritten(config)[4]
                    && !(typeof(src) <: EnzymeCore.Const)
                    && !(typeof(dst) <: EnzymeCore.Const)
                    ) ? copy(idx.val) : nothing

    return EnzymeRules.AugmentedReturn(primal, shadow, cache_idx)
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.gather!)}, ::Type{RT}, cache_idx, dst::OutType, src, idx::EnzymeCore.Const) where {OutType, RT}

    # Don't cache idx if not overwritten
    if !(typeof(src) <: EnzymeCore.Const) && !(typeof(dst) <: EnzymeCore.Const)
        if !EnzymeRules.overwritten(config)[4]
            cache_idx = idx.val
        end
    end

    ddsts = dst.dval
    dsrcs = (typeof(src) <: EnzymeCore.Const) ? ddsts : src.dval

    if EnzymeRules.width(config) == 1
        ddsts = (ddsts,)
        dsrcs = (dsrcs,)
    end

    for (ddst, dsrc) in zip(ddsts, dsrcs)
        if !(typeof(dst) <: EnzymeCore.Const) && ddst !== dst.val

            if !(typeof(src) <: EnzymeCore.Const) && dsrc !== src.val
                NNlib.scatter!(+, dsrc, ddst, cache_idx)
            end

            ddst .= 0
        end
    end

    return (nothing, nothing, nothing)
end



function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.scatter!)}, ::Type{RT}, op::EnzymeCore.Const, dst::OutType, src, idx::EnzymeCore.Const) where {OutType, RT}

    @assert !(OutType <: EnzymeCore.Const)
    if OutType <: EnzymeCore.Duplicated || OutType <: EnzymeCore.BatchDuplicated
        func.val(op.val, dst.val, src.val, idx.val)
    end

    primal = if EnzymeRules.needs_primal(config)
        dst.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        dst.dval
    else
        nothing
    end

    # Cache idx if its overwritten
    cache_idx = ( EnzymeRules.overwritten(config)[4]
                    && !(typeof(src) <: EnzymeCore.Const)
                    && !(typeof(dst) <: EnzymeCore.Const)
                    ) ? copy(idx.val) : nothing

    return EnzymeRules.AugmentedReturn(primal, shadow, cache_idx)
end

function EnzymeRules.reverse(config,
										func::EnzymeCore.Const{typeof(NNlib.scatter!)},
										::Type{RT},
										cache_idx,
										op::Union{EnzymeCore.Const{typeof(+)},EnzymeCore.Const{typeof(-)}}, dst::OutType,
										src,
										idx::EnzymeCore.Const) where {OutType, RT}

    # Don't cache idx if not overwritten
    if !(typeof(src) <: EnzymeCore.Const) && !(typeof(dst) <: EnzymeCore.Const)
        if !EnzymeRules.overwritten(config)[4]
            cache_idx = idx.val
        end
    end

    ddsts = dst.dval
    dsrcs = (typeof(src) <: EnzymeCore.Const) ? ddsts : src.dval

    if EnzymeRules.width(config) == 1
        ddsts = (ddsts,)
        dsrcs = (dsrcs,)
    end

    for (ddst, dsrc) in zip(ddsts, dsrcs)
        if !(typeof(dst) <: EnzymeCore.Const) && ddst !== dst.val

            if !(typeof(src) <: EnzymeCore.Const) && dsrc !== src.val

                if eltype(typeof(op)) == typeof(+)
                    dsrc .+= NNlib.gather(ddst, cache_idx)
                else
                    @assert eltype(typeof(op)) == typeof(-)
                    dsrc .-= NNlib.gather(ddst, cache_idx)
                end
            end

        end
    end

    return (nothing, nothing, nothing, nothing)
end



for pool in [:maxpool, :meanpool, :lpnormpool]
    pool! = Symbol(pool, :!)
    ∇pool = Symbol(:∇, pool, :!)

    @eval begin

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof($pool!)}, ::Type{RT}, y::OutType, x, dims; kwargs...) where {OutType, RT}

    if OutType <: EnzymeCore.Duplicated || OutType <: EnzymeCore.BatchDuplicated
        func.val(y.val, x.val, dims.val; kwargs...)
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

    cache_y = ( EnzymeRules.overwritten(config)[2] 
                && !(typeof(x) <: EnzymeCore.Const) 
                && !(typeof(y) <: EnzymeCore.Const) 
                ) ? copy(y.val) : nothing

    cache_x = ( EnzymeRules.overwritten(config)[3]
                && !(typeof(x) <: EnzymeCore.Const) 
                && !(typeof(y) <: EnzymeCore.Const) 
                ) ? copy(x.val) : nothing

    cache = (cache_y, cache_x)

    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof($pool!)}, ::Type{RT}, cache, y, x, dims; kwargs...) where {RT}
    cache_y, cache_x = cache

    # Don't cache y if not overwritten
    if !(typeof(x) <: EnzymeCore.Const) && !(typeof(y) <: EnzymeCore.Const)
        if !EnzymeRules.overwritten(config)[2]
            cache_y = y.val
        end
    end

    # Don't cache x if not overwritten
    if !(typeof(x) <: EnzymeCore.Const) && !(typeof(y) <: EnzymeCore.Const)
        if !EnzymeRules.overwritten(config)[3]
            cache_x = x.val
        end
    end

    dys = y.dval
    dxs = (typeof(x) <: EnzymeCore.Const) ? dys : x.dval

    if EnzymeRules.width(config) == 1
        dys = (dys,)
        dxs = (dxs,)
    end

    for (dy, dx) in zip(dys, dxs)
        if !(typeof(y) <: EnzymeCore.Const) && dy !== y.val

            if !(typeof(x) <: EnzymeCore.Const) && dx !== x.val
                NNlib.$(∇pool)(dx, dy, cache_y, cache_x, dims.val; alpha=eltype(dx)(1), beta=eltype(dx)(1), kwargs...)
            end

            dy .= 0
        end
    end

    return (nothing, nothing, nothing)
end

end
end

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib._dropout!)}, ::Type{RT}, rng, dst::OutType, src, p, dims) where {OutType, RT}

    T = float(real(eltype(dst.val)))
    val = convert(T, 1/(1-p.val))
    keep = if dims.val isa Colon
        similar(dst.val, T, size(dst.val))
    else
        similar(dst.val, T, ntuple(d -> d in dims.val ? size(dst.val,d) : 1, ndims(dst.val)))
    end
    rand!(rng.val, keep)
    
    keep = keep .> p.val

    if OutType <: EnzymeCore.Duplicated || OutType <: EnzymeCore.BatchDuplicated
        dst.val .= (keep .* val) .* src.val
    end

    primal = if EnzymeRules.needs_primal(config)
        dst.val
    else
        nothing
    end
    shadow = if EnzymeRules.needs_shadow(config)
        dst.dval
    else
        nothing
    end

    if typeof(dst) <: EnzymeCore.Const || typeof(src) <: EnzymeCore.Const
        keep = nothing
    end

    return EnzymeRules.AugmentedReturn(primal, shadow, keep)
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib._dropout!)}, ::Type{RT}, keep, rng, dst::OutType, src, p, dims) where {OutType, RT}
    T = float(real(eltype(dst.val)))
    val = convert(T, 1/(1-p.val))

    ddsts = dst.dval
    dsrcs = (typeof(src) <: EnzymeCore.Const) ? ddsts : src.dval

    if EnzymeRules.width(config) == 1
        ddsts = (ddsts,)
        dsrcs = (dsrcs,)
    end

    for (ddst, dsrc) in zip(ddsts, dsrcs)
        if !(typeof(dst) <: EnzymeCore.Const) && ddst !== dst.val

            if !(typeof(src) <: EnzymeCore.Const) && dsrc !== src.val
                dsrc .+= (keep .* val) .* ddst
            end

            ddst .= 0
        end
    end

    dp = if typeof(p) <: EnzymeCore.Active
        typeof(p.val)(0)
    else
        nothing
    end

    return (nothing, nothing, nothing, dp, nothing)
end


end
