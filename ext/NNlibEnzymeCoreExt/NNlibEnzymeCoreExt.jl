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


# batched_mul
#
# Without a custom rule, Enzyme differentiates through NNlib's threaded
# `batched_gemm!`, whose `Threads.@spawn`/`Threads.@sync` is not supported and
# (on Julia 1.12) hits an internal `cmpxchg` error in `wait(::Task)`.
# See https://github.com/FluxML/NNlib.jl/issues/707 and
# https://github.com/EnzymeAD/Enzyme.jl/issues/3150.
#
# The derivatives mirror the ChainRules `rrule` in src/batched/batchedmul.jl:
#   dA = Δ ⊠ Bᴴ   (summed over the batch dim if `size(A,3) == 1`)
#   dB = Aᴴ ⊠ Δ   (summed over the batch dim if `size(B,3) == 1`)

@inline _batched_mul_const(x) = x isa EnzymeCore.Const

function EnzymeRules.forward(config::EnzymeRules.FwdConfig,
        func::EnzymeCore.Const{typeof(NNlib.batched_mul)}, ::Type{RT},
        A::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}},
        B::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}}) where {RT}

    bothconst = _batched_mul_const(A) && _batched_mul_const(B)

    # The primal is needed if requested, or to size the zero tangent when both
    # arguments are Const but a shadow is still required (e.g. runtime activity).
    primal = (EnzymeRules.needs_primal(config) ||
              (EnzymeRules.needs_shadow(config) && bothconst)) ?
             func.val(A.val, B.val) : nothing

    EnzymeRules.needs_shadow(config) || return primal

    # dC = dA ⊠ B + A ⊠ dB (a missing term means that argument is Const)
    dC(dA, dB) =
        if bothconst
            zero(primal)
        elseif _batched_mul_const(A)
            NNlib.batched_mul(A.val, dB)
        elseif _batched_mul_const(B)
            NNlib.batched_mul(dA, B.val)
        else
            NNlib.batched_mul(dA, B.val) .+ NNlib.batched_mul(A.val, dB)
        end

    shadow = if EnzymeRules.width(config) == 1
        dC(_batched_mul_const(A) ? nothing : A.dval,
           _batched_mul_const(B) ? nothing : B.dval)
    else
        ntuple(i -> dC(_batched_mul_const(A) ? nothing : A.dval[i],
                       _batched_mul_const(B) ? nothing : B.dval[i]),
               Val(EnzymeRules.width(config)))
    end

    EnzymeRules.needs_primal(config) || return shadow
    return EnzymeRules.width(config) == 1 ?
        EnzymeCore.Duplicated(primal, shadow) :
        EnzymeCore.BatchDuplicated(primal, shadow)
end

function EnzymeRules.augmented_primal(config::EnzymeRules.RevConfig,
        func::EnzymeCore.Const{typeof(NNlib.batched_mul)}, ::Type{RT},
        A::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}},
        B::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}}) where {RT}

    C = func.val(A.val, B.val)

    primal = EnzymeRules.needs_primal(config) ? C : nothing
    shadow = if EnzymeRules.needs_shadow(config)
        EnzymeRules.width(config) == 1 ? zero(C) :
            ntuple(_ -> zero(C), Val(EnzymeRules.width(config)))
    else
        nothing
    end

    # Cache A if it's overwritten and needed for dB (i.e. B is active),
    # cache B if it's overwritten and needed for dA (i.e. A is active).
    cache_A = ( EnzymeRules.overwritten(config)[2]
                && !_batched_mul_const(B) ) ? copy(A.val) : nothing
    cache_B = ( EnzymeRules.overwritten(config)[3]
                && !_batched_mul_const(A) ) ? copy(B.val) : nothing

    return EnzymeRules.AugmentedReturn(primal, shadow, (shadow, cache_A, cache_B))
end

function EnzymeRules.reverse(config::EnzymeRules.RevConfig,
        func::EnzymeCore.Const{typeof(NNlib.batched_mul)}, ::Type{RT}, tape,
        A::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}},
        B::EnzymeCore.Annotation{<:AbstractArray{<:Any,3}}) where {RT}

    dCs, cache_A, cache_B = tape

    # Nothing to propagate if the return wasn't differentiated.
    dCs === nothing && return (nothing, nothing)

    # Recover values not cached because they were not overwritten.
    if !_batched_mul_const(B) && cache_A === nothing
        cache_A = A.val
    end
    if !_batched_mul_const(A) && cache_B === nothing
        cache_B = B.val
    end

    dAs = _batched_mul_const(A) ? dCs : A.dval
    dBs = _batched_mul_const(B) ? dCs : B.dval

    if EnzymeRules.width(config) == 1
        dCs = (dCs,)
        dAs = (dAs,)
        dBs = (dBs,)
    end

    for (dC, dA, dB) in zip(dCs, dAs, dBs)
        if !_batched_mul_const(A)
            tmp = NNlib.batched_mul(dC, NNlib.batched_adjoint(cache_B))
            dA .+= size(A.val, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
        if !_batched_mul_const(B)
            tmp = NNlib.batched_mul(NNlib.batched_adjoint(cache_A), dC)
            dB .+= size(B.val, 3) == 1 ? sum(tmp; dims=3) : tmp
        end
    end

    return (nothing, nothing)
end


end
