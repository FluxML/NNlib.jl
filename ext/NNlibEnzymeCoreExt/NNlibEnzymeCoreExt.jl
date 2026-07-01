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

# `softmax!`/`logsoftmax!` gradients depend only on the output `y` (= the written
# destination) and the seed `dy`, via `∇softmax!(dx, dy, y; dims)`. Without a rule
# Enzyme would try to differentiate the underlying implementation directly — on a
# GPU that is a cuDNN `ccall` it cannot handle (Enzyme recurses/hangs), and even on
# the CPU a hand-off to `∇softmax!` matches the ChainRules `rrule`. The rule caches
# `y` in the augmented pass and accumulates `∇softmax!` into the input shadow on
# reverse. These are first-order rules (they use the fast, non-`within_gradient`
# `∇softmax!`), matching what `test_gradients` compares.
for (fwd!, bwd!) in ((:(NNlib.softmax!),    :(NNlib.∇softmax!)),
                     (:(NNlib.logsoftmax!), :(NNlib.∇logsoftmax!)))
    @eval begin

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof($fwd!)}, ::Type{RT},
                                      out::OutType, x; dims=1) where {OutType, RT}
    if OutType <: EnzymeCore.Duplicated || OutType <: EnzymeCore.BatchDuplicated
        func.val(out.val, x.val; dims)
    end

    primal = EnzymeRules.needs_primal(config) ? out.val : nothing
    shadow = EnzymeRules.needs_shadow(config) ? out.dval : nothing

    # Cache the output `y` (needed by `∇softmax!`); `out.val` may be overwritten
    # downstream, so copy it unless neither input nor output is differentiated.
    cache_y = ( !(typeof(x) <: EnzymeCore.Const) && !(typeof(out) <: EnzymeCore.Const)
              ) ? copy(out.val) : nothing

    return EnzymeRules.AugmentedReturn(primal, shadow, cache_y)
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof($fwd!)}, ::Type{RT}, cache_y,
                             out::OutType, x; dims=1) where {OutType, RT}
    if !(typeof(x) <: EnzymeCore.Const) && !(typeof(out) <: EnzymeCore.Const)
        dys = out.dval
        dxs = x.dval
        if EnzymeRules.width(config) == 1
            dys = (dys,)
            dxs = (dxs,)
        end

        for (dy, dx) in zip(dys, dxs)
            # `∇softmax!` overwrites its destination, so accumulate via a temporary.
            dx .+= $bwd!(similar(dx), dy, cache_y; dims)
            dy .= 0
        end
    end

    return (nothing, nothing)
end

    end
end

# ---------------------------------------------------------------------------
# Allocating ops whose gradient has a dedicated `∇` kernel: `imrotate`,
# `upsample_nearest`, `upsample_linear` (the latter also covers `upsample_bi/
# trilinear`, which forward to it). Enzyme cannot differentiate these forward
# kernels on the GPU ("Active kernel arguments not supported"), so we wrap them:
# the augmented pass allocates the output shadow, and the reverse pass feeds it to
# the `∇` kernel and accumulates into the input shadow. This mirrors each op's
# ChainRules `rrule`. (Written explicitly rather than via `Enzyme.@import_rrule`,
# per Enzyme maintainers' recommendation, and so the rules stay in the lighter
# EnzymeCore extension.)

# Allocate the return shadow (a zeroed copy of `y`), respecting the batch width.
_enz_shadow(config, y) =
    EnzymeRules.width(config) == 1 ? EnzymeCore.make_zero(y) :
        ntuple(_ -> EnzymeCore.make_zero(y), EnzymeRules.width(config))

# Pair each return-shadow (`dy`) with its input shadow (`dx`) across the width.
_enz_pairs(config, dy, dx) =
    EnzymeRules.width(config) == 1 ? ((dy, dx),) : zip(dy, dx)

# The i-th slice of a return-shadow / input-shadow across the batch width.
# `Val`-dispatched so the width-1 method never contains an array `getindex`:
# on a width-1 `Duplicated`, `.dval` is the array itself (not a tuple of arrays),
# and indexing it would scalar-index the GPU array (and poison broadcast typing).
_enz_slice(::Val{1}, s, i) = s
_enz_slice(::Val{W}, s, i) where {W} = s[i]

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.imrotate)},
        ::Type{RT}, arr, θ;
        method=:bilinear, rotation_center=size(arr.val) .÷ 2 .+ 1) where {RT}
    y = func.val(arr.val, θ.val; method, rotation_center)
    shadow = EnzymeRules.needs_shadow(config) ? _enz_shadow(config, y) : nothing
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    # `∇imrotate` reads only `arr`'s shape/backend (the op is linear in `arr`), so
    # keeping `arr.val` — not a copy — is safe even if it is later overwritten.
    return EnzymeRules.AugmentedReturn(primal, shadow, (shadow, arr.val, θ.val, method, rotation_center))
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.imrotate)},
        ::Type{RT}, cache, arr, θ; kwargs...) where {RT}
    shadow, arr_val, θ_val, method, rotation_center = cache
    if !(arr isa EnzymeCore.Const)
        for (dy, dx) in _enz_pairs(config, shadow, arr.dval)
            dx .+= NNlib.∇imrotate(dy, arr_val, θ_val; method, rotation_center)
        end
    end
    return (nothing, nothing)   # θ is `NoTangent` in the rrule
end

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.upsample_nearest)},
        ::Type{RT}, x, s) where {RT}
    y = func.val(x.val, s.val)
    shadow = EnzymeRules.needs_shadow(config) ? _enz_shadow(config, y) : nothing
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (shadow, s.val))
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.upsample_nearest)},
        ::Type{RT}, cache, x, s) where {RT}
    shadow, scales = cache
    if !(x isa EnzymeCore.Const)
        for (dy, dx) in _enz_pairs(config, shadow, x.dval)
            dx .+= NNlib.∇upsample_nearest(dy, scales)
        end
    end
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.upsample_linear)},
        ::Type{RT}, x; size, align_corners::Bool=true) where {RT}
    y = func.val(x.val; size, align_corners)
    shadow = EnzymeRules.needs_shadow(config) ? _enz_shadow(config, y) : nothing
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    # `∇upsample_linear` needs the original spatial size of `x` (see the rrule).
    insize = Base.size(x.val)[1:ndims(x.val)-2]
    return EnzymeRules.AugmentedReturn(primal, shadow, (shadow, insize, align_corners))
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.upsample_linear)},
        ::Type{RT}, cache, x; kwargs...) where {RT}
    shadow, insize, align_corners = cache
    if !(x isa EnzymeCore.Const)
        for (dy, dx) in _enz_pairs(config, shadow, x.dval)
            dx .+= NNlib.∇upsample_linear(dy; size=insize, align_corners)
        end
    end
    return (nothing,)
end

# `grid_sample(x, grid)` — allocating, differentiable in both `x` and `grid`;
# `∇grid_sample(dy, x, grid)` returns `(∇x, ∇grid)` and reads the input & grid
# values, so both are cached.
function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.grid_sample)},
        ::Type{RT}, x, grid; padding_mode=:zeros) where {RT}
    y = func.val(x.val, grid.val; padding_mode)
    shadow = EnzymeRules.needs_shadow(config) ? _enz_shadow(config, y) : nothing
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (shadow, copy(x.val), copy(grid.val), padding_mode))
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.grid_sample)},
        ::Type{RT}, cache, x, grid; kwargs...) where {RT}
    shadow, xval, gridval, padding_mode = cache
    if !(x isa EnzymeCore.Const) || !(grid isa EnzymeCore.Const)
        wv = Val(EnzymeRules.width(config))
        for i in 1:EnzymeRules.width(config)
            dy = _enz_slice(wv, shadow, i)
            ∇x, ∇grid = NNlib.∇grid_sample(dy, xval, gridval; padding_mode)
            x    isa EnzymeCore.Const || (_enz_slice(wv, x.dval, i)    .+= ∇x)
            grid isa EnzymeCore.Const || (_enz_slice(wv, grid.dval, i) .+= ∇grid)
        end
    end
    return (nothing, nothing)
end

# `ctc_loss(ŷ, y)` returns a scalar (Active return), differentiable in `ŷ` only
# (`y` is the target). We reuse `ctc_alpha` from the augmented pass and feed it to
# `∇ctc_loss` on reverse, scaled by the incoming cotangent `dval.val`.
function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.ctc_loss)},
        ::Type{RT}, ŷ, y) where {RT}
    tmp = NNlib.ctc_alpha(ŷ.val, y.val)
    primal = EnzymeRules.needs_primal(config) ? tmp.loss : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing, tmp)
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.ctc_loss)},
        dval::EnzymeCore.Active, tmp, ŷ, y) where {}
    if !(ŷ isa EnzymeCore.Const)
        grad = NNlib.∇ctc_loss(ŷ.val, y.val, tmp)
        wv = Val(EnzymeRules.width(config))
        for i in 1:EnzymeRules.width(config)
            Δ = _enz_slice(wv, dval.val, i)
            _enz_slice(wv, ŷ.dval, i) .+= Δ .* grad
        end
    end
    return (nothing, nothing)
end

# `batchnorm(g, b, x, running_mean, running_var, momentum)` — allocating, GPU-only
# (cuDNN); differentiable in `g`, `b`, `x`. `∇batchnorm` returns `(dg, db, dx)`
# (with `dg`/`db` possibly `nothing` when non-affine).
function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.batchnorm)},
        ::Type{RT}, g, b, x, running_mean, running_var, momentum; kwargs...) where {RT}
    y = func.val(g.val, b.val, x.val, running_mean.val, running_var.val, momentum.val; kwargs...)
    shadow = EnzymeRules.needs_shadow(config) ? _enz_shadow(config, y) : nothing
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    cache = (shadow, copy(g.val), copy(b.val), copy(x.val),
             running_mean.val, running_var.val, momentum.val)
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.batchnorm)},
        ::Type{RT}, cache, g, b, x, running_mean, running_var, momentum; kwargs...) where {RT}
    shadow, gval, bval, xval, rm, rv, mom = cache
    wv = Val(EnzymeRules.width(config))
    for i in 1:EnzymeRules.width(config)
        dy = _enz_slice(wv, shadow, i)
        dg, db, dx = NNlib.∇batchnorm(gval, bval, xval, dy, rm, rv, mom; kwargs...)
        (g isa EnzymeCore.Const || dg === nothing) || (_enz_slice(wv, g.dval, i) .+= dg)
        (b isa EnzymeCore.Const || db === nothing) || (_enz_slice(wv, b.dval, i) .+= db)
        x isa EnzymeCore.Const || (_enz_slice(wv, x.dval, i) .+= dx)
    end
    return (nothing, nothing, nothing, nothing, nothing, nothing)
end

# `unfold(x, cdims)` / `fold(y, output_size, cdims)` are adjoints of each other
# (the `unfold` pullback is `fold` and vice versa — see their rrules). Enzyme can't
# differentiate the underlying `unfold!`/`fold!` KA kernels (LLVM verifier crash /
# ReadOnlyMemoryError), so route the reverse pass through the sibling operator. The
# rules target the `DenseConvDims` methods; the `kernel_size` convenience methods
# forward to these, so Enzyme reaches the rule through them too.
function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.unfold)},
        ::Type{RT}, x, cdims::EnzymeCore.Const{<:NNlib.DenseConvDims}) where {RT}
    y = func.val(x.val, cdims.val)
    shadow = EnzymeRules.needs_shadow(config) ? _enz_shadow(config, y) : nothing
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (shadow, Base.size(x.val), cdims.val))
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.unfold)},
        ::Type{RT}, cache, x, cdims::EnzymeCore.Const{<:NNlib.DenseConvDims}) where {RT}
    shadow, xsize, cd = cache
    if !(x isa EnzymeCore.Const)
        wv = Val(EnzymeRules.width(config))
        for i in 1:EnzymeRules.width(config)
            dy = _enz_slice(wv, shadow, i)
            _enz_slice(wv, x.dval, i) .+= NNlib.fold(dy, xsize, cd)
        end
    end
    return (nothing, nothing)
end

function EnzymeRules.augmented_primal(config, func::EnzymeCore.Const{typeof(NNlib.fold)},
        ::Type{RT}, x, output_size, cdims::EnzymeCore.Const{<:NNlib.DenseConvDims}) where {RT}
    y = func.val(x.val, output_size.val, cdims.val)
    shadow = EnzymeRules.needs_shadow(config) ? _enz_shadow(config, y) : nothing
    primal = EnzymeRules.needs_primal(config) ? y : nothing
    return EnzymeRules.AugmentedReturn(primal, shadow, (shadow, cdims.val))
end

function EnzymeRules.reverse(config, func::EnzymeCore.Const{typeof(NNlib.fold)},
        ::Type{RT}, cache, x, output_size, cdims::EnzymeCore.Const{<:NNlib.DenseConvDims}) where {RT}
    shadow, cd = cache
    if !(x isa EnzymeCore.Const)
        wv = Val(EnzymeRules.width(config))
        for i in 1:EnzymeRules.width(config)
            dy = _enz_slice(wv, shadow, i)
            _enz_slice(wv, x.dval, i) .+= NNlib.unfold(dy, cd)
        end
    end
    return (nothing, nothing, nothing)
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
