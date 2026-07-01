module NNlibEnzymeCoreCUDNNExt

# Enzyme reverse-mode rule for the cuDNN activation forward.
#
# On a `CuArray`, broadcasting `tanh`/`σ`/`elu`/`relu` is routed to cuDNN via
# `Base.materialize` overrides in `NNlibCUDACUDNNExt` (see `ext/.../activations.jl`),
# which call `cuDNN.cudnnActivationForward!(y, x; mode)`. Enzyme cannot differentiate
# the cuDNN `ccall`s underneath (it aborts with `unsupported tag gc-transition`, both
# on the activation forward and on the activation-descriptor creation). The rule below
# intercepts the whole `cudnnActivationForward!` call so Enzyme never descends into
# those `ccall`s, and hands the reverse pass to `cudnnActivationBackward` — rebuilding
# the same activation descriptor from `mode`/`coef` — so the gradient is exact.

import EnzymeCore
using EnzymeCore.EnzymeRules
using cuDNN: cuDNN, cudnnActivationForward!, cudnnActivationBackward,
             cudnnActivationDescriptor, cudnnTensorDescriptor, scalingParameter, handle,
             CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN

function EnzymeRules.augmented_primal(config,
        func::EnzymeCore.Const{typeof(cudnnActivationForward!)}, ::Type{RT},
        y, x; mode=CUDNN_ACTIVATION_RELU, nanOpt=CUDNN_NOT_PROPAGATE_NAN, coef=1, kw...) where {RT}

    yv = func.val(y.val, x.val; mode, nanOpt, coef, kw...)

    # cuDNN's activation backward needs both the input `x` and output `y`, plus the
    # activation descriptor (rebuilt from `mode`/`nanOpt`/`coef` in `reverse`).
    cache = (!(x isa EnzymeCore.Const) && !(y isa EnzymeCore.Const)) ?
            (copy(x.val), copy(y.val), mode, nanOpt, coef) : nothing

    primal = EnzymeRules.needs_primal(config) ? yv : nothing
    shadow = EnzymeRules.needs_shadow(config) ? y.dval : nothing   # return aliases `y`
    return EnzymeRules.AugmentedReturn(primal, shadow, cache)
end

function EnzymeRules.reverse(config,
        func::EnzymeCore.Const{typeof(cudnnActivationForward!)}, ::Type{RT}, cache,
        y, x; kw...) where {RT}

    if cache !== nothing
        cache_x, cache_y, mode, nanOpt, coef = cache
        T = eltype(cache_x)
        d = cudnnActivationDescriptor(mode, nanOpt, Cdouble(coef))
        xDesc, yDesc = cudnnTensorDescriptor(cache_x), cudnnTensorDescriptor(cache_y)
        a = scalingParameter(T, 1)
        b = scalingParameter(T, 1)   # beta != 0 => accumulate into the input shadow

        dys = y.dval
        dxs = x.dval
        if EnzymeRules.width(config) == 1
            dys = (dys,)
            dxs = (dxs,)
        end

        for (dy, dx) in zip(dys, dxs)
            cudnnActivationBackward(handle(), d, a, yDesc, cache_y, yDesc, dy,
                                    xDesc, cache_x, b, xDesc, dx)
            dy .= 0
        end
    end

    return (nothing, nothing)
end

end # module
