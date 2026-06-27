# Shared setup loaded into every ParallelTestRunner worker via the `init_code`
# block in `runtests.jl`. This holds the common imports and helpers that the
# individual test files rely on (previously the preamble of `runtests.jl` plus
# `test_utils.jl`, which has been folded in here).
#
# Backend packages (CUDA/AMDGPU/Metal) are NOT loaded here: GPU workers get them
# either from the generated shared-suite entries or from the per-backend
# `ext_*/test_setup.jl` files (see `runtests.jl`).

using NNlib, Test, Statistics, Random
using ChainRulesCore, ChainRulesTestUtils
using Base.Broadcast: broadcasted
import EnzymeTestUtils
using EnzymeCore
import FiniteDifferences
import ForwardDiff
using Zygote: Zygote, gradient
using StableRNGs
using Adapt
using ImageTransformations
using Interpolations: Constant
using KernelAbstractions
using FFTW
import ReverseDiff as RD        # used in `pooling.jl`
using SpecialFunctions
using ADTypes
using Functors: Functors
using MLDataDevices: cpu_device, gpu_device
using Enzyme: Enzyme, Active, ReverseWithPrimal, EnzymeCore, Duplicated, Const
using Mooncake

const rng = StableRNG(123)

cpu(x) = cpu_device()(x)

### GRADIENTS

function withgradient(f::F, adtype::AutoZygote, x::Vararg{Any,N}) where {F,N}
    return Zygote.withgradient(f, x...)
end

_default_fdm() = FiniteDifferences.central_fdm(5, 1, max_range=1e-2)

function withgradient(f::F, adtype::AutoFiniteDifferences, x::Vararg{Any,N}) where {F, N}
    y = f(x...)
    gs = FiniteDifferences.grad(adtype.fdm, x -> f(x...), x)[1]
    return (; val = y, grad = gs)
end


function withgradient(f::F, adtype::AutoEnzyme, x::Vararg{Any,N}; zero::Bool=true) where {F,N}
    return _enzyme_withgradient(f, map(_trymake_duplicated, x)...; zero)
end

_trymake_duplicated(x::Duplicated) = x
_trymake_duplicated(x::Const) = x
_trymake_duplicated(x::Active) = x
# Scalars are immutable, so they can't accumulate gradient through a `Duplicated`
# shadow; Enzyme returns them as `Active` derivatives instead.
_trymake_duplicated(x::Number) = Active(x)
_trymake_duplicated(x) = Duplicated(x, EnzymeCore.make_zero(x))

function _enzyme_withgradient(f, args::Union{Const, Active, Duplicated}...; zero::Bool=true)
    for x in args
        zero && x isa Duplicated && EnzymeCore.remake_zero!(x.dval)
    end

    ad = Enzyme.set_runtime_activity(ReverseWithPrimal)
    # `autodiff` returns `((active_grads...,), primal)`; `Duplicated` grads land in
    # their shadow, `Active` grads come back positionally in the first tuple.
    derivs, result = Enzyme.autodiff(ad, Const(f), Active, args...)

    di = 0
    grad = map(args) do x
        x isa Active ? derivs[di += 1] : _grad_or_nothing(x)
    end
    return (; val = result, grad)
end

# This function strips the returned gradient to be Zygote-like:
_grad_or_nothing(dup::Duplicated) = Functors.fmapstructure(_grad_or_nothing, dup.dval; prune=nothing)
_grad_or_nothing(::Const) = nothing
_grad_or_nothing(x) = x

function withgradient(f::F, adtype::AutoMooncake, args::Vararg{Any,N}) where {F,N}
    config = Mooncake.Config(friendly_tangents=true)
    cache = Mooncake.prepare_gradient_cache(f, args...; config)
    val, grads = Mooncake.value_and_gradient!!(cache, f, args...)
    return (val=val, grad=grads[2:end])
end

# Convert the floating-point arrays in `x` to Float64 precision (a local stand-in
# for `Flux.f64`, so the tests don't depend on Flux). Recurses through Functors-
# compatible containers; non-float leaves (ints, functions, ...) are left as-is.
struct Float64Adaptor end
Adapt.adapt_storage(::Float64Adaptor, x::AbstractArray{<:AbstractFloat}) =
    convert(AbstractArray{Float64}, x)
Adapt.adapt_storage(::Float64Adaptor, x::AbstractArray{<:Complex{<:AbstractFloat}}) =
    convert(AbstractArray{Complex{Float64}}, x)
f64(x) = Functors.fmap(adapt(Float64Adaptor()), x)

function test_gradients(
            f,
            xs...;
            rtol=1e-4, atol=1e-4,
            test_gpu = false,
            # On a GPU backend we only check the GPU gradient against the reference; the
            # CPU-vs-reference comparison is already covered by the dedicated CPU run.
            test_cpu = !test_gpu,
            # Optional GPU-adapted version of `f`. Use when `f` captures CPU arrays
            # that must also be on GPU (e.g. index arrays in gather/scatter). When
            # `nothing`, `f |> gpu_dev` is attempted (works for closures that only
            # capture non-array scalars/config objects).
            f_gpu = nothing,
            reference::AbstractADType = test_cpu ? AutoFiniteDifferences(;fdm=_default_fdm()) : AutoZygote(),
            compare::AbstractADType = AutoZygote(),
            loss = (f, xs...) -> mean(f(xs...)),
            )

    @assert test_cpu || test_gpu "at least one of `test_cpu` or `test_gpu` must be true"

    cpu_dev = cpu_device()

    if test_gpu
        gpu_dev = gpu_device(force=true)
        cpu_dev = cpu_device()
        xs_gpu = xs |> gpu_dev
        _f_gpu = isnothing(f_gpu) ? (f |> gpu_dev) : f_gpu
    end

    ## Let's make sure first that the forward pass works.
    l = loss(f, xs...)
    @assert l isa Number "loss should return a number, got $(typeof(l))"

    # We only differentiate the inputs `xs`, not `f`: `f` is captured in the closures
    # below so its (possibly non-differentiable) configuration — e.g. `DenseConvDims`,
    # `PoolDims`, kwargs — is never perturbed by the AD/finite-difference backends.

    # Compute reference gradients with inputs promoted to f64 precision. `f` itself is
    # left untouched (we don't differentiate it, so it needn't be reconstructed by `f64`).
    y, gs = withgradient((xs...) -> loss(f, xs...), reference, f64(xs)...)
    @assert isapprox(l, y; rtol, atol) "forward pass mismatch: $l ≉ $y (reference)"

    if test_cpu
        y2, gs2 = withgradient((xs...) -> loss(f, xs...), compare, xs...)
        @assert isapprox(l, y2; rtol, atol) "forward pass mismatch: $l ≉ $y2 (compare)"
        check_equal(gs, gs2; rtol, atol)
    end

    if test_gpu
        l_gpu = loss(_f_gpu, xs_gpu...)
        @assert l_gpu isa Number "gpu loss should return a number, got $(typeof(l_gpu))"

        y_gpu, gs_gpu = withgradient((xs...) -> loss(_f_gpu, xs...), compare, xs_gpu...)
        @assert isapprox(l_gpu, y_gpu; rtol, atol) "gpu forward pass mismatch: $l_gpu ≉ $y_gpu"
        check_equal(gs, gs_gpu |> cpu_dev; rtol, atol)
    end

    return true
end

function check_equal(a, b; rtol=1e-4, atol=1e-4)
    for (x, y) in zip(a, b)
        @assert isapprox(x, y; rtol, atol) "gradient mismatch: $x ≉ $y"
    end
    return true
end
