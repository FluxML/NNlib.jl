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

const IntOrTuple = Union{Int, NTuple{N,Int} where N}

gradtest(f, dims::IntOrTuple...; kw...) =
    gradtest(f, randn.(Ref(rng), Float64, dims)...; kw...) # julia v1.3 compat
    # gradtest(f, randn.(rng, Float64, dims)...; kw...)

"""
Compare numerical gradient and automatic gradient
given by Zygote. `f` has to be a scalar valued function.

Applies also `ChainRulesTestUtils.test_rrule` if the rrule for `f` is explicitly defined.
"""
function gradtest(
    f, xs...; atol = 1e-6, rtol = 1e-6, fkwargs = NamedTuple(),
    check_rrule = false, fdm = :central, check_broadcast = false,
    skip = false, broken = false,
)
    if check_rrule
        test_rrule(f, xs...; fkwargs = fkwargs)
    end

    if check_broadcast
        length(fkwargs) > 0 && @warn("CHECK_BROADCAST: dropping keywords args")
        h = (xs...) -> sum(f.(xs...))
    else
        h = (xs...) -> sum(f(xs...; fkwargs...))
    end

    y_true = h(xs...)
    if fdm == :central
        fdm_obj = FiniteDifferences.central_fdm(5, 1)
    elseif fdm == :forward
        fdm_obj = FiniteDifferences.forward_fdm(5, 1)
    elseif fdm == :backward
        fdm_obj = FiniteDifferences.backward_fdm(5, 1)
    end
    # @show fdm fdm_obj

    gs_fd = FiniteDifferences.grad(fdm_obj, h, xs...)

    y_ad, pull = Zygote.pullback(h, xs...)
    gs_ad = pull(one(y_ad))

    @test y_true ≈ y_ad  atol = atol rtol = rtol
    for (g_ad, g_fd) in zip(gs_ad, gs_fd)
        if skip
            @test_skip g_ad ≈ g_fd   atol = atol rtol = rtol
        elseif broken
            @test_broken g_ad ≈ g_fd   atol = atol rtol = rtol
        else
            @test g_ad ≈ g_fd   atol = atol rtol = rtol
        end
    end
    return true
end

"""
    gpu_gradtest(f, xs...; checkgrad=true, atol=1e-6, kws...)

Compare `f`'s output and gradients on the device vs CPU. `xs...` should already
be on the device. Used by the shared `common_testsuite/` suites on GPU backends
(the per-backend `gpu/*/test_setup.jl` files define their own `gputest`, which
takes CPU inputs instead).
"""
function gpu_gradtest(f, xs...; checkgrad=true, atol=1e-6, kws...)
    cpu_xs = map(x -> adapt(CPU(), x), xs)

    cpu_y = f(cpu_xs...; kws...)
    y = f(xs...; kws...)
    @test collect(cpu_y) ≈ collect(y)

    if checkgrad
        cpu_grad = gradient((x...) -> sum(f(x...; kws...)), cpu_xs...)
        gpu_grad = gradient((x...) -> sum(f(x...; kws...)), xs...)

        for (cpu_g, gpu_g) in zip(cpu_grad, adapt(CPU(), gpu_grad))
            if cpu_g === nothing
                @test gpu_g === nothing
            else
                @test collect(cpu_g) ≈ collect(gpu_g) atol=atol
            end
        end
    end
end


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
            test_cpu = true,
            reference::AbstractADType = AutoFiniteDifferences(; fdm = _default_fdm()),
            compare::AbstractADType = AutoZygote(),
            loss = (f, xs...) -> mean(f(xs...)),
            )

    
    cpu_dev = cpu_device()
    
    if test_gpu
        gpu_dev = gpu_device(force=true)
        cpu_dev = cpu_device()
        xs_gpu = xs |> gpu_dev
        f_gpu = f |> gpu_dev
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
        check_equal_leaves(gs, gs2; rtol, atol)
    end

    if test_gpu
        l_gpu = loss(f_gpu, xs_gpu...)
        @assert l_gpu isa Number "gpu loss should return a number, got $(typeof(l_gpu))"

        y_gpu, gs_gpu = withgradient((xs...) -> loss(f_gpu, xs...), compare, xs_gpu...)
        @assert isapprox(l_gpu, y_gpu; rtol, atol) "gpu forward pass mismatch: $l_gpu ≉ $y_gpu"
        check_equal_leaves(gs, gs_gpu |> cpu_dev; rtol, atol)
    end

    return true
end

# Compares two gradient collections leaf-by-leaf and `@assert`s they agree. Since
# `test_gradients` only differentiates `f`'s inputs — assumed to be numbers or numerical
# arrays — the gradients line up one-to-one and we can compare them directly without
# traversing nested structures. Comparing with `≈` also makes wrapper types interoperate,
# e.g. a `Transpose` reference gradient vs a dense Zygote gradient. Returns `true` so
# callers can write `@test test_gradients(...)` (and `... broken=true` / `skip=true`).
function check_equal_leaves(a, b; rtol=1e-4, atol=1e-4)
    for (x, y) in zip(a, b)
        @assert isapprox(x, y; rtol, atol) "gradient mismatch: $x ≉ $y"
    end
    return true
end

# CPU adapter used by the shared `common_testsuite/` suites in place of `gradtest`: it
# runs `test_gradients` wrapped in `@test` and maps `gradtest`'s `fdm` symbol onto a
# finite-difference reference. The GPU branch of those suites keeps using `gpu_gradtest`.
function cpu_gradtest(f, xs...; fdm::Symbol = :central, kws...)
    fdm_obj = fdm === :forward  ? FiniteDifferences.forward_fdm(5, 1) :
              fdm === :backward ? FiniteDifferences.backward_fdm(5, 1) :
                                  _default_fdm()
    return @test test_gradients(f, xs...; reference = AutoFiniteDifferences(; fdm = fdm_obj), kws...)
end
