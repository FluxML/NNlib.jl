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
    # TODO: revamp when https://github.com/JuliaDiff/ChainRulesTestUtils.jl/pull/166
    # is merged
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
    gputest(f, xs...; checkgrad=true, atol=1e-6, kws...)

Compare gradients computed on the device vs CPU.
`xs...` should already be on the device.
"""
function gputest(f, xs...; checkgrad=true, atol=1e-6, kws...)
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
