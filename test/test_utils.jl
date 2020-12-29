using Random
const IntOrTuple = Union{Int, NTuple{N,Int} where N}

gradtest(f, dims::IntOrTuple...; kw...) = 
    gradtest(f, rand.(rng, Float64, dims)...; kw...)

"""
Compare numerical gradient and automatic gradient
given by Zygote. `f` has to be a scalar valued function.

Applies also `ChainRulesTestUtils.rrule_test` if the rrule for `f` is explicitly defined.
"""
function gradtest(f, xs...; atol=1e-6, rtol=1e-6, fkwargs=(;), 
                    check_rrule=false, 
                    check_broadcast=false,
                    broken=false)
    
    if check_rrule
        y = f(xs...; fkwargs...)
        simil(x) = x isa Number ? rand(rng, typeof(x)) : rand!(rng, similar(x)) 
        ȳ =  simil(y)
        xx̄s = [(x, simil(x)) for x in xs]
        rrule_test(f, ȳ, xx̄s...; fkwargs=fkwargs)
    end

    h = check_broadcast ? 
        (xs...) -> sum(sin.(f.(xs...; fkwargs...))) :
        (xs...) -> sum(sin.(f(xs...; fkwargs...)))
    
    y_true = h(xs...)
    fdm = FiniteDifferences.central_fdm(5, 1)
    gs_fd = FiniteDifferences.grad(fdm, h, xs...) 
    
    y_ad, pull = Zygote.pullback(h, xs...)
    gs_ad = pull(one(y_ad))
    
    @test y_true ≈ y_ad  atol=atol rtol=rtol
    for (g_ad, g_fd) in zip(gs_ad, gs_fd)
        if broken
            @test_broken g_ad ≈ g_fd   atol=atol rtol=rtol
        else
            @test g_ad ≈ g_fd   atol=atol rtol=rtol
        end
    end
    return true
end
