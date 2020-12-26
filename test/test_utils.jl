"""
Compare numerical gradient and automatic gradient
given by Zygote. `f` has to be a scalar valued function.

Use `ChainRulesTestUtils.rrule_test` instead 
if the rrule is explicitly defined.
"""
function zygote_gradient_test(f, xs...; atol=1e-9, rtol=1e-9)
    y_true = f(xs...)
    fdm = FiniteDifferences.central_fdm(5, 1)
    gs_fd = FiniteDifferences.grad(fdm, f, xs...) 
    
    y_ad, pull = Zygote.pullback(f, xs...)
    gs_ad = pull(one(y_ad))
    
    @test y_true ≈ y_ad  atol=atol rtol=rtol
    for (g_ad, g_fd) in zip(gs_ad, gs_fd)
        @test g_ad ≈ g_fd   atol=atol rtol=rtol
    end
end
