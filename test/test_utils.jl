"""
Compare numerical and automatic gradient.
`f` has to be a scalar valued function.

Use `ChainRulesTestUtils.rrule_test` instead 
if the rrule is explicitly defined.
"""
function autodiff_test(f, xs...; atol=1e-9, rtol=1e-9)
    fdm = FiniteDifferences.central_fdm(5, 1)
    gs_ad = Zygote.gradient(f, xs...)
    gs_fd = FiniteDifferences.grad(fdm, f, xs...) 
    for (g_ad, g_fd) in zip(gs_ad, gs_fd)
        @test g_ad ≈ g_fd   atol=atol rtol=rtol
    end
end