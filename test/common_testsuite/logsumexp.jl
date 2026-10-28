function logsumexp_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    types = nameof(Backend) === :MetalBackend ? (Float32, Float16) : (Float32, Float64, Float16)
    @testset "logsumexp ($T)" for T in types
        x = device(T.(randn(Float32, 7, 5, 3)))
        xc = Float64.(cpu(x))
        for dims in (:, 1, 2, 3, (1, 3), (2, 3))
            y = logsumexp(x; dims)
            ref = logsumexp(xc; dims)
            @test eltype(y) == (T == Float16 ? Float32 : T)
            @test cpu(y) ≈ ref rtol=1f-5
        end
    end
end
