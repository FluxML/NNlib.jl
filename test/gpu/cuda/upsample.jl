# In 16 bits, the backward kernels add with paired atomics (#854), and the source
# positions are computed in Float32: they used to drift past 256 (BFloat16) or 2048
# (Float16) pixels, and to go out of bounds.
@testset "upsample_linear in $T" for T in (Float16, BFloat16)
    for (sz, outsz) in (((600, 3, 2), (2999,)), ((300, 40, 2, 2), (1499, 97)), ((300, 6, 5, 2, 1), (1499, 13, 11)))
        x = randn(Float32, sz)
        ref = upsample_linear(Float32.(T.(cu(x))); size = outsz)
        y = upsample_linear(T.(cu(x)); size = outsz)
        @test eltype(y) == T
        @test Float32.(y) ≈ ref rtol = 2e-2

        Δ = randn(Float32, size(ref))
        for align_corners in (true, false)
            ref = ∇upsample_linear(Float32.(T.(cu(Δ))); size = sz[1:end-2], align_corners)
            dx = ∇upsample_linear(T.(cu(Δ)); size = sz[1:end-2], align_corners)
            @test eltype(dx) == T
            @test Float32.(dx) ≈ ref rtol = 2e-2
        end
    end
end
