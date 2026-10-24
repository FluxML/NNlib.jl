# In 16 bits, the backward kernels of `upsample_linear` add with paired atomics (#854),
# and the source positions and weights are computed in Float32: they used to drift past
# 256 (BFloat16) or 2048 (Float16) pixels, and the linear ones to go out of bounds.
@testset "resizing in $T" for T in (Float16, BFloat16)
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

    short, long = randn(Float32, 600, 5, 2, 1), randn(Float32, 2999, 5, 2, 1)
    for (f, x) in ((x -> upsample_bicubic(x; size = (2999, 5)), short),
                   (x -> ∇upsample_bicubic(x; size = (600, 5)), long),
                   (x -> upsample_bicubic(x; size = (600, 5), antialias = true), long),
                   (x -> ∇upsample_bicubic(x; size = (2999, 5), antialias = true), short),
                   (x -> upsample_bilinear(x; size = (600, 5), align_corners = false, antialias = true), long))
        x16 = T.(cu(x))
        y = f(x16)
        @test eltype(y) == T
        @test Float32.(y) ≈ f(Float32.(x16)) rtol = 2e-2
    end
end
