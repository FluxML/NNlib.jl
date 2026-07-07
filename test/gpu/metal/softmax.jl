using NNlib: ∇logsoftmax, ∇logsoftmax!, ∇softmax, ∇softmax!, logsoftmax, logsoftmax!,
             softmax, softmax!

@testset "softmax" begin
    for (T, atol) in ((Float16, 1f-2), (Float32, 1f-5))
        @testset "$T dims=$dims" for dims in (1, 2, 3)
            x = randn(T, 5, 4, 3)
            dy = randn(T, size(x))
            gx = DEVICE(x)
            gdy = DEVICE(dy)

            y = softmax(x; dims)
            gy = softmax(gx; dims)
            @test Array(gy) ≈ y atol=atol

            out = similar(gx)
            @test Array(softmax!(out, gx; dims)) ≈ y atol=atol

            dx = ∇softmax(dy, y; dims)
            gdx = ∇softmax(gdy, gy; dims)
            @test Array(gdx) ≈ dx atol=atol
            @test Array(∇softmax!(similar(gx), gdy, gy; dims)) ≈ dx atol=atol

            y = logsoftmax(x; dims)
            gy = logsoftmax(gx; dims)
            @test Array(gy) ≈ y atol=atol
            @test Array(logsoftmax!(out, gx; dims)) ≈ y atol=atol

            dx = ∇logsoftmax(dy, y; dims)
            gdx = ∇logsoftmax(gdy, gy; dims)
            @test Array(gdx) ≈ dx atol=atol
            @test Array(∇logsoftmax!(similar(gx), gdy, gy; dims)) ≈ dx atol=atol
        end
    end
end

@testset "softmax fallback dims" begin
    x = randn(Float32, 3, 4, 2)
    gx = DEVICE(x)
    @test Array(softmax(gx; dims = :)) ≈ softmax(x; dims = :) atol=1f-5
    @test Array(logsoftmax(gx; dims = (1, 2))) ≈ logsoftmax(x; dims = (1, 2)) atol=1f-5
end
