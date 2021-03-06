@testset "Broadcast Fix" begin
    if CUDA.has_cudnn()
        @test testf(x -> logσ.(x), rand(5))
    end
end
