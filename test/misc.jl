@testset "pixel_shuffle" begin
    x = reshape(1:16, (2,2,4,1))
    y = pixel_shuffle(x, 2)
    @test size(x) == (4, 4, 1, 1)

    x = reshape(1:4*3*27*2, (4,3,27,2))
    y = pixel_shuffle(x, 3)
    @test size(x) == (12, 9, 3, 2)
end

