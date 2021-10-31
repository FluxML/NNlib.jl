@testset "Grid Sampling" begin
    x = ones(Float64, (2, 2, 1, 1))
    grid = Array{Float64}(undef, 2, 2, 2, 1)
    grid[:, 1, 1, 1] .= (-1, -1)
    grid[:, 2, 1, 1] .= (1, -1)
    grid[:, 1, 2, 1] .= (-1, 1)
    grid[:, 2, 2, 1] .= (1, 1)

    padding_mode = 0
    sampled = NNlib.grid_sampler(x, grid, padding_mode)
    @test x == sampled

    external_grad = ones(size(sampled))
    ∇input, ∇grid = NNlib.∇grid_sampler(external_grad, x, grid, padding_mode)
    @test ∇input == x
end
