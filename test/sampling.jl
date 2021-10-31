@testset "Grid Sampling" begin
    for T in (Float16, Float32, Float64)
        x = ones(T, (2, 2, 1, 1))
        grid = Array{T}(undef, 2, 2, 2, 1)
        grid[:, 1, 1, 1] .= (-1, -1)
        grid[:, 2, 1, 1] .= (1, -1)
        grid[:, 1, 2, 1] .= (-1, 1)
        grid[:, 2, 2, 1] .= (1, 1)

        padding_mode = Val(:zeros)
        sampled = NNlib.grid_sampler(x, grid, padding_mode)
        @test x == sampled
        @test eltype(sampled) == T

        external_grad = ones(size(sampled))
        ∇input, ∇grid = NNlib.∇grid_sampler(external_grad, x, grid, padding_mode)
        @test ∇input == x
        @test eltype(∇input) == T
        @test eltype(∇grid) == T
    end
end
