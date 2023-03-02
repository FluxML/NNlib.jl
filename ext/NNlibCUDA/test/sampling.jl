@testset "Grid Sampling" begin
    for T in (Float32, Float64)
        x = ones(T, (2, 2, 1, 1))
        grid = Array{T}(undef, 2, 2, 2, 1)
        grid[:, 1, 1, 1] .= (-1, -1)
        grid[:, 2, 1, 1] .= (1, -1)
        grid[:, 1, 2, 1] .= (-1, 1)
        grid[:, 2, 2, 1] .= (1, 1)

        ∇grid_true = Array{T}(undef, size(grid))
        ∇grid_true[:, :, 1, 1] = [[0.0, 0.0] [-0.5, 0.0]]
        ∇grid_true[:, :, 2, 1] = [[0.0, -0.5] [-0.5, -0.5]]

        x_gpu, grid_gpu = CuArray(x), CuArray(grid)

        padding_mode = :zeros
        y_gpu = grid_sample(x_gpu, grid_gpu; padding_mode=padding_mode)
        @test x == collect(y_gpu)
        @test eltype(y_gpu) == T

        external_grad = CUDA.ones(T, size(y_gpu))
        ∇input, ∇grid = ∇grid_sample(external_grad, x_gpu, grid_gpu; padding_mode=padding_mode)
        @test x == collect(∇input)
        @test ∇grid_true == collect(∇grid)
        @test eltype(∇input) == T
        @test eltype(∇grid) == T

        padding_mode = :border
        fill!(∇grid_true, 0.0)
        sampled = grid_sample(x_gpu, grid_gpu; padding_mode=padding_mode)
        @test x == collect(sampled)
        @test eltype(sampled) == T

        ∇input, ∇grid = ∇grid_sample(external_grad, x_gpu, grid_gpu; padding_mode=padding_mode)
        @test x == collect(∇input)
        @test ∇grid_true == collect(∇grid)
        @test eltype(∇input) == T
        @test eltype(∇grid) == T
    end
end

@testset "Compare grid sampling with NNlib" begin
    w, h, c, n = 16, 16, 2, 4
    input = rand(Float64, w, h, c, n)
    grid = zeros(Float64, 2, w, h, n)
    @inbounds for xi in 1:w, yi in 1:h, ni in 1:n
        grid[1, xi, yi, ni] = (xi / w) * 2.0 - 1.0 + 0.01
        grid[2, xi, yi, ni] = (yi / h) * 2.0 - 1.0
    end
    for padding_mode in (:zeros, :border)
        gputest(grid_sample, input, grid; atol=1e-6, padding_mode=padding_mode)
    end
end
