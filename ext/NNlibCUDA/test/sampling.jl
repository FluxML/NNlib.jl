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

        padding_mode = Val(:zeros)
        y_gpu = grid_sample(x_gpu, grid_gpu; padding_mode=padding_mode)
        @test x == collect(y_gpu)
        @test eltype(y_gpu) == T

        external_grad = CUDA.ones(T, size(y_gpu))
        ∇input, ∇grid = ∇grid_sample(external_grad, x_gpu, grid_gpu; padding_mode=padding_mode)
        @test x == collect(∇input)
        @test ∇grid_true == collect(∇grid)
        @test eltype(∇input) == T
        @test eltype(∇grid) == T

        padding_mode = Val(:border)
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
    for T in (Float32, Float64), padding_mode in (Val(:zeros), Val(:border))
        w, h, c, n = 16, 16, 2, 4
        input = rand(T, w, h, c, n)
        grid = zeros(T, 2, w, h, n)
        @inbounds for xi in 1:w, yi in 1:h, ni in 1:n
            grid[1, xi, yi, ni] = (xi / w) * 2 - 1 + 0.01
            grid[2, xi, yi, ni] = (yi / h) * 2 - 1
        end

        dx = CuArray(input)
        dg = CuArray(grid)
        out_grad_gpu = CUDA.ones(T, (w, h, c, n))
        out_grad = ones(T, (w, h, c, n))

        y_gpu = grid_sample(dx, dg; padding_mode=padding_mode)
        y = grid_sample(input, grid; padding_mode=padding_mode)
        @assert y ≈ collect(y_gpu)

        ∇input_gpu, ∇grid_gpu = ∇grid_sample(out_grad_gpu, dx, dg; padding_mode=padding_mode)
        ∇input, ∇grid = ∇grid_sample(out_grad, input, grid; padding_mode=padding_mode)
        @assert ∇input ≈ collect(∇input_gpu)
        @assert ∇grid ≈ collect(∇grid_gpu)
    end
end
