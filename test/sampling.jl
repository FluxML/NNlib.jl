@testset "Known gradients" begin
    x = ones(Float64, (2, 2, 1, 1))
    grid = Array{Float64}(undef, 2, 2, 2, 1)
    grid[:, 1, 1, 1] .= (-1, -1)
    grid[:, 2, 1, 1] .= (1, -1)
    grid[:, 1, 2, 1] .= (-1, 1)
    grid[:, 2, 2, 1] .= (1, 1)

    ∇grid_true = Array{Float64}(undef, size(grid))
    ∇grid_true[:, :, 1, 1] = [[0.0, 0.0] [-0.5, 0.0]]
    ∇grid_true[:, :, 2, 1] = [[0.0, -0.5] [-0.5, -0.5]]

    padding_mode = :zeros
    sampled = grid_sample(x, grid; padding_mode=padding_mode)
    @test x == sampled
    @test eltype(sampled) == Float64
    external_grad = ones(size(sampled))
    ∇input, ∇grid = ∇grid_sample(external_grad, x, grid; padding_mode=padding_mode)
    @test ∇input == x
    @test ∇grid == ∇grid_true
    @test eltype(∇input) == Float64
    @test eltype(∇grid) == Float64

    # ∇grid from FiniteDifferences is incorrent in case when 0-padding.
    # gradtest(grid_sample, x, grid; fkwargs=(padding_mode=padding_mode,))

    padding_mode = :border
    fill!(∇grid_true, 0.0)
    sampled = grid_sample(x, grid; padding_mode=padding_mode)
    @test x == sampled
    @test eltype(sampled) == Float64
    external_grad = ones(size(sampled))
    ∇input, ∇grid = ∇grid_sample(external_grad, x, grid; padding_mode=padding_mode)
    @test ∇input == x
    @test ∇grid == ∇grid_true
    @test eltype(∇input) == Float64
    @test eltype(∇grid) == Float64

    gradtest(grid_sample, x, grid; fkwargs=(padding_mode=padding_mode,))
end

@testset "Test out-of-bounds for different paddings" begin
    x = ones(Float64, (2, 2, 1, 1))
    grid = Array{Float64}(undef, 2, 3, 2, 1)
    grid[:, 1, 1, 1] .= (-3, -1)
    grid[:, 2, 1, 1] .= (0, -1)
    grid[:, 3, 1, 1] .= (3, -1)
    grid[:, 1, 2, 1] .= (-1, 3)
    grid[:, 2, 2, 1] .= (0, 1)
    grid[:, 3, 2, 1] .= (1, 3)

    # With 0-padding, out-of-bound values are will contribute nothing to
    # the output values, because they are too far from any bound.
    y = grid_sample(x, grid; padding_mode=:zeros)
    y_true = reshape(Float64[[0, 1, 0] [0, 1, 0]], size(y))
    @test y_true == y

    # With border-padding, out-of-bound values simly become border values
    # and the result should be all ones.
    y = grid_sample(x, grid; padding_mode=:border)
    y_true = ones(Float64, size(y))
    @test y_true == y
end

@testset "Known gradients 3D" begin
    x = ones(Float64, (2, 2, 2, 1, 1))  # 3D input with depth=2
    grid = Array{Float64}(undef, 3, 2, 2, 2, 1)  # 3D grid with depth=2
    grid[:, 1, 1, 1, 1] .= (-1, -1, -1)
    grid[:, 2, 1, 1, 1] .= (1, -1, -1)
    grid[:, 1, 2, 1, 1] .= (-1, 1, -1)
    grid[:, 2, 2, 1, 1] .= (1, 1, -1)
    grid[:, 1, 1, 2, 1] .= (-1, -1, 1)
    grid[:, 2, 1, 2, 1] .= (1, -1, 1)
    grid[:, 1, 2, 2, 1] .= (-1, 1, 1)
    grid[:, 2, 2, 2, 1] .= (1, 1, 1)

    ∇grid_true = Array{Float64}(undef, size(grid))
    ∇grid_true[:, 1, 1, 1, 1] .= (0.0, 0.0, 0.0)
    ∇grid_true[:, 2, 1, 1, 1] .= (-0.5, 0.0, 0.0)
    ∇grid_true[:, 1, 2, 1, 1] .= (0.0, -0.5, 0.0)
    ∇grid_true[:, 2, 2, 1, 1] .= (-0.5, -0.5, 0.0)
    ∇grid_true[:, 1, 1, 2, 1] .= (0.0, 0.0, -0.5)
    ∇grid_true[:, 2, 1, 2, 1] .= (-0.5, 0.0, -0.5)
    ∇grid_true[:, 1, 2, 2, 1] .= (0.0, -0.5, -0.5)
    ∇grid_true[:, 2, 2, 2, 1] .= (-0.5, -0.5, -0.5)

    # ∇grid_true[:, :, :, 1, 1] = [
    #     [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
    #     [[0.0, -0.5, 0.0], [-0.5, -0.5, 0.0]]
    # ]
    # ∇grid_true[:, :, :, 2, 1] = [
    #     [[0.0, 0.0, -0.5], [-0.5, 0.0, -0.5]]
    #     [[0.0, -0.5, -0.5], [-0.5, -0.5, -0.5]]
    # ]

    padding_mode = :zeros
    sampled = grid_sample(x, grid; padding_mode=padding_mode)
    @test x == sampled
    @test eltype(sampled) == Float64
    external_grad = ones(size(sampled))
    ∇input, ∇grid = ∇grid_sample(external_grad, x, grid; padding_mode=padding_mode)
    @test ∇input == x
    @test ∇grid == ∇grid_true
    @test eltype(∇input) == Float64
    @test eltype(∇grid) == Float64

    # ∇grid from FiniteDifferences is incorrect in case when 0-padding.
    # gradtest(grid_sample, x, grid; fkwargs=(padding_mode=padding_mode,))

    padding_mode = :border
    fill!(∇grid_true, 0.0)
    sampled = grid_sample(x, grid; padding_mode=padding_mode)
    @test x == sampled
    @test eltype(sampled) == Float64
    external_grad = ones(size(sampled))
    ∇input, ∇grid = ∇grid_sample(external_grad, x, grid; padding_mode=padding_mode)
    @test ∇input == x
    @test ∇grid == ∇grid_true
    @test eltype(∇input) == Float64
    @test eltype(∇grid) == Float64

    gradtest(grid_sample, x, grid; fkwargs=(padding_mode=padding_mode,))
end

@testset "Test out-of-bounds for different paddings 3D" begin
    x = ones(Float64, (2, 2, 2, 1, 1))  # 3D input with depth=2
    grid = Array{Float64}(undef, 3, 2, 2, 2, 1)  # 3D grid with depth=2
    grid[:, 1, 1, 1, 1] .= (-3, -1, -1)
    grid[:, 2, 1, 1, 1] .= (0, -1, -1)
    grid[:, 1, 2, 1, 1] .= (-1, 3, -1)
    grid[:, 2, 2, 1, 1] .= (0, 1, -1)
    grid[:, 1, 1, 2, 1] .= (-1, -1, 3)
    grid[:, 2, 1, 2, 1] .= (0, -1, 3)
    grid[:, 1, 2, 2, 1] .= (-1, 1, 3)
    grid[:, 2, 2, 2, 1] .= (0, 1, 3)

    # With 0-padding, out-of-bound values will contribute nothing to
    # the output values, because they are too far from any bound.
    y = grid_sample(x, grid; padding_mode=:zeros)
    y_true = reshape(Float64[[0, 1] [0, 1] [0, 0] [0, 0]], size(y))
    @test y_true == y

    # With border-padding, out-of-bound values simply become border values
    # and the result should be all ones.
    y = grid_sample(x, grid; padding_mode=:border)
    y_true = ones(Float64, size(y))
    @test y_true == y
end
