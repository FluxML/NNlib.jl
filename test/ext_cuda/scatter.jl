dsts = Dict(
    0 => cu([3, 4, 5, 6, 7]),
    1 => cu([3 3 4 4 5;
        5 5 6 6 7]),
)
srcs = Dict(
    (0, true) => cu(ones(Int, 3, 4)),
    (0, false) => cu(ones(Int, 3) * collect(1:4)'),
    (1, true) => cu(ones(Int, 2, 3, 4)),
    (1, false) => cu([1, 2] .* reshape(ones(Int, 3) * collect(1:4)', 1, 3, 4)),
)
idxs = [
    cu([1 2 3 4;
        4 2 1 3;
        3 5 5 3]),  # integer index
    cu([(1,) (2,) (3,) (4,);
        (4,) (2,) (1,) (3,);
        (3,) (5,) (5,) (3,)]),  # tuple index
    cu(CartesianIndex.([(1,) (2,) (3,) (4,);
        (4,) (2,) (1,) (3,);
        (3,) (5,) (5,) (3,)])),  # CartesianIndex index
]

types = [CuArray{Int32}, CuArray{Int64}, CuArray{Float32}, CuArray{Float64}, CuSparseMatrixCSC{Float32}, CuSparseMatrixCSR{Float32}, CuSparseMatrixCOO{Float32}]


@testset "scatter" begin
    for T = types
        @testset "$(T)" begin
            @testset "+" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(+, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad=true)

                    mutated = false
                    gputest(src -> NNlib.scatter(+, src, idx), T(srcs[(dims, mutated)]), checkgrad=true)
                end
            end

            @testset "-" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(-, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad=true)

                    mutated = false
                    gputest(src -> NNlib.scatter(-, src, idx), T(srcs[(dims, mutated)]), checkgrad=true)
                end
            end

            @testset "max" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(max, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad=true)

                    mutated = false
                    gputest(src -> NNlib.scatter(max, src, idx), T(srcs[(dims, mutated)]), checkgrad=true)
                end
            end

            @testset "min" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(min, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad=true)

                    mutated = false
                    gputest(src -> NNlib.scatter(min, src, idx), T(srcs[(dims, mutated)]), checkgrad=true)
                end
            end
        end
    end


    # Specialized sparse scatter kernels. Duplicated as test cases above do not cover sparse arrays.
    dsts_sp = Dict(
        0 => cu(sparse([3, 4, 5, 6, 7])),
        1 => cu(sparse([3 3 4 4 5;
            5 5 6 6 7])),
    )
    srcs_sp = Dict(
        (0, true) => cu(sparse(ones(Int, 3, 4))),
        (0, false) => cu(sparse(ones(Int, 3) * collect(1:4)')),
        # No sparse equivalent for 3D arrays
    )
    types_sp = [
        CuSparseMatrixCSC{Int32}, CuSparseMatrixCSC{Int64}, CuSparseMatrixCSC{Float32}, CuSparseMatrixCSC{Float64},
        CuSparseMatrixCSR{Int32}, CuSparseMatrixCSR{Int64}, CuSparseMatrixCSR{Float32}, CuSparseMatrixCSR{Float64},
        CuSparseMatrixCOO{Int32}, CuSparseMatrixCOO{Int64}, CuSparseMatrixCOO{Float32}, CuSparseMatrixCOO{Float64}
    ]

    @testset "scatter sparse-specialized" begin
        for T = types_sp
            @testset "$(T)" begin
                @testset "+" begin
                    # Dims is implicitly 0. No sparse equivant for multidimensional src/dst
                    for idx = idxs
                        mutated = true
                        gputest((dst, src) -> NNlib.scatter!(+, dst, src, idx), T(copy(dsts[0])), T(srcs[(0, mutated)]), checkgrad=true)

                        mutated = false
                        gputest(src -> NNlib.scatter(+, src, idx), T(srcs[(0, mutated)]), checkgrad=true)
                    end
                end

                @testset "-" begin
                    for idx = idxs
                        mutated = true
                        gputest((dst, src) -> NNlib.scatter!(-, dst, src, idx), T(copy(dsts[0])), T(srcs[(0, mutated)]), checkgrad=true)

                        mutated = false
                        gputest(src -> NNlib.scatter(-, src, idx), T(srcs[(0, mutated)]), checkgrad=true)
                    end
                end

                @testset "max" begin
                    for idx = idxs
                        mutated = true
                        gputest((dst, src) -> NNlib.scatter!(max, dst, src, idx), T(copy(dsts[0])), T(srcs[(0, mutated)]), checkgrad=true)

                        mutated = false
                        gputest(src -> NNlib.scatter(max, src, idx), T(srcs[(0, mutated)]), checkgrad=true)
                    end
                end

                @testset "min" begin
                    for idx = idxs
                        mutated = true
                        gputest((dst, src) -> NNlib.scatter!(min, dst, src, idx), T(copy(dsts[0])), T(srcs[(0, mutated)]), checkgrad=true)

                        mutated = false
                        gputest(src -> NNlib.scatter(min, src, idx), T(srcs[(0, mutated)]), checkgrad=true)
                    end
                end
            end
        end

        # Sparse-specialized for operations not tested on eltype <: Integer
        for T = [CuSparseMatrixCSC{Float32}, CuSparseMatrixCSC{Float64}, CuSparseMatrixCSR{Float32}, CuSparseMatrixCSR{Float64}, CuSparseMatrixCOO{Float32}, CuSparseMatrixCOO{Float64}]
            @testset "$(T)" begin
                # Dims is implicitly 0. No sparse equivant for multidimensional src/dst
                @testset "*" begin
                    for idx = idxs
                        mutated = true
                        gputest((dst, src) -> NNlib.scatter!(*, dst, src, idx), T(copy(dsts[0])), T(srcs[(0, mutated)]), checkgrad=true)

                        mutated = false
                        gputest(src -> NNlib.scatter(*, src, idx), T(srcs[(0, mutated)]), checkgrad=true)
                    end
                end

                @testset "/" begin
                    for idx = idxs, dims = [0, 1]
                        mutated = true
                        gputest((dst, src) -> NNlib.scatter!(/, dst, src, idx), T(copy(dsts[0])), T(srcs[(0, mutated)]), checkgrad=true)

                        mutated = false
                        gputest(src -> NNlib.scatter(/, src, idx), T(srcs[(0, mutated)]), checkgrad=true)
                    end
                end

                @testset "mean" begin
                    for idx = idxs, dims = [0, 1]
                        mutated = true
                        gputest((dst, src) -> NNlib.scatter!(mean, dst, src, idx), T(copy(dsts[0])), T(srcs[(0, mutated)]), checkgrad=true)

                        mutated = false
                        gputest(src -> NNlib.scatter(mean, src, idx), T(srcs[(0, mutated)]), checkgrad=true)
                    end
                end
            end
        end
    end



    for T = [CuArray{Float32}, CuArray{Float64}]
        @testset "$(T)" begin
            @testset "*" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(*, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad=true)

                    mutated = false
                    gputest(src -> NNlib.scatter(*, src, idx), T(srcs[(dims, mutated)]), checkgrad=true)
                end
            end

            @testset "/" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(/, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad=true)

                    mutated = false
                    gputest(src -> NNlib.scatter(/, src, idx), T(srcs[(dims, mutated)]), checkgrad=true)
                end
            end

            @testset "mean" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(mean, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad=true)

                    mutated = false
                    gputest(src -> NNlib.scatter(mean, src, idx), T(srcs[(dims, mutated)]), checkgrad=true)
                end
            end
        end
    end
end
