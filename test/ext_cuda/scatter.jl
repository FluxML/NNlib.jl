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
        (3,) (5,) (5,) (3,)
    ]),  # tuple index
    cu(CartesianIndex.([
                (1,) (2,) (3,) (4,);
                (4,) (2,) (1,) (3,);
                (3,) (5,) (5,) (3,)
            ])),  # CartesianIndex index
]

types = [CuArray{Int32}, CuArray{Int64}, CuArray{Float32}, CuArray{Float64}]

@testset "scatter" begin
    for T in types
        @testset "$(T)" begin
            @testset "+" begin
                for idx in idxs, dims in [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(+, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad = true)

                    mutated = false
                    gputest(src -> NNlib.scatter(+, src, idx), T(srcs[(dims, mutated)]), checkgrad = true)
                end
            end

            @testset "-" begin
                for idx in idxs, dims in [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(-, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad = true)

                    mutated = false
                    gputest(src -> NNlib.scatter(-, src, idx), T(srcs[(dims, mutated)]), checkgrad = true)
                end
            end

            @testset "max" begin
                for idx in idxs, dims in [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(max, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad = true)

                    mutated = false
                    gputest(src -> NNlib.scatter(max, src, idx), T(srcs[(dims, mutated)]), checkgrad = true)
                end
            end

            @testset "min" begin
                for idx in idxs, dims in [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(min, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad = true)

                    mutated = false
                    gputest(src -> NNlib.scatter(min, src, idx), T(srcs[(dims, mutated)]), checkgrad = true)
                end
            end
        end
    end


    # Specialized sparse scatter kernels. Duplicated as test cases above do not cover sparse arrays.
    srcs_sp(T, fmt) = sparse(cu(ones(T, 3, 4)), fmt = fmt)
    num_types = [Float32, Float64]

    @testset "scatter sparse-specialized" begin
        for T in num_types
            @testset "+" begin
                # Dims is implicitly 0. No sparse equivant for multidimensional src
                for idx in idxs, fmt in [:csc, :csr, :bsr, :coo]
                    # mutated = true
                    gputest((dst, src, idx) -> NNlib.scatter!(+, dst, src, idx), copy(dsts[0]), srcs_sp(T, fmt), idx)
                end
            end

            @testset "-" begin
                for idx in idxs, fmt in [:csc, :csr, :bsr, :coo]
                    # mutated = true
                    gputest((dst, src, idx) -> NNlib.scatter!(-, dst, src, idx), copy(dsts[0]), srcs_sp(T, fmt), idx)
                end
            end

            @testset "max" begin
                for idx in idxs, fmt in [:csc, :csr]
                    # mutated = true
                    gputest((dst, src, idx) -> NNlib.scatter!(max, dst, src, idx), copy(dsts[0]), srcs_sp(T, fmt), idx)
                end
            end

            @testset "min" begin
                for idx in idxs, fmt in [:csc, :csr]
                    # mutated = true
                    gputest((dst, src, idx) -> NNlib.scatter!(min, dst, src, idx), copy(dsts[0]), srcs_sp(T, fmt), idx)
                end
            end

            # Sparse-specialized for operations not tested on eltype <: Integer
            @testset "$(T)" begin
                # Dims is implicitly 0. No sparse equivant for multidimensional src/dst
                @testset "*" begin
                    for idx in idxs, fmt in [:csc, :csr, :bsr, :coo]
                        # mutated = true
                        gputest((dst, src, idx) -> NNlib.scatter!(*, dst, src, idx), copy(dsts[0]), srcs_sp(T, fmt), idx)
                    end
                end

                @testset "/" begin
                    for idx in idxs, fmt in [:csc, :csr]
                        # mutated = true
                        gputest((dst, src, idx) -> NNlib.scatter!(/, dst, src, idx), copy(dsts[0]), srcs_sp(T, fmt), idx)
                    end
                end

                @testset "mean" begin
                    for idx in idxs, fmt in [:csc, :csr]
                        # mutated = true
                        gputest((dst, src, idx) -> NNlib.scatter!(mean, dst, src, idx), copy(dsts[0]), srcs_sp(T, fmt), idx)
                    end
                end
            end
        end
    end


    for T in [CuArray{Float32}, CuArray{Float64}]
        @testset "$(T)" begin
            @testset "*" begin
                for idx in idxs, dims in [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(*, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad = true)

                    mutated = false
                    gputest(src -> NNlib.scatter(*, src, idx), T(srcs[(dims, mutated)]), checkgrad = true)
                end
            end

            @testset "/" begin
                for idx in idxs, dims in [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(/, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad = true)

                    mutated = false
                    gputest(src -> NNlib.scatter(/, src, idx), T(srcs[(dims, mutated)]), checkgrad = true)
                end
            end

            @testset "mean" begin
                for idx in idxs, dims in [0, 1]
                    mutated = true
                    gputest((dst, src) -> NNlib.scatter!(mean, dst, src, idx), T(copy(dsts[dims])), T(srcs[(dims, mutated)]), checkgrad = true)

                    mutated = false
                    gputest(src -> NNlib.scatter(mean, src, idx), T(srcs[(dims, mutated)]), checkgrad = true)
                end
            end
        end
    end
end
