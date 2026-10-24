dsts = Dict(
    0 => cu([3, 4, 5, 6, 7]),
    1 => cu([3 3 4 4 5;
             5 5 6 6 7]),
)
srcs = Dict(
    (0, true) => cu(ones(Int, 3, 4)),
    (0, false) => cu(ones(Int, 3) * collect(1:4)'),
    (1, true) => cu(ones(Int, 2, 3, 4)),
    (1, false) => cu([1, 2] .* reshape(ones(Int, 3) * collect(1:4)', 1,3,4)),
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

types = [CuArray{Int32}, CuArray{Int64}, CuArray{Float32}, CuArray{Float64}]


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

    @testset "index on CPU, destination on GPU (#415)" begin
        # A CPU index with a GPU destination/source must run on the GPU (not fall back
        # to slow scalar indexing) and match an all-GPU call.
        idx_cpu = [1 2 3 4; 4 2 1 3; 3 5 5 3]
        idx_gpu = cu(idx_cpu)
        dst = cu(Float32[3, 4, 5, 6, 7])
        src = cu(ones(Float32, 3, 4))
        @test Array(NNlib.scatter!(+, copy(dst), src, idx_cpu)) ==
              Array(NNlib.scatter!(+, copy(dst), src, idx_gpu))
        # `mean` exercises the dispatch path kept unambiguous from the `+` methods
        @test Array(NNlib.scatter!(mean, copy(dst), src, idx_cpu)) ≈
              Array(NNlib.scatter!(mean, copy(dst), src, idx_gpu))
    end

    @testset "out-of-bounds index (#416)" begin
        # Out-of-range indices must error cleanly instead of silently corrupting
        # memory in the @inbounds kernel, for indices on either the GPU or the CPU.
        @test_throws ArgumentError NNlib.scatter!(+, cu(Float32[1, 2, 3]), cu(Float32[1, 2]), cu([2, 9]))
        @test_throws ArgumentError NNlib.scatter!(+, cu(Float32[1, 2, 3]), cu(Float32[1, 2]), [2, 9])
        @test_throws ArgumentError NNlib.scatter!(+, CUDA.zeros(Float32, 2, 3), cu(Float32[1 2 3 4; 5 6 7 8]), cu([2, 1, 1, 9]))
    end

    # Paired 16-bit atomics. Sums of small integers are exact in 16 bits, and the
    # BFloat16 rounding happens on the GPU (see bfloat16.jl).
    @testset "$T with colliding indices (#854)" for T in (Float16, BFloat16)
        for n in (1, 2, 7, 8)  # the first and last elements are alone in their word or not
            idx = cu(repeat(1:n, 100))
            src = CUDA.ones(T, length(idx))
            @test Float32.(Array(NNlib.scatter(+, src, idx; dstsize = n))) == fill(100f0, n)
            @test Float32.(Array(NNlib.scatter(-, src, idx; dstsize = n))) == fill(-100f0, n)
        end
        i = repeat(1:5, 20)
        src = repeat(Float32[1, 2, 3], 1, 100)
        for idx in (cu(i), cu(Tuple.(i)), cu(CartesianIndex.(i)))
            @test Float32.(Array(NNlib.scatter(+, T.(cu(src)), idx))) == repeat(Float32[20, 40, 60], 1, 5)
            @test Float32.(Array(NNlib.scatter!(+, CUDA.zeros(T, 3, 5), cu(src), idx))) == repeat(Float32[20, 40, 60], 1, 5)
        end
        # a destination that starts halfway through a 32-bit word
        for off in (0, 1)
            buf = CUDA.fill(T(-0.0), 13)
            NNlib.scatter!(+, view(buf, (1 + off):(12 + off)), T.(cu(Float32[1, 2, 3, 4, 5])), cu([1, 3, 12, 12, 1]))
            expected = fill(-0f0, 13)
            expected[off .+ [1, 3, 12]] .= [6, 2, 7]
            @test Float32.(Array(buf)) == expected
            @test signbit.(Float32.(Array(buf))) == signbit.(expected)  # the -0.0 neighbours too
        end
    end
end
