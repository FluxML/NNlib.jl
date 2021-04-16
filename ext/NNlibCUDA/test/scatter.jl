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
]
res = Dict(
    (+, 0, true) => cu([5, 6, 9, 8, 9]),
    (+, 1, true) => cu([5 5 8 6 7;
                        7 7 10 8 9]),
    (+, 0, false) => cu([4, 4, 12, 5, 5]),
    (+, 1, false) => cu([4 4 12 5 5;
                         8 8 24 10 10]),
    (-, 0, true) => cu([1, 2, 1, 4, 5]),
    (-, 1, true) => cu([1 1 0 2 3;
                        3 3 2 4 5]),
    (-, 0, false) => cu([-4, -4, -12, -5, -5]),
    (-, 1, false) => cu([-4 -4 -12 -5 -5;
                         -8 -8 -24 -10 -10]),
    (max, 0, true) => cu([3, 4, 5, 6, 7]),
    (max, 1, true) => cu([3 3 4 4 5;
                          5 5 6 6 7]),
    (max, 0, false) => cu([3, 2, 4, 4, 3]),
    (max, 1, false) => cu([3 2 4 4 3;
                           6 4 8 8 6]),
    (min, 0, true) => cu([1, 1, 1, 1, 1]),
    (min, 1, true) => cu([1 1 1 1 1;
                          1 1 1 1 1]),
    (min, 0, false) => cu([1, 2, 1, 1, 2]),
    (min, 1, false) => cu([1 2 1 1 2;
                           2 4 2 2 4]),
    (*, 0, true) => cu([3, 4, 5, 6, 7]),
    (*, 1, true) => cu([3 3 4 4 5;
                        5 5 6 6 7]),
    (*, 0, false) => cu([3, 4, 48, 4, 6]),
    (*, 1, false) => cu([3 4 48 4 6;
                        12 16 768 16 24]),
    (/, 0, true) => cu([0.75, 1., 0.3125, 1.5, 1.75]),
    (/, 1, true) => cu([0.75 0.75 0.25 1. 1.25;
                        1.25 1.25 0.375 1.5 1.75]),
    (/, 0, false) => cu([1//3, 1//4, 1//48, 1//4, 1//6]),
    (/, 1, false) => cu([1//3 1//4 1//48 1//4 1//6;
                         1//12 1//16 1//768 1//16 1//24]),
    (mean, 0, true) => cu([4., 5., 6., 7., 8.]),
    (mean, 1, true) => cu([4. 4. 5. 5. 6.;
                           6. 6. 7. 7. 8.]),
    (mean, 0, false) => cu([2, 2, 3, 2.5, 2.5]),
    (mean, 1, false) => cu([2. 2. 3. 2.5 2.5;
                            4. 4. 6. 5. 5.]),
)

types = [CuArray{Int32}, CuArray{Int64}, CuArray{Float32}, CuArray{Float64}]


@testset "scatter" begin
    for T = types
        @testset "$(T)" begin
            @testset "+" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    @test NNlib.scatter!(+, T(dsts[dims]), T(srcs[(dims, mutated)]), idx) == T(res[(+, dims, mutated)])

                    mutated = false
                    @test NNlib.scatter(+, T(srcs[(dims, mutated)]), idx) == T(res[(+, dims, mutated)])
                end
            end

            @testset "-" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    @test NNlib.scatter!(-, T(dsts[dims]), T(srcs[(dims, mutated)]), idx) == T(res[(-, dims, mutated)])

                    mutated = false
                    @test NNlib.scatter(-, T(srcs[(dims, mutated)]), idx) == T(res[(-, dims, mutated)])
                end
            end

            @testset "max" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    @test NNlib.scatter!(max, T(dsts[dims]), T(srcs[(dims, mutated)]), idx) == T(res[(max, dims, mutated)])

                    mutated = false
                    @test NNlib.scatter(max, T(srcs[(dims, mutated)]), idx) == T(res[(max, dims, mutated)])
                end
            end

            @testset "min" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    @test NNlib.scatter!(min, T(dsts[dims]), T(srcs[(dims, mutated)]), idx) == T(res[(min, dims, mutated)])

                    mutated = false
                    @test NNlib.scatter(min, T(srcs[(dims, mutated)]), idx) == T(res[(min, dims, mutated)])
                end
            end
        end
    end


    for T = [CuArray{Float32}, CuArray{Float64}]
        @testset "$(T)" begin
            @testset "*" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    @test NNlib.scatter!(*, T(dsts[dims]), T(srcs[(dims, mutated)]), idx) == T(res[(*, dims, mutated)])

                    mutated = false
                    @test NNlib.scatter(*, T(srcs[(dims, mutated)]), idx) == T(res[(*, dims, mutated)])
                end
            end

            @testset "/" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    @test NNlib.scatter!(/, T(dsts[dims]), T(srcs[(dims, mutated)].*2), idx) == T(res[(/, dims, mutated)])

                    mutated = false
                    @test NNlib.scatter(/, T(srcs[(dims, mutated)].*2), idx) == T(res[(/, dims, mutated)])
                end
            end

            @testset "mean" begin
                for idx = idxs, dims = [0, 1]
                    mutated = true
                    @test NNlib.scatter!(mean, T(dsts[dims]), T(srcs[(dims, mutated)]), idx) == T(res[(mean, dims, mutated)])

                    mutated = false
                    @test NNlib.scatter(mean, T(srcs[(dims, mutated)]), idx) == T(res[(mean, dims, mutated)])
                end
            end
        end
    end
end
