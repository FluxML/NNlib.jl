using NNlib: scatter, scatter!

dsts = Dict(
    0 => [3, 4, 5, 6, 7],
    1 => [3 3 4 4 5;
          5 5 6 6 7],
)
srcs = Dict(
    (0, true) => ones(Int, 3, 4),
    (0, false) => ones(Int, 3) * collect(1:4)',
    (1, true) => ones(Int, 2, 3, 4),
    (1, false) => [1, 2] .* reshape(ones(Int, 3) * collect(1:4)', 1,3,4),
)
idxs = Dict(
    :int => [1 2 3 4;
             4 2 1 3;
             3 5 5 3],
    :tup => [(1,) (2,) (3,) (4,);
             (4,) (2,) (1,) (3,);
             (3,) (5,) (5,) (3,)],
    :car => CartesianIndex.(
            [(1,) (2,) (3,) (4,);
             (4,) (2,) (1,) (3,);
             (3,) (5,) (5,) (3,)]),
)
res = Dict(
    (+, 0, true) => [5, 6, 9, 8, 9],
    (+, 1, true) => [5 5 8 6 7;
                     7 7 10 8 9],
    (+, 0, false) => [4, 4, 12, 5, 5],
    (+, 1, false) => [4 4 12 5 5;
                      8 8 24 10 10],
    (-, 0, true) => [1, 2, 1, 4, 5],
    (-, 1, true) => [1 1 0 2 3;
                     3 3 2 4 5],
    (-, 0, false) => [-4, -4, -12, -5, -5],
    (-, 1, false) => [-4 -4 -12 -5 -5;
                      -8 -8 -24 -10 -10],
    (max, 0, true) => [3, 4, 5, 6, 7],
    (max, 1, true) => [3 3 4 4 5;
                       5 5 6 6 7],
    (max, 0, false) => [3, 2, 4, 4, 3],
    (max, 1, false) => [3 2 4 4 3;
                        6 4 8 8 6],
    (min, 0, true) => [1, 1, 1, 1, 1],
    (min, 1, true) => [1 1 1 1 1;
                 1 1 1 1 1],
    (min, 0, false) => [1, 2, 1, 1, 2],
    (min, 1, false) => [1 2 1 1 2;
                        2 4 2 2 4],
    (*, 0, true) => [3, 4, 5, 6, 7],
    (*, 1, true) => [3 3 4 4 5;
               5 5 6 6 7],
    (*, 0, false) => [3, 4, 48, 4, 6],
    (*, 1, false) => [3 4 48 4 6;
                      12 16 768 16 24],
    (/, 0, true) => [0.75, 1., 0.3125, 1.5, 1.75],
    (/, 1, true) => [0.75 0.75 0.25 1. 1.25;
               1.25 1.25 0.375 1.5 1.75],
    (/, 0, false) => [1//3, 1//4, 1//48, 1//4, 1//6],
    (/, 1, false) => [1//3 1//4 1//48 1//4 1//6;
                      1//12 1//16 1//768 1//16 1//24],
    (mean, 0, true) => [4., 5., 6., 7., 8.],
    (mean, 1, true) => [4. 4. 5. 5. 6.;
                  6. 6. 7. 7. 8.],
    (mean, 0, false) => [2, 2, 3, 2.5, 2.5],
    (mean, 1, false) => [2. 2. 3. 2.5 2.5;
                         4. 4. 6. 5. 5.],
)

types = [UInt8, UInt16, UInt32, UInt64, UInt128,
         Int8, Int16, Int32, Int64, Int128, BigInt,
         Float16, Float32, Float64, BigFloat, Rational]

@testset "scatter" begin
    for T = types
        @testset "$T" begin
            PT = promote_type(T, Int)
            @testset "+" begin
                for idx = values(idxs), dims = [0, 1]
                    mutated = true
                    @test scatter!(+, T.(copy(dsts[dims])), T.(srcs[(dims, mutated)]), idx) == T.(res[(+, dims, mutated)])
                    @test scatter!(+, T.(copy(dsts[dims])), srcs[(dims, mutated)], idx) == PT.(res[(+, dims, mutated)])
                    @test scatter!(+, copy(dsts[dims]), T.(srcs[(dims, mutated)]), idx) == PT.(res[(+, dims, mutated)])

                    mutated = false
                    @test scatter(+, T.(srcs[(dims, mutated)]), idx) == T.(res[(+, dims, mutated)])
                end
            end

            @testset "-" begin
                for idx = values(idxs), dims = [0, 1]
                    mutated = true
                    @test scatter!(-, T.(copy(dsts[dims])), T.(srcs[(dims, mutated)]), idx) == T.(res[(-, dims, mutated)])
                    @test scatter!(-, T.(copy(dsts[dims])), srcs[(dims, mutated)], idx) == PT.(res[(-, dims, mutated)])
                    @test scatter!(-, copy(dsts[dims]), T.(srcs[(dims, mutated)]), idx) == PT.(res[(-, dims, mutated)])

                    mutated = false
                    if !(T in [UInt8, UInt16, UInt32, UInt64, UInt128])
                        @test scatter(-, T.(srcs[(dims, mutated)]), idx) == T.(res[(-, dims, mutated)])
                    end
                end
            end

            @testset "max" begin
                for idx = values(idxs), dims = [0, 1]
                    mutated = true
                    @test scatter!(max, T.(copy(dsts[dims])), T.(srcs[(dims, mutated)]), idx) == T.(res[(max, dims, mutated)])
                    @test scatter!(max, T.(copy(dsts[dims])), srcs[(dims, mutated)], idx) == PT.(res[(max, dims, mutated)])
                    @test scatter!(max, copy(dsts[dims]), T.(srcs[(dims, mutated)]), idx) == PT.(res[(max, dims, mutated)])

                    mutated = false
                    if !(T in [BigInt])
                        @test scatter(max, T.(srcs[(dims, mutated)]), idx) == T.(res[(max, dims, mutated)])
                    end
                end
            end

            @testset "min" begin
                for idx = values(idxs), dims = [0, 1]
                    mutated = true
                    @test scatter!(min, T.(copy(dsts[dims])), T.(srcs[(dims, mutated)]), idx) == T.(res[(min, dims, mutated)])
                    @test scatter!(min, T.(copy(dsts[dims])), srcs[(dims, mutated)], idx) == PT.(res[(min, dims, mutated)])
                    @test scatter!(min, copy(dsts[dims]), T.(srcs[(dims, mutated)]), idx) == PT.(res[(min, dims, mutated)])

                    mutated = false
                    if !(T in [BigInt])
                        @test scatter(min, T.(srcs[(dims, mutated)]), idx) == T.(res[(min, dims, mutated)])
                    end
                end
            end

            @testset "*" begin
                for idx = values(idxs), dims = [0, 1]
                    mutated = true
                    @test scatter!(*, T.(copy(dsts[dims])), T.(srcs[(dims, mutated)]), idx) == T.(res[(*, dims, mutated)])
                    @test scatter!(*, T.(copy(dsts[dims])), srcs[(dims, mutated)], idx) == PT.(res[(*, dims, mutated)])
                    @test scatter!(*, copy(dsts[dims]), T.(srcs[(dims, mutated)]), idx) == PT.(res[(*, dims, mutated)])

                    mutated = false
                    if !(T in [UInt8, Int8])
                        @test scatter(*, T.(srcs[(dims, mutated)]), idx) == T.(res[(*, dims, mutated)])
                    end
                end
            end
        end
    end

    for T = [Float16, Float32, Float64, BigFloat, Rational]
        @testset "$T" begin
            PT = promote_type(T, Float64)
            @testset "/" begin
                for idx = values(idxs), dims = [0, 1]
                    mutated = true
                    @test scatter!(/, T.(dsts[dims]), T.(srcs[(dims, mutated)].*2), idx) == T.(res[(/, dims, mutated)])
                    @test scatter!(/, T.(dsts[dims]), srcs[(dims, mutated)].*2, idx) == PT.(res[(/, dims, mutated)])
                    @test scatter!(/, T.(dsts[dims]), T.(srcs[(dims, mutated)].*2), idx) == PT.(res[(/, dims, mutated)])

                    mutated = false
                    @test scatter(/, T.(srcs[(dims, mutated)]), idx) == T.(res[(/, dims, mutated)])
                end
            end

            @testset "mean" begin
                for idx = values(idxs), dims = [0, 1]
                    mutated = true
                    @test scatter!(mean, T.(dsts[dims]), T.(srcs[(dims, mutated)]), idx) == T.(res[(mean, dims, mutated)])
                    @test scatter!(mean, T.(dsts[dims]), srcs[(dims, mutated)], idx) == PT.(res[(mean, dims, mutated)])
                    @test scatter!(mean, copy(dsts[dims]), T.(srcs[(dims, mutated)]), idx) == PT.(res[(mean, dims, mutated)])

                    mutated = false
                    @test scatter(mean, T.(srcs[(dims, mutated)]), idx) == T.(res[(mean, dims, mutated)])
                end
            end
        end
    end

    @test_throws AssertionError scatter!(+, dsts[0], srcs[(1, true)], idxs[:int])
    idx = [1 2 3 4; 4 2 1 3; 6 7 8 9]
    @test_throws BoundsError scatter!(+, dsts[1], srcs[(1, true)], idx)
end
