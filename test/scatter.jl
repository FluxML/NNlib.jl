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

function test_scatter(backend, types, ops; pt, ops_skip_types)
    cpu = CPU()
    for T in types
        PT = promote_type(T, pt)
        @testset failfast=true "$T" begin
            for op in ops
                skip_types = get(ops_skip_types, op, [])
                @testset "$op" begin
                    for idx = values(idxs), dims = [0, 1]
                        idx = adapt(backend, idx)
                        dst = adapt(backend, dsts[dims])

                        mutated = true
                        target_y = res[(op, dims, mutated)]
                        src = adapt(backend, srcs[(dims, mutated)])
                        if op == /
                            src = src .* T(2)
                        end

                        @test adapt(cpu, scatter!(op, T.(dst), T.(src), idx)) == T.(target_y)
                        @test adapt(cpu, scatter!(op, T.(dst), src, idx)) == PT.(target_y)
                        if op == /
                            @test adapt(cpu, scatter!(op, T.(dst), T.(src), idx)) == PT.(target_y)
                        else
                            @test adapt(cpu, scatter!(op, copy(dst), T.(src), idx)) == PT.(target_y)
                        end

                        if T ∉ skip_types
                            mutated = false
                            src = adapt(backend, srcs[(dims, mutated)])
                            @test adapt(cpu, scatter(op, T.(src), idx)) == T.(res[(op, dims, mutated)])
                        end
                    end
                end
            end
        end
    end
end

function scatter_testsuite(Backend)
    backend = Backend()
    gradtest_fn = backend == CPU() ? gradtest : gputest

    ops_skip_types = Dict(
        (+) => [],
        (-) => [UInt8, UInt16, UInt32, UInt64, UInt128],
        (*) => [UInt8, Int8],
        max => [BigInt],
        min => [BigInt])
    types = if backend == CPU()
        [UInt8,  UInt32, UInt64, Int32, Int64, Float16, Float32, Float64, BigFloat, Rational]
    elseif Symbol(typeof(backend)) == :CUDABackend
        [Int32, Int64, Float32, Float64]
    else
        # Need LLVM 15+ for atomic fmin/fmax:
        # https://reviews.llvm.org/D127041
        # But min/max can be done by reinterpreting an array to `UInt`.
        [Int32, Int64, UInt32, UInt64]
    end
    ops = backend == CPU() ?
        (+, -, max, min, *) :
        (+, -, max, min)
    test_scatter(backend, types, ops; pt=Int, ops_skip_types)

    types = backend == CPU() ?
        [Float16, Float32, BigFloat, Rational] :
        [Float32, Float64]
    ops = if backend == CPU()
        (/, mean)
    elseif Symbol(typeof(backend)) == :CUDABackend
        (*, /, mean)
    else
        # LLVM does not support atomic fmul/fdiv:
        # https://llvm.org/docs/LangRef.html#atomicrmw-instruction
        (mean,)
    end
    test_scatter(backend, types, ops; pt=Float64, ops_skip_types=Dict())

    if backend == CPU()
        @testset "scatter exceptions" begin
            @test_throws AssertionError scatter!(+, dsts[0], srcs[(1, true)], idxs[:int])
            idx = [1 2 3 4; 4 2 1 3; 6 7 8 9]
            @test_throws BoundsError scatter!(+, dsts[1], srcs[(1, true)], idx)
        end
    end
end

# @testset "∇scatter" begin
#     T = Float64
#     fdm(op) = op == min ? :backward : :forward
#     # fdm(op) = :forward

#     @testset "dstsize" begin
#         idx = [2, 2, 3, 4, 4]
#         src = ones(3, 5)
#         y = scatter(+, src, idx, dstsize = (3, 6))
#         @test size(y) == (3, 6)
#         gradtest(x -> scatter(+, x, idx, dstsize = (3,6)), src)
#     end

#     @testset "∂dst" begin
#         for op in (+, -, *, /, mean, max, min)
#             gradtest(xs -> scatter!(op, copy(xs), srcs[(0, true)], idxs[:int]), T.(dsts[0]), fdm=fdm(op))
#             gradtest(xs -> scatter!(op, copy(xs), srcs[(1, true)], idxs[:int]), T.(dsts[1]), fdm=fdm(op))
#         end
#     end

#     @testset "∂src" begin
#         for op in (+, -, *, /, mean, max, min)
#             gradtest(xs -> scatter!(op, T.(dsts[0]), xs, idxs[:int]), T.(srcs[(0, true)]), fdm=fdm(op))
#             gradtest(xs -> scatter!(op, T.(dsts[1]), xs, idxs[:int]), T.(srcs[(1, true)]), fdm=fdm(op))

#             gradtest(xs -> scatter(op, xs, idxs[:int]), T.(srcs[(0, false)]), fdm=fdm(op))
#             gradtest(xs -> scatter(op, xs, idxs[:int]), T.(srcs[(1, false)]), fdm=fdm(op))
#         end
#     end
# end
