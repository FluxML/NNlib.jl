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
        @testset "$T" begin
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
        # But fmin/fmax can be done by reinterpreting an array to `UInt`.
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
            idx = [1 2 3 4; 4 2 1 3; 6 7 8 9]
            @test_throws AssertionError scatter!(+, copy(dsts[0]), srcs[(1, true)], idxs[:int])
            @test_throws BoundsError scatter!(+, copy(dsts[1]), srcs[(1, true)], idx)
        end
    end

    @testset "∇scatter" begin
        T = Float64
        fdm(op) = op == min ? :backward : :forward

        @testset "dstsize" begin
            idx = adapt(backend, [2, 2, 3, 4, 4])
            src = adapt(backend, ones(T, 3, 5))
            y = scatter(+, src, idx, dstsize = (3, 6))
            @test eltype(y) == T
            @test size(y) == (3, 6)
            backend == CPU() ?
                gradtest_fn(x -> scatter(+, x, idx; dstsize=(3, 6)), src) :
                gradtest_fn((x, i) -> scatter(+, x, i; dstsize=(3, 6)), src, idx)
        end

        @testset "∂dst" begin
            ops = if backend == CPU() || Symbol(typeof(backend)) == :CUDABackend
                (+, -, *, /, mean, max, min)
            else
                (+, -, mean, max, min)
            end
            for op in ops, i in (0, 1)
                PT = ( # If not CPU and CUDA -> use Int64 for min/max.
                    backend != CPU() &&
                    Symbol(typeof(backend)) != :CUDABackend &&
                    (op == max || op == min)) ? Int64 : T

                src = adapt(backend, srcs[(i, true)])
                idx = adapt(backend, idxs[:int])
                dst = adapt(backend, PT.(dsts[i]))
                backend == CPU() ?
                    gradtest_fn(x -> scatter!(op, copy(x), src, idx), dst; fdm=fdm(op)) :
                    gradtest_fn((x, s, i) -> scatter!(op, x, s, i), dst, src, idx)
            end
        end

        @testset "∂src" begin
            ops = if backend == CPU() || Symbol(typeof(backend)) == :CUDABackend
                (+, -, *, /, mean, max, min)
            else
                (+, -, mean, max, min)
            end
            for op in ops, i in (0, 1)
                PT = ( # If not CPU and CUDA -> use Int64 for min/max.
                    backend != CPU() &&
                    Symbol(typeof(backend)) != :CUDABackend &&
                    (op == max || op == min)) ? Int64 : T
                src = PT.(adapt(backend, srcs[(i, false)]))
                idx = adapt(backend, idxs[:int])
                backend == CPU() ?
                    gradtest_fn(xs -> scatter(op, xs, idx), src; fdm=fdm(op)) :
                    gradtest_fn((xs, i) -> scatter(op, xs, i), src, idx)
            end
        end
    end
end
