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

function test_scatter(device, types, ops; pt, ops_skip_types)
    for T in types, IT in (Int8, Int64)
        PT = promote_type(T, pt)
        @testset "eltype $T - idx eltype $IT - $op" for op in ops
            skip_types = get(ops_skip_types, op, [])
            for idx = values(idxs), dims = [0, 1]
                # Tests with indices of different types.
                eltype(idx) == Int && (idx = IT.(idx);)

                idx = device(idx)
                dst = device(dsts[dims])

                mutated = true
                target_y = res[(op, dims, mutated)]
                src = device(srcs[(dims, mutated)])
                if op == /
                    src = src .* T(2)
                end

                @test cpu(scatter!(op, T.(dst), T.(src), idx)) == T.(target_y)
                @test cpu(scatter!(op, T.(dst), src, idx)) == PT.(target_y)
                if op == /
                    @test cpu(scatter!(op, T.(dst), T.(src), idx)) == PT.(target_y)
                else
                    @test cpu(scatter!(op, copy(dst), T.(src), idx)) == PT.(target_y)
                end

                if T ∉ skip_types
                    mutated = false
                    src = device(srcs[(dims, mutated)])
                    @test cpu(scatter(op, T.(src), idx)) == T.(res[(op, dims, mutated)])
                end
            end
        end
    end
end

function scatter_testsuite(Backend)
    device(x) = adapt(Backend(), x)

    ops_skip_types = Dict(
        (+) => [],
        (-) => [UInt8, UInt16, UInt32, UInt64, UInt128],
        (*) => [UInt8, Int8],
        max => [BigInt],
        min => [BigInt])
    # All GPU backends now cover the full op set on float `dst`: CUDA/AMDGPU through
    # Atomix, Metal through `Metal.@atomic` (a compare-and-swap fallback) in
    # NNlibMetalExt. (Atomix's Metal atomics still lack float `max`/`min`/`*`/`/`, hence
    # the ext.) So AMDGPU/Metal are tested like CUDA, with `Float` types for `min`/`max`.
    types = Backend == CPU ?
        [UInt8,  UInt32, UInt64, Int32, Int64, Float16, Float32, Float64, BigFloat, Rational] :
        [Int32, Int64, Float32, Float64]
    ops = Backend == CPU ?
        (+, -, max, min, *) :
        (+, -, max, min)
    test_scatter(device, types, ops; pt=Int, ops_skip_types)

    types = Backend == CPU ?
        [Float16, Float32, BigFloat, Rational] :
        [Float32, Float64]
    ops = Backend == CPU ?
        (/, mean) :
        (*, /, mean)
    test_scatter(device, types, ops; pt=Float64, ops_skip_types=Dict())

    if Backend == CPU
        @testset "scatter exceptions" begin
            idx = [1 2 3 4; 4 2 1 3; 6 7 8 9]
            @test_throws AssertionError scatter!(+, copy(dsts[0]), srcs[(1, true)], idxs[:int])
            @test_throws BoundsError scatter!(+, copy(dsts[1]), srcs[(1, true)], idx)
        end
    end

    @testset "∇scatter" begin
        T = Float64
        # `scatter`'s `min`/`max`/`*`/`/` gradients are kinked; use a one-sided
        # finite-difference reference (forward, or backward for `min`) to stay on the
        # correct branch.
        get_reference_ad(op) = Backend != CPU ? AutoZygote() : fdm(op)  
        fdm(op) = AutoFiniteDifferences(fdm = op == min ?
            FiniteDifferences.backward_fdm(5, 1) : FiniteDifferences.forward_fdm(5, 1))

        @testset "dstsize" begin
            idx_cpu = [2, 2, 3, 4, 4]
            src = ones(T, 3, 5)
            y = scatter(+, src, idx_cpu, dstsize = (3, 6))
            @test eltype(y) == T
            @test size(y) == (3, 6)
            idx_d = device(idx_cpu)
            @test test_gradients(x -> scatter(+, x, idx_cpu; dstsize=(3, 6)), src;
                test_gpu = Backend != CPU,
                f_gpu = x -> scatter(+, x, idx_d; dstsize=(3, 6)))
        end

        @testset "∂dst" begin
            for op in (+, -, *, /, mean, max, min), i in (0, 1), IT in (Int8, Int64)
                src = srcs[(i, true)]
                idx = IT.(idxs[:int])
                dst = T.(dsts[i])
                src_d = device(src); idx_d = device(idx)
                @test test_gradients(x -> scatter!(op, copy(x), src, idx), dst;
                    test_gpu = Backend != CPU, reference = get_reference_ad(op),
                    f_gpu = x -> scatter!(op, copy(x), src_d, idx_d))
            end
        end

        @testset "∂src" begin
            for op in (+, -, *, /, mean, max, min), i in (0, 1), IT in (Int8, Int64)
                src = T.(srcs[(i, false)])
                idx = IT.(idxs[:int])
                idx_d = device(idx)
                @test test_gradients(xs -> scatter(op, xs, idx), src;
                    test_gpu = Backend != CPU, reference = get_reference_ad(op),
                    f_gpu = xs -> scatter(op, xs, idx_d))
            end
        end

        # Regression test for #703: `*`/`/` gradients used to error for a 1-D
        # (vector) index array, because `reverse_indices` mishandled linear keys.
        @testset "∂src vector index (#703) - $op" for op in (*, /)
            idx = [3, 1, 2, 2]      # 1-D index, with a uniquely-mapped value
            src = T[10, 100, 1000, 1]
            idx_d = device(idx)
            @test test_gradients(xs -> scatter(op, xs, idx), src;
                test_gpu = Backend != CPU, reference = get_reference_ad(op),
                f_gpu = xs -> scatter(op, xs, idx_d))
        end


        @static if Test_Enzyme

        @testset "EnzymeRules" begin
            idx = device([2, 2, 3, 4, 4])
            src = device(ones(T, 3, 5))

            for op in (+, -)

                dst = scatter(op, src, idx)

                for Tret in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
                    Tdst in (EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
                    Tsrc in (EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated)

                    Tret == EnzymeCore.Const && continue # ERROR                 
                    EnzymeTestUtils.are_activities_compatible(Tret, Tdst, Tsrc) || continue

                    EnzymeTestUtils.test_reverse(scatter!, Tret, (op, EnzymeCore.Const), (dst, Tdst), (src, Tsrc), (idx, EnzymeCore.Const))
                end
            end
        end

        end
    end
end
