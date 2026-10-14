@testset "softmax integer input" begin
    @test softmax(Int[0, 0]) == [0.5, 0.5]
end

@testset "softmax on different dims" begin
    xs = rand(fill(2, 5)...)
    out = similar(xs)
    for (fn!, fn) in [(softmax!, softmax), (logsoftmax!, logsoftmax)], i = 1:ndims(xs)
        @test fn!(out, xs; dims = i) == fn(xs; dims = i)
    end
end

@testset "softmax" begin
    xs = rand(5, 5)
    @test all(sum(softmax(xs), dims = 1) .≈ 1)
    @test all(sum(softmax(xs; dims = 2), dims = 2) .≈ 1)
    @test sum(softmax(vec(xs))) ≈ 1
    @test log.(softmax(xs; dims = 2)) ≈ logsoftmax(xs; dims = 2)

    xs = [-100_000.0, -100_000.0]
    @test softmax(xs) ≈ [0.5, 0.5]
    @test logsoftmax(xs) ≈ log.([0.5, 0.5])

    xs = rand(5)
    @test softmax(xs) ≈ exp.(xs) ./ sum(exp.(xs))
    @test logsoftmax(xs) ≈ log.(softmax(xs))

    xs = Float32[1, 2, 3000.0]
    @test logsoftmax(xs) ≈ [-2999, -2998, 0]

    xs = Float32[1 2 3; 1000 2000 3000]
    @test logsoftmax(xs) ≈ [-999 -1998 -2997; 0 0 0.0]

    y = logsoftmax(xs)
    @test ∇logsoftmax(ones(Float32, size(xs)), y) ≈ Float32[1 1 1; -1 -1 -1]
    
    y = softmax(xs)
    @test ∇softmax(ones(Float32, size(xs)), y) ≈ zeros(Float32, size(xs))

    # These values precalculated using PyTorch's nn.LogSoftmax
    xs = [
        -0.238639 0.748142 -0.283194 -0.525461 -1.5348 -0.797842
        0.690384 0.211427 0.254794 -0.213572 -0.314174 -0.372663
        -1.146370 -0.577988 0.718952 0.919720 -0.620773 0.929977
    ]
    ys = [
        0.237703 -0.621474 0.448193 0.546047 0.564185 0.632273
        -0.930163 0.0519798 0.0549979 0.3799 -0.477112 0.437428
        0.69246 0.569494 -0.503191 -0.925947 -0.0870738 -1.0697
    ]
    
    y = logsoftmax(xs)
    @test ∇logsoftmax(ones(size(xs)), y) ≈ ys rtol = 1e-6
    
    y = softmax(xs)
    @test ∇softmax(ones(size(xs)), y) ≈ zeros(size(xs)) atol = 1e-6
end

@testset "softmax with Inf, NaN" begin
    @test softmax(Float32[1 2; 3 Inf]) ≈    Float32[0.11920292 0.0; 0.880797 1.0]
    @test softmax(Float32[1 -Inf; 3 Inf]) ≈ Float32[0.11920292 0.0; 0.880797 1.0]
    @test softmax(Float32[1 Inf; 3 Inf]) ≈  Float32[0.11920292 0.5; 0.880797 0.5]

    @test softmax(Float32[1 2; 3 NaN]) ≈    Float32[0.11920292 NaN; 0.880797 NaN] nans=true
    @test softmax(Float32[1 2; 3 Inf]; dims=2) ≈ Float32[0.26894143 0.7310586; 0.0 1.0]
    @test softmax(Float32[1 2; 3 Inf]; dims=(:)) ≈ Float32[0.0 0.0; 0.0 1.0]
    @test softmax(Float32[1 2; 3 Inf]; dims=(1,2)) ≈ Float32[0.0 0.0; 0.0 1.0]

    @test exp.(logsoftmax(Float32[1 2; 3 Inf])) ≈ softmax(Float32[1 2; 3 Inf])
    @test exp.(logsoftmax(Float32[1 -Inf; 3 Inf])) ≈ softmax(Float32[1 -Inf; 3 Inf])
    @test exp.(logsoftmax(Float32[1 Inf; 3 Inf])) ≈ softmax(Float32[1 Inf; 3 Inf])
end

# `Array`s of `Float32` and `Float64` have loops of their own, divided among tasks from
# 2^17 elements on: on a contiguous slice for `pre == 1`, on tiles of rows otherwise.
@testset "CPU implementation" begin
    generic!(f) = f === softmax ? NNlib._softmax! : NNlib._logsoftmax!
    mutating!(f) = f === softmax ? softmax! : logsoftmax!

    @testset "$f, $T" for f in (softmax, logsoftmax), T in (Float32, Float64)
        f! = mutating!(f)
        for (sz, dims) in (((17,), 1), ((1,), 1), ((33, 7, 5), 1), ((33, 7, 5), 2), ((33, 7, 5), 3),
                           ((33, 7, 5), (1, 2)), ((33, 7, 5), (2, 3)), ((33, 7, 5), :),
                           ((300, 20), 2), ((1, 50, 1), 2), ((2, 3, 64, 5), 3),
                           # enough elements for the tasks
                           ((600, 600), 1), ((600, 600), 2), ((600, 600), :), ((3, 100_000), 1),
                           ((3, 100_000), 2), ((100_000, 3), 2), ((70, 60, 70), 2), ((70, 60, 70), (2, 3)))
            x = randn(rng, T, sz)
            y = f!(similar(x), x; dims)
            @test y ≈ generic!(f)(similar(x), x; dims) rtol = 20eps(T)
            @test f(x; dims) == y
            z = copy(x)
            @test f!(z; dims) === z
            @test z == y
            @test NNlib.@disallow_spawns(f!(similar(x), x; dims)) == y
        end
    end

    @testset "non-finite entries, $f" for f in (softmax, logsoftmax)
        f! = mutating!(f)
        # the generic implementation agrees, unless a slice has both a NaN and an Inf
        for T in (Float32, Float64), dims in (1, 2, :),
            x in (T[1 2; 3 Inf], T[1 -Inf; 3 Inf], T[1 Inf; 3 Inf], T[1 2; 3 NaN], T[-Inf -Inf; -Inf 1],
                  T[floatmax(T) 1; 2 3], T[-floatmax(T) -floatmax(T); 1 -Inf], T[floatmax(T) Inf; -floatmax(T) 0])
            @test f(x; dims) ≈ generic!(f)(similar(x), x; dims) nans = true
        end
        # the same on slices and tiles long enough for the vectorized loops, and for the tasks
        for sz in ((40, 50), (600, 600)), dims in (1, 2), bad in (Inf32, -Inf32, NaN32), pos in (1, 23, 40)
            x = randn(rng, Float32, sz)
            x[pos, 3] = x[5, pos] = x[end, end] = bad
            @test f(x; dims) ≈ generic!(f)(similar(x), x; dims) nans = true
        end
        x = randn(rng, Float32, 40, 50)
        x[21:40, :] .= -Inf32   # as left by a mask
        for dims in (1, 2)
            @test f(x; dims) ≈ generic!(f)(similar(x), x; dims) nans = true
        end
        # a NaN takes its slice, whatever else is there
        x = Float32[NaN 1 Inf; 2 Inf 3; Inf NaN 4]
        @test all(isnan, f(x; dims = :))
        @test isnan.(f(x; dims = 1)) == [true true false; true true false; true true false]
        @test isnan.(f(x; dims = 2)) == [true true true; false false false; true true true]
    end

    @testset "left to the generic implementation" begin
        x = randn(rng, Float32, 4, 5, 6)
        @test softmax(x; dims = (1, 3)) ≈ NNlib._softmax!(similar(x), x; dims = (1, 3))
        @test logsoftmax(x; dims = (1, 3)) ≈ NNlib._logsoftmax!(similar(x), x; dims = (1, 3))
        @test softmax(x; dims = 4) == ones(Float32, 4, 5, 6)
        @test softmax!(zeros(Float64, 4, 5, 6), x) ≈ softmax(x)
        @test softmax(view(x, :, 2:4, :)) ≈ softmax(x[:, 2:4, :])
        @test softmax(PermutedDimsArray(x, (2, 1, 3))) ≈ permutedims(softmax(x; dims = 2), (2, 1, 3))
        @test softmax(zeros(Float32, 0, 3)) == zeros(Float32, 0, 3)
        # arrays sharing memory, without being the same
        a = vec(copy(x))
        b = vec(reshape(a, size(x)))
        @test a !== b
        @test softmax!(a, b) ≈ softmax(vec(x))
    end
end

@testset "mutating softmax" begin
    map([
        Float64[1 2 3; 5 6 7],
        Float64[
            -0.238639 0.748142 -0.283194 -0.525461 -1.5348 -0.797842
            0.690384 0.211427 0.254794 -0.213572 -0.314174 -0.372663
            -1.146370 -0.577988 0.718952 0.919720 -0.620773 0.929977
        ],
    ]) do xs
        out = similar(xs)
        softmax!(out, xs)
        @test out ≈ softmax(xs) rtol = 1e-6
        logsoftmax!(out, xs)
        @test out ≈ logsoftmax(xs) rtol = 1e-6

        @testset "$fn(Float64, $(size(xs)))" for fn in [zeros, ones, rand]
            Δ = fn(Float64, size(xs))
            y = softmax(xs) 
            ∇softmax!(out, Δ, y)
            @test out ≈ ∇softmax(Δ, y)  rtol = 1e-6
            
            y = logsoftmax(xs)
            ∇logsoftmax!(out, Δ, y)
            @test out ≈ ∇logsoftmax(Δ, y)  rtol = 1e-6
        end
    end
end

@testset "logsumexp" begin
    flogsoft(x; dims) = mean(x .- logsoftmax(x; dims = dims), dims = dims)

    x = rand(3, 4)
    @test logsumexp(x) ≈ flogsoft(x, dims = :)
    @test logsumexp(x; dims = 1) ≈ flogsoft(x, dims = 1)
end

@testset "AutoDiff" begin
    for f in (softmax, logsoftmax), d in (:, 1, 2)
        @test test_gradients(x -> f(x; dims = d), randn(rng, 3, 4))
    end
    @test test_gradients(x -> softmax(x) .* (1:3), randn(rng, 3))
    @test test_gradients(x -> softmax(x) .* (1:3), randn(rng, 3, 5))
    @test test_gradients(x -> softmax(x, dims = 2) .* (1:3), randn(rng, 3, 5))

    @test test_gradients(x -> logsoftmax(x) .* (1:3), randn(rng, 3))
    @test test_gradients(x -> logsoftmax(x) .* (1:3), randn(rng, 3, 5))
    @test test_gradients(x -> logsoftmax(x, dims = 2) .* (1:3), randn(rng, 3, 5))

    for d  in (:, 1, 2)
        @test test_gradients(x -> logsumexp(x; dims = d), randn(rng, 3, 4))
    end
end

# Without rules for them Enzyme and Mooncake would differentiate the tasks of the CPU
# implementation. Reverse mode has the rules behind `test_gradients` above.
if NNLIB_TEST_ENZYME

@testset "EnzymeRules, forward mode" begin
    D, BD, C = EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated, EnzymeCore.Const
    for f! in (softmax!, logsoftmax!), dims in (1, 2, :), (Tret, Tout, Tx) in ((D, D, D), (BD, BD, BD), (D, D, C))
        EnzymeTestUtils.test_forward(f!, Tret, (zeros(3, 4), Tout), (randn(rng, 3, 4), Tx);
                                     fkwargs = (; dims), atol = 1e-6, rtol = 1e-6)
    end
    # in place, on a size that takes the tasks
    for (f, f!) in ((softmax, softmax!), (logsoftmax, logsoftmax!))
        x, v, w = randn(rng, 600, 600), randn(rng, 600, 600), randn(rng, 600, 600)
        d = Enzyme.autodiff(Enzyme.Forward, (x, w) -> sum(w .* f!(copy(x))), Duplicated(x, v), Const(w))[1]
        @test d ≈ sum(v .* gradient(x -> sum(w .* f(x)), x)[1])
    end
end

end # NNLIB_TEST_ENZYME

@testset "Mooncake, mutating functions" begin
    for (f, f!) in ((softmax, softmax!), (logsoftmax, logsoftmax!)), sz in ((3, 4), (600, 600)), dims in (1, 2)
        x, w = randn(rng, Float32, sz), randn(rng, Float32, sz)
        y, (dx,) = withgradient(x -> sum(w .* f!(similar(x), x; dims)), AutoMooncake(), x)
        @test dx ≈ gradient(x -> sum(w .* f(x; dims)), x)[1]
    end
end

@testset "Second derivatives" begin
    x = [1 2 3; 6 5 4]
    H = Zygote.hessian_dual(x -> sum(sin, softmax(x)), x)
    @test H ≈ Zygote.hessian_reverse(x -> sum(sin, softmax(x)), x)

    H2 = Zygote.hessian_dual(x -> sum(sin, logsoftmax(x)), x)
    @test H2 ≈ Zygote.hessian_reverse(x -> sum(sin, logsoftmax(x)), x)

    H3 = Zygote.hessian_dual(x -> sum(sin, logsumexp(x)), x)
    @test H3 ≈ Zygote.hessian_reverse(x -> sum(sin, logsumexp(x)), x)

    # Reverse-over-reverse goes through the rules for the gradients themselves.
    for d in (:, 1, 2, (1, 2))
        y, dy = softmax(randn(rng, 3, 4); dims = d), randn(rng, 3, 4)
        ChainRulesTestUtils.test_rrule(NNlib.∇softmax, dy, y; fkwargs = (; dims = d))
        ChainRulesTestUtils.test_rrule(NNlib.∇logsoftmax, dy, log.(y); fkwargs = (; dims = d))
    end
end
