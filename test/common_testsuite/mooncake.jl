# Tests for `ext/NNlibMooncakeExt/rules.jl`. Ported from `test/ext/nnlib/nnlib.jl` in
# Mooncake.jl, without the cases that belong to its LuxLib extension.

using Mooncake.TestUtils: test_rule, test_rule_throws
using Mooncake.Nfwd: NDual, ndual_partial, ndual_value
using JET: JET  # `test_rule`'s stability checks need Mooncake's JET extension

dropout_tester_1(Trng, x, p) = dropout(Trng(1), x, p; dims=1)
dropout_tester_2(Trng, x, p) = dropout(Trng(1), x, p; dims=2)
dropout_tester_3(Trng, x, p) = dropout(Trng(1), x, p; dims=(1, 2))

# Fix p so finite differences test the input gradient, not the sampling probability.
function dropout_mixed_tester(Trng, x, ::Val{p}, ::Val{dims}) where {p,dims}
    return dims === nothing ? dropout(Trng(1), x, p) : dropout(Trng(1), x, p; dims)
end

# At p == 0 `dropout` returns its input itself, so mutating the result doubles the sum. `p`
# is fixed inside because `test_rule` would perturb it to `-ε`, outside `dropout`'s domain.
function dropout_alias_tester(Trng, x)
    y = dropout(Trng(1), x, zero(eltype(x)))
    y .*= 2
    return sum(x)
end

function dropout_alias_tester_dims(Trng, x)
    y = dropout(Trng(1), x, zero(eltype(x)); dims=1)
    y .*= 2
    return sum(x)
end

# Every activation `bias_act!` has a rule for, but `identity`. `tanh` is among them as `bias_act!`
# computes `tanh_fast` for it.
const bias_act_activations = [getfield(NNlib, f) for (f, _) in NNlib.UNARY_ACTS]
push!(bias_act_activations, tanh)

function mooncake_testsuite(Backend)
    gpu = Backend != CPU
    device(x) = adapt(Backend(), x)
    T = gpu ? Float32 : Float64
    _rand(rng, size...) = device(randn(rng, T, size...))
    _ones(d...) = device(ones(T, d...))
    # Random `x` and `b` (`false` if `bsize` is `nothing`), with `x .+ b` kept off the kinks of the
    # piecewise activations, that finite differences cannot cross.
    function bias_act_args(rng, xsize, bsize)
        x = randn(rng, T, xsize...)
        b = bsize === nothing ? zeros(T, 1) : randn(rng, T, bsize...)
        kinks = T[-3, -1, -0.5, 0, 0.5, 1, 3, 6]
        while true
            near = [minimum(abs.(z .- kinks)) < 0.1 for z in x .+ b]
            any(near) || break
            x .+= T(0.23) .* near
        end
        return device(x), bsize === nothing ? false : device(b)
    end
    Trng = gpu ? typeof(NNlib._rng_from_array(_ones(1))) : StableRNG
    # CUDA.jl#2886 is fixed after 5.9.6, allowing these checks on the GPU.
    cuda = gpu && CUDA.functional() && pkgversion(CUDA) > v"5.9.6"

    if !gpu || cuda
        @testset "dropout rejects invalid mixed-precision probabilities" begin
            for p in (-eps(Float64), nextfloat(1.0), Inf, NaN), kw in (false, true)
                rng = Trng(123)
                x = device(ones(Float32, 4))
                args = kw ? (Core.kwcall, (dims=1,), dropout, rng, x, p) : (dropout, rng, x, p)
                test_rule_throws(
                    StableRNG(123),
                    args...;
                    err=ArgumentError,
                    msg="dropout expects a probability 0 <= p <= 1",
                    mode=Mooncake.ReverseMode,
                    primal=true,
                )
                @test rand(rng, Float32, 8) == rand(Trng(123), Float32, 8)
            end
        end
    end

    gpu || @testset "batched_mul with a singleton batch" for batches in ((3, 3), (1, 3), (3, 1))
        rng = StableRNG(123)
        test_rule(rng, batched_mul, randn(rng, 3, 2, batches[1]), randn(rng, 2, 5, batches[2]);
            is_primitive=true, unsafe_perturb=true)
    end

    @testset "affine normalization" begin
        rng = StableRNG(123)
        test_rule(rng, NNlib._affine_normalize,
            device(randn(rng, Float32, 2, 5)),
            device(randn(rng, Float32, 2, 1)),
            # Keep variance + epsilon positive under finite-difference perturbations.
            device(1.0f0 .+ rand(rng, Float32, 2, 1)),
            device(randn(rng, Float32, 2, 1)),
            device(randn(rng, Float32, 2, 1)),
            1.0f-5;
            is_primitive=true, unsafe_perturb=true)
    end

    @testset "rules" begin
        rng = StableRNG(123)
        x = randn(rng, T, 5, 4, 3, 2)
        w = randn(rng, T, 2, 2, 3, 3)
        dense_cdims = DenseConvDims(x, w)
        sep_cd = DepthwiseConvDims(x, w)
        y = device(conv(x, w, dense_cdims))
        y_sep = device(depthwiseconv(x, w, sep_cd))
        pool_dims = PoolDims(size(x), 2)
        x, w = device(x), device(w)

        # Wider than its `src` and not representable in it, so NNlib's rounding is observable;
        # `2.0` round-trips and would pass either way.
        mixed_src = device(ones(Float32, 3))
        mixed_init = 2.1

        # (interface_only, perf_flag, is_primitive, f, args...)
        test_cases = Any[
            (false, :none, true, batched_mul, _rand(rng, 3, 2, 3), _rand(rng, 2, 5, 3)),

            # dropout
            (true, :none, false, dropout_tester_1, Trng, _rand(rng, 2, 2), T(0.5)),
            (true, :none, false, dropout_tester_2, Trng, _rand(rng, 2, 2), T(0.1)),
            (true, :none, false, dropout_tester_3, Trng, _rand(rng, 2, 2), T(0.4)),
            # `p` may be a non-floating `Real` as well.
            (true, :none, false, dropout_tester_2, Trng, _rand(rng, 2, 2), 1 // 10),
            (false, :none, false, dropout_alias_tester, Trng, _rand(rng, 2, 2)),
            (false, :none, false, dropout_alias_tester_dims, Trng, _rand(rng, 2, 2)),
            # The aliasing arm, `p ≤ 0`, which returns the input `CoDual`. On a device these
            # also need the `materialize!` rule for the forward broadcast.
            (false, :none, false, dropout_alias_tester, Trng, _rand(rng, 2, 2)'),
            (false, :none, false, dropout_alias_tester, Trng, transpose(_rand(rng, 2, 2))),
            # ... and the other arm, `p > 0`, where the cotangent goes via `arrayify`'s wrapper.
            (true, :none, false, dropout_tester_1, Trng, _rand(rng, 2, 2)', T(0.5)),
            (true, :none, false, dropout_tester_2, Trng, transpose(_rand(rng, 2, 2)), T(0.1)),

            # softmax
            (false, :stability, true, softmax, _rand(rng, 2)),
            (false, :stability, true, softmax, _rand(rng, 2, 2)),
            (false, :stability, true, Core.kwcall, (dims=1,), softmax, _rand(rng, 2)),
            (false, :stability, true, Core.kwcall, (dims=1,), softmax, _rand(rng, 3, 3)),
            (false, :stability, true, Core.kwcall, (dims=2,), softmax, _rand(rng, 3, 3)),
            (false, :stability, true, Core.kwcall, (dims=(1, 2),), softmax, _rand(rng, 3, 3)),
            (false, :stability, true, Core.kwcall, (dims=(1, 2),), softmax, _rand(rng, 3, 3, 2)),
            (false, :none, false, x -> softmax(x; dims=1), _rand(rng, 3, 2)),
            (false, :none, false, x -> softmax(x; dims=2), _rand(rng, 3, 2)),
            (false, :none, false, x -> softmax(x; dims=(1, 2)), _rand(rng, 3, 2)),
            (false, :stability, true, softmax, _rand(rng, 2, 3)'),
            (false, :stability, true, Core.kwcall, (dims=1,), softmax, _rand(rng, 3, 3)'),
            (false, :stability, true, Core.kwcall, (dims=2,), softmax, _rand(rng, 3, 3)'),
            (false, :stability, true, Core.kwcall, (dims=1,), softmax, transpose(_rand(rng, 3, 3))),
            (false, :stability, true, Core.kwcall, (dims=2,), softmax, transpose(_rand(rng, 3, 3))),

            # logsoftmax
            (false, :stability, true, logsoftmax, _rand(rng, 2)),
            (false, :stability, true, logsoftmax, _rand(rng, 2, 3)),
            (false, :stability, true, logsoftmax, _rand(rng, 2, 3, 2)),
            (false, :stability, true, Core.kwcall, (dims=1,), logsoftmax, _rand(rng, 2)),
            (false, :stability, true, Core.kwcall, (dims=1,), logsoftmax, _rand(rng, 3, 3)),
            (false, :stability, true, Core.kwcall, (dims=2,), logsoftmax, _rand(rng, 3, 3)),
            (false, :stability, true, Core.kwcall, (dims=(1, 2),), logsoftmax, _rand(rng, 3, 3)),
            (false, :stability, true, Core.kwcall, (dims=(1, 2),), logsoftmax, _rand(rng, 3, 3, 2)),
            (false, :stability, true, logsoftmax, _rand(rng, 2, 3)'),
            (false, :stability, true, Core.kwcall, (dims=1,), logsoftmax, _rand(rng, 3, 3)'),
            (false, :stability, true, Core.kwcall, (dims=2,), logsoftmax, _rand(rng, 3, 3)'),
            (false, :stability, true, Core.kwcall, (dims=1,), logsoftmax, transpose(_rand(rng, 3, 3))),
            (false, :stability, true, Core.kwcall, (dims=2,), logsoftmax, transpose(_rand(rng, 3, 3))),

            # logsumexp
            (false, :stability, true, logsumexp, _rand(rng, 2)),
            (false, :stability, true, logsumexp, _rand(rng, 3, 3)),
            (false, :stability, true, logsumexp, _rand(rng, 3, 3, 2)),
            (false, :stability, true, Core.kwcall, (dims=1,), logsumexp, _rand(rng, 2)),
            (false, :stability, true, Core.kwcall, (dims=1,), logsumexp, _rand(rng, 3, 3)),
            (false, :stability, true, Core.kwcall, (dims=2,), logsumexp, _rand(rng, 3, 3)),
            (false, :stability, true, Core.kwcall, (dims=(1, 2),), logsumexp, _rand(rng, 3, 3)),
            (false, :stability, true, Core.kwcall, (dims=(1, 2),), logsumexp, _rand(rng, 3, 3, 2)),
            (false, :stability, true, logsumexp, _rand(rng, 2, 3)'),
            (false, :stability, true, Core.kwcall, (dims=1,), logsumexp, _rand(rng, 3, 3)'),
            (false, :stability, true, Core.kwcall, (dims=2,), logsumexp, _rand(rng, 3, 3)'),
            (false, :stability, true, Core.kwcall, (dims=1,), logsumexp, transpose(_rand(rng, 3, 3))),
            (false, :stability, true, Core.kwcall, (dims=2,), logsumexp, transpose(_rand(rng, 3, 3))),

            # upsample_nearest
            (false, :stability, true, upsample_nearest, _rand(rng, 3), (2,)),
            (false, :stability, true, upsample_nearest, _rand(rng, 3, 2), (2, 2)),
            (false, :stability, true, upsample_nearest, _rand(rng, 3, 2, 3), (2, 2, 5)),

            # fold, unfold
            (false, :none, true, NNlib.fold, _rand(rng, 12, 12, 2), size(x), dense_cdims),
            (false, :none, true, NNlib.unfold, x, dense_cdims),

            # scatter
            (false, :none, true, scatter, +, _rand(rng, 2), [1, 3]),
            (false, :none, true, Core.kwcall, (;), scatter, +, _rand(rng, 2), [1, 3]),
            # `scatter(max/min, ...)` with sources tied for one destination, hence `_ones`.
            (false, :none, true, scatter, max, _ones(3), [1, 1, 2]),
            (false, :none, true, scatter, min, _ones(3), [1, 1, 2]),
            (false, :none, true, Core.kwcall, (;), scatter, max, _ones(3), [1, 1, 2]),
            # ndims(src) > ndims(idx): the tie splits per row, and the helper's gather/scatter
            # shapes have to agree on the extra leading dimension.
            (false, :none, true, scatter, max, _ones(2, 3), [1, 1, 2]),
            # `init` is differentiable, so the keyword NamedTuple's rdata is not NoRData. The
            # cases put it below every source, above every source, and tied with one — a tie whose
            # symmetric split is the mean of the one-sided derivatives a central difference takes.
            (false, :none, true, Core.kwcall, (init=T(0.5),), scatter, max, _ones(3), [1, 1, 2]),
            (false, :none, true, Core.kwcall, (init=T(2.0),), scatter, max, _ones(3), [1, 1, 2]),
            (false, :none, true, Core.kwcall, (init=T(1.0),), scatter, max, device(T[1, 2]), [1, 2]),
            (false, :none, true, Core.kwcall, (init=mixed_init,), scatter, max, mixed_src, [1, 1, 2]),
            # An `init` with no rdata. It wins outright, so nothing ties it.
            (false, :none, true, Core.kwcall, (init=2,), scatter, max, _ones(3), [1, 1, 2]),
            # A `NaN` in `src` must reach the gradient in both arms rather than be silenced to
            # zero. `interface_only`, since a `NaN` gradient is not finite-differenceable.
            (true, :none, true, scatter, max, [NaN, 1.0, 2.0], [1, 1, 2]),
            (true, :none, true, Core.kwcall, (init=T(2.0),), scatter, max, [NaN, 1.0, 2.0], [1, 1, 2]),
            # `init` over a multi-dim `src`, where the tie count and the `init` indicator have to
            # agree on the extra leading dimension.
            (false, :none, true, Core.kwcall, (init=T(2.0),), scatter, max, _ones(2, 3), [1, 1, 2]),
            # `init=nothing` is NNlib's own default and takes the no-`init` path.
            (false, :none, true, Core.kwcall, (init=nothing,), scatter, max, _ones(3), [1, 1, 2]),
            # `dstsize` past the largest index leaves destinations no index reaches, holding
            # `scatter_empty`; nothing gathers them, so they take no part in the gradient, and the
            # slice keeps their `-Inf`s out of the sum.
            (false, :none, false, x -> sum(scatter(max, x, [1, 1, 2]; dstsize=(4,))[1:2]), _ones(3)),

            # gather
            (false, :none, true, gather, _rand(rng, 2, 4), [1, 3, 1]),
            # Wrapped `src`: the pullback scatters into `arrayify`'s fdata, which keeps the
            # wrapper, and `scatter!` takes only a dense destination. Square, so the wrapped
            # trailing size still admits the indices.
            (false, :none, true, gather, _rand(rng, 4, 4)', [1, 3, 1]),
            (false, :none, true, gather, transpose(_rand(rng, 4, 4)), [1, 3, 1]),

            # conv
            (false, :none, true, Core.kwcall, (;), conv, x, w, dense_cdims),
            (false, :none, true, conv, x, w, dense_cdims),
            (false, :none, true, Core.kwcall, (;), ∇conv_data, y, w, dense_cdims),
            (false, :none, true, ∇conv_data, y, w, dense_cdims),
            (false, :none, true, Core.kwcall, (;), ∇conv_filter, x, y, dense_cdims),
            (false, :none, true, ∇conv_filter, x, y, dense_cdims),

            # pooling
            (false, :none, true, maxpool, x, pool_dims),
            (false, :none, true, Core.kwcall, (;), maxpool, x, pool_dims),
            (false, :none, true, meanpool, x, pool_dims),
            (false, :none, true, Core.kwcall, (;), meanpool, x, pool_dims),

            # padding
            (false, :none, false, x -> pad_constant(x, 1, T(2.0)), x),
            (false, :none, false, x -> pad_constant(x, 1, T(2.0); dims=:), x),

            # bias_act!(σ, x, b): modifies x in-place
            (false, :stability, true, bias_act!, identity, _rand(rng, 8, 4), _rand(rng, 8)),
            (false, :stability, true, bias_act!, identity, _rand(rng, 8), _rand(rng, 8)),
            (false, :none, true, bias_act!, relu, bias_act_args(rng, (3, 3, 4, 2), (1, 1, 4, 1))...),
            (false, :none, true, bias_act!, swish, bias_act_args(rng, (3, 3, 4, 2), (1, 1, 4, 1))...),
            (false, :none, true, bias_act!, relu, bias_act_args(rng, (8, 4), (1,))...),
            (false, :none, true, bias_act!, relu, bias_act_args(rng, (8, 4), nothing)...),
            (false, :none, true, bias_act!, swish, bias_act_args(rng, (8, 4), nothing)...),
            (false, :none, true, bias_act!, tanh, bias_act_args(rng, (8, 4), nothing)...),
            [(false, :none, true, bias_act!, σ, bias_act_args(rng, (8, 4), (8,))...)
             for σ in bias_act_activations]...,
        ]
        if !gpu || cuda
            append!(test_cases, [
                (
                    false,
                    :none,
                    false,
                    dropout_mixed_tester,
                    Trng,
                    wrap(device(ones(S, 32, 4))),
                    Val(p),
                    Val(dims),
                ) for (S, p) in ((Float32, 0.9), (Float64, 0.1f0), (Float32, 0.0)) for
                dims in (nothing, 1) for wrap in (identity, adjoint, transpose)
            ])
        end
        if gpu
            push!(test_cases,
                (false, :none, true, bias_act!, tanh, device(fill(20.0f0, 8)), device(zeros(Float32, 8))),
            )
        else
            # These fail on CUDA.
            push!(test_cases,
                (false, :none, false, x -> softmax(5x), _rand(rng, 3, 2)),
                (false, :none, true, Core.kwcall, (;), depthwiseconv, x, w, sep_cd),
                (false, :none, true, depthwiseconv, x, w, sep_cd),
                (false, :none, true, Core.kwcall, (;), ∇depthwiseconv_data, y_sep, w, sep_cd),
                (false, :none, true, ∇depthwiseconv_data, y_sep, w, sep_cd),
            )
        end

        @testset "$(typeof(fargs))" for (interface_only, perf_flag, is_primitive, fargs...) in test_cases
            perf_flag = gpu ? :none : perf_flag
            test_rule(StableRNG(123), fargs...;
                perf_flag, is_primitive, interface_only, mode=Mooncake.ReverseMode)
        end
    end

    @testset "saturated bias activations" begin
        ref_tanh = 4 * exp(-40.0f0) / (1 + exp(-40.0f0))^2
        ref_sigmoid = exp(-20.0f0) / (1 + exp(-20.0f0))^2
        for (activation, ref) in ((tanh, ref_tanh), (tanh_fast, ref_tanh),
                                  (NNlib.σ, ref_sigmoid), (sigmoid_fast, ref_sigmoid))
            x = device(fill(20.0f0, 1))
            b = device(zeros(Float32, 1))
            out = Mooncake.frule!!(
                Mooncake.zero_dual(bias_act!),
                Mooncake.zero_dual(activation),
                Mooncake.Dual(x, device(ones(Float32, 1))),
                Mooncake.Dual(b, device(zeros(Float32, 1))),
            )
            @test only(cpu(Mooncake.tangent(out))) ≈ ref

            x = device(fill(20.0f0, 1))
            dx = device(zeros(Float32, 1))
            b = device(zeros(Float32, 1))
            db = device(zeros(Float32, 1))
            out, pullback = Mooncake.rrule!!(
                Mooncake.zero_fcodual(bias_act!),
                Mooncake.zero_fcodual(activation),
                Mooncake.CoDual(x, dx),
                Mooncake.CoDual(b, db),
            )
            fill!(Mooncake.tangent(out), 1)
            pullback(Mooncake.NoRData())
            @test only(cpu(dx)) ≈ ref
            @test only(cpu(db)) ≈ ref
        end
    end

    # The pullback has to leave `x` as it found it, whatever its own output was overwritten with,
    # and its gradients are those of NNlib's `rrule`.
    @testset "bias_act! restores x: $σ, $(b isa Bool ? b : size(b))" for σ in (identity, relu, elu, tanh, swish),
                                                         (xs, b) in (((8, 4), _rand(rng, 8)),
                                                                     ((8, 4), false),
                                                                     ((3, 3, 4, 2), _rand(rng, 1, 1, 4, 1)))
        x0 = _rand(rng, xs...)
        x, dx, db = copy(x0), zero(x0), b isa Bool ? Mooncake.NoFData() : zero(b)
        out, pullback = Mooncake.rrule!!(
            Mooncake.zero_fcodual(bias_act!), Mooncake.zero_fcodual(σ),
            Mooncake.CoDual(x, dx), Mooncake.CoDual(b, db),
        )
        @test Mooncake.primal(out) === x
        @test cpu(x) == cpu(bias_act!(σ, copy(x0), b))
        fill!(Mooncake.tangent(out), 1)
        pullback(Mooncake.NoRData())
        @test x == x0
        ref_dx, ref_db = gradient((x, b) -> sum(bias_act!(σ, copy(x), b)), x0, b)
        @test cpu(dx) ≈ cpu(ref_dx)
        b isa Bool || @test cpu(db) ≈ cpu(ref_db)
    end

    @testset "bias_act! forward mode: $σ" for σ in (identity, bias_act_activations...)
        for bsize in ((8,), nothing)
            test_rule(StableRNG(123), bias_act!, σ, bias_act_args(rng, (8, 4), bsize)...;
                perf_flag=:none, is_primitive=true, mode=Mooncake.ForwardMode)
        end
    end

    gpu || @testset "logsumexp Inf/NaN stability" begin
        function test_logsumexp_inf(x, dims)
            cdx = Mooncake.zero_fcodual(copy(x))
            y, pb = Mooncake.rrule!!(
                Mooncake.zero_fcodual(Core.kwcall),
                Mooncake.zero_fcodual((; dims)),
                Mooncake.zero_fcodual(logsumexp),
                cdx,
            )
            Mooncake.tangent(y) .= 1
            pb(Mooncake.NoRData())
            return Mooncake.primal(y), Mooncake.tangent(cdx)
        end

        # All Inf inputs
        y, dx = test_logsumexp_inf(Float32[Inf, Inf], 1)
        @test all(isinf.(y)) && all(y .> 0)
        @test !any(isnan.(dx))
        @test dx ≈ Float32[0.5, 0.5]

        # All Inf inputs, matrix case
        y, dx = test_logsumexp_inf(Float32[Inf Inf; Inf Inf], 1)
        @test !any(isnan.(y)) && !any(isnan.(dx))
        @test dx ≈ Float32[0.5 0.5; 0.5 0.5]

        # All -Inf inputs
        y, dx = test_logsumexp_inf(Float32[-Inf, -Inf], 1)
        @test all(isinf.(y)) && all(y .< 0)
        @test !any(isnan.(dx))
        @test dx ≈ Float32[0.5, 0.5]

        # Mixed Inf and finite inputs
        y, dx = test_logsumexp_inf(Float32[Inf, 1.0f0], 1)
        @test all(isinf.(y)) && all(y .> 0)
        @test !any(isnan.(dx))
        @test dx ≈ Float32[1.0f0, 0.0f0]

        y_nd = logsumexp(NDual{Float32,1}[
            NDual{Float32,1}(Inf32, (1.0f0,)), NDual{Float32,1}(Inf32, (0.0f0,))
        ])
        @test isinf(ndual_value(y_nd)) && ndual_value(y_nd) > 0
        @test !isnan(ndual_partial(y_nd, 1))
        @test ndual_partial(y_nd, 1) ≈ 0.5f0

        y_nd_neg = logsumexp(NDual{Float32,1}[
            NDual{Float32,1}(-Inf32, (1.0f0,)), NDual{Float32,1}(-Inf32, (0.0f0,))
        ])
        @test isinf(ndual_value(y_nd_neg)) && ndual_value(y_nd_neg) < 0
        @test !isnan(ndual_partial(y_nd_neg, 1))
        @test ndual_partial(y_nd_neg, 1) ≈ 0.5f0
    end

    # NNlib's `exp(-abs(x))` form of σ has a kink at zero that the function itself does not.
    @testset "sigmoid at zero and saturation: $f" for f in (NNlib.σ, NNlib.sigmoid_fast)
        if gpu
            # On a device the broadcast is evaluated on `NDual`s in-kernel, a code path of its own.
            test_rule(StableRNG(123), x -> sum(f.(x)), device([0.0f0, 0.5f0, -0.5f0]);
                is_primitive=false)
        else
            test_rule(StableRNG(123), f, 0.0; perf_flag=:stability)
            # The reported failure was a gradient through a broadcast containing an exact zero.
            test_rule(StableRNG(123), x -> sum(f.(x)), [0.0, 0.5, -0.5]; is_primitive=false)
            # Saturated inputs: NaN, Inf and stability, not precision — see `test_rule`'s Limitations.
            test_rule(StableRNG(123), f, 37.0; perf_flag=:stability)
            test_rule(StableRNG(123), f, 17.0f0; perf_flag=:stability)

            # Float16 collapses first, at x >= 8, where `test_rule` fails whichever formula is in
            # place: in its `ȳ·ẏ + x̄·ẋ` check `ẏ` is exactly 0 for every step (spacing 4.88e-4
            # against a true change of 3.35e-6) and `ẋ` 2.3% off at best, so compare against a
            # wider-precision reference. `Ω * (1 - Ω)` returns exactly 0 here, so these are the
            # only tests pinning σ's precision.
            ref16 = (b = big(8.0); y = inv(1 + exp(-b)); Float64(y * (1 - y)))
            x16 = NDual{Float16,1}(Float16(8), (one(Float16),))
            @test Float64(ndual_partial(f(x16), 1)) ≈ ref16 rtol = 1e-2
            d16 = Mooncake.frule!!(
                Mooncake.Dual(f, Mooncake.NoTangent()), Mooncake.Dual(Float16(8), one(Float16))
            )
            @test Float64(Mooncake.tangent(d16)) ≈ ref16 rtol = 1e-2
            _, pb16 = Mooncake.rrule!!(Mooncake.zero_fcodual(f), Mooncake.zero_fcodual(Float16(8)))
            @test Float64(pb16(one(Float16))[2]) ≈ ref16 rtol = 1e-2
        end
    end

    # `tanh_fast`'s primal discards an `ifelse` branch that overflows to `NaN`, which reverse
    # mode picked up as `0 * Inf`, poisoning `gelu`/`gelu_tanh` from `|x| ≈ 21` upwards.
    gpu || @testset "tanh_fast across its branches" begin
        test_rule(StableRNG(123), tanh_fast, 0.5; perf_flag=:stability)
        test_rule(StableRNG(123), tanh_fast, 400.0; perf_flag=:stability)
        # Below |x| ≈ 0.13 the primal switches to a polynomial while the rule stays analytic, so
        # this pins the gap between them as small enough for finite differences not to see it.
        test_rule(StableRNG(123), tanh_fast, 0.05; perf_flag=:stability)
        test_rule(StableRNG(123), x -> sum(gelu_tanh.(x)), [1.0, 25.0]; is_primitive=false)
        # `tanh_fast(::Float32)` is a different body, so none of the cases above reach it.
        # Through `gelu_tanh` the Float32 gradient goes `NaN` from |x| ≈ 258.8 without the rule.
        test_rule(StableRNG(123), gelu_tanh, 300.0f0; is_primitive=false)
        # As for σ, the saturated derivative's precision is beyond finite differences. `1 - Ω^2`
        # returns exactly 0 at Float16(6), against a true 2.46e-5, so it separates the two forms.
        ref16 = Float64(1 - tanh(big(6.0))^2)
        d16 = Mooncake.frule!!(
            Mooncake.Dual(tanh_fast, Mooncake.NoTangent()), Mooncake.Dual(Float16(6), one(Float16))
        )
        @test Float64(Mooncake.tangent(d16)) ≈ ref16 rtol = 1e-2
        _, pb16 = Mooncake.rrule!!(
            Mooncake.zero_fcodual(tanh_fast), Mooncake.zero_fcodual(Float16(6))
        )
        @test Float64(pb16(one(Float16))[2]) ≈ ref16 rtol = 1e-2
    end

    # `elu`, `celu` and `selu` compute `_expm1` whatever the sign of the input. Where it overflows
    # (`x > 88.7` for `Float32`, `709.8` for `Float64`) its derivative is `Inf` too, and a zero
    # cotangent times it was `NaN`. For `celu` the argument is `x / α`, which does so much earlier.
    @testset "elu, celu and selu above the overflow of expm1: $T" for (T, large) in (
            gpu ? [(Float32, 100.0f0)] : [(Float32, 100.0f0), (Float64, 1000.0)]
        )
        grad(f, x) = Mooncake.value_and_gradient!!(Mooncake.prepare_gradient_cache(f, x), f, x)[2][2]
        elu2(x) = elu(x, 2)
        celu_half(x) = celu(x, 0.5)
        celu_small(x) = celu(x, 0.01)
        small = T === Float32 ? T(5) : T(8)  # `small / 0.01` is past the overflow

        gpu || @testset "scalar $f" for (f, x, d) in (
                (elu, large, 1), (celu, large, 1), (selu, large, T(NNlib.selu_λ)),
                (elu2, large, 1), (celu_half, large, 1), (celu_small, small, 1),
                (elu, T(-1), exp(T(-1))), (selu, T(-1), T(NNlib.selu_λ * NNlib.selu_α) * exp(T(-1))),
            )
            @test grad(f, x) ≈ d
            frule = Mooncake.build_frule(f, x)
            @test Mooncake.tangent(frule(Mooncake.zero_dual(f), Mooncake.Dual(x, one(T)))) ≈ d
        end

        xs = device(T[1, large, -1, large, 0])
        xsmall = device(T[1, small, -1, small, 0])
        # Through the rules for NNlib's activations, and through the code of the others
        @testset "broadcast and bias_act!: $f" for (f, x) in (
                (elu, xs), (celu, xs), (selu, xs),
                (elu2, xs), (celu_half, xs), (celu_small, xsmall),
            )
            expected = gradient(x -> sum(f.(x)), cpu(x))[1]
            @test all(isfinite, expected)
            @test cpu(grad(x -> sum(f.(x)), x)) ≈ expected
            gbias(x) = sum(bias_act!(f, reshape(copy(x), 1, :), device(zeros(T, 1))))
            @test cpu(grad(gbias, x)) ≈ expected
        end
    end

    # These rules are reverse-only primitives, so forward mode traces the primal; declared for
    # both modes it would find no `frule!!` and raise. CPU arrays only: a traced GPU kernel
    # launch is a foreigncall, raising `MissingForeigncallRuleError`.
    gpu || @testset "forward mode traces reverse-only rules" begin
        x = randn(StableRNG(123), 3)
        for f in (
            # `softmax` is returned whole: its outputs sum to 1, so `1ᵀJ = 0` and a summed case
            # is blind to every error in it. `logsoftmax` does not, so summing it is a real check.
            x -> softmax(x),
            x -> softmax(x; dims=1),
            x -> sum(logsoftmax(x)),
            x -> sum(logsoftmax(x; dims=1)),
            x -> logsumexp(x),
            x -> sum(logsumexp(x; dims=1)),
            x -> sum(gather(x, [1, 3])),
        )
            test_rule(StableRNG(123), f, x; mode=Mooncake.ForwardMode, is_primitive=false)
        end
    end

    # `gather`'s GPU kernel launch does not survive the forward transform (an illegal
    # instruction no test can catch), so its rule raises instead.
    gpu && @testset "forward mode over gather raises" begin
        gather_sum(z) = sum(gather(z, [1, 3]))
        xg = device(randn(StableRNG(123), Float32, 4))
        rule = Mooncake.build_frule(gather_sum, xg)
        @test_throws ArgumentError rule(
            Mooncake.zero_dual(gather_sum), Mooncake.Dual(copy(xg), device(ones(Float32, 4)))
        )
        # `Adjoint`/`Transpose` of a GPU array are `AnyGPUArray`, so NNlib sends them to the
        # same kernel: they have to hit the guard, not slip past a bare `AbstractGPUArray`.
        @testset "$wrap" for wrap in (adjoint, transpose)
            w = wrap(copy(xg))
            r = Mooncake.build_frule(gather_sum, w)
            @test_throws ArgumentError r(Mooncake.zero_dual(gather_sum), Mooncake.zero_dual(w))
        end
    end
end
