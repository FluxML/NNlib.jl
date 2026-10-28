using CUDA, NNlib, cuDNN, AcceleratedKernels, BenchmarkTools, Printf
using BFloat16s: BFloat16

const SECONDS = parse(Float64, get(ENV, "BENCH_SECONDS", "0.2"))
const PART = isempty(ARGS) ? "all" : only(ARGS)
PART == "--list" && (println("reductions\nbatched_mul\nattention\nall"); exit())
PART in ("all", "reductions", "batched_mul", "attention") || error("Expected reductions, batched_mul, attention, or all")

function timed(f)
    f()
    CUDA.synchronize()
    return minimum(@benchmark CUDA.@sync($f()) seconds=SECONDS evals=1).time / 1000
end

function compare(label, materialized, lazy; rtol=2f-5)
    a = copy(materialized())
    b = lazy()
    @assert isapprox(Array(a), Array(b); rtol, atol=1f-5)
    before, after = timed(materialized), timed(lazy)
    @printf("%-65s %10.1f %10.1f %8.2f\n", label, before, after, before / after)
    flush(stdout)
end

function lazy_logsumexp(x; dims)
    m = NNlib.fast_maximum(x; dims)
    return m .+ log.(sum(NNlib._exp, Base.broadcasted(-, x, m); dims))
end

function materialized_logsumexp(x; dims)
    m = NNlib.fast_maximum(x; dims)
    return m .+ log.(sum(NNlib._exp.(x .- m); dims))
end

function reductions()
    println("\nReduction: materialized / lazy (µs), ratio materialized/lazy")
    for T in (Float32, BFloat16), (sz, dims) in
            (((50304, 8192), 1), ((32000, 1024), 1), ((512, 512, 64), 1), ((8192, 768), 2))
        x = T.(CUDA.randn(Float32, sz))
        compare("logsumexp $T $sz dims=$dims", () -> materialized_logsumexp(x; dims), () -> lazy_logsumexp(x; dims))
        CUDA.unsafe_free!(x)
        CUDA.reclaim()
    end
    # The expressions reduced by _broadcast_sum in #830, before fused normalization replaced it.
    for T in (Float32, BFloat16), (sz, dims) in
            (((768, 1024, 8), 1), ((4096, 2048), 1), ((4096, 2048), 2), ((64, 64, 32, 16), (1, 2)))
        x = T.(CUDA.randn(Float32, sz))
        μ = sum(Float32, x; dims) ./ prod(size(x, d) for d in (dims isa Integer ? (dims,) : dims))
        for (name, f, args) in (("square", x -> abs2(Float32(x)), (x,)),
                                ("centered square", (x, μ) -> abs2(Float32(x) - μ), (x, μ)))
            bc = Broadcast.instantiate(Broadcast.broadcasted(f, args...))
            compare("$name $T $sz dims=$dims", () -> sum(copy(bc); dims), () -> sum(bc; dims))
        end
        CUDA.unsafe_free!(x)
        CUDA.reclaim()
    end
end

function batched()
    println("\nBatched multiplication: per-slice / batched cuBLAS (µs), ratio")
    for T in (Float16, BFloat16), (m, k, n, batch) in ((32, 32, 32, 128), (128, 128, 128, 32), (64, 512, 64, 64))
        A, B = T.(CUDA.randn(Float32, m, k, batch)), T.(CUDA.randn(Float32, k, n, batch))
        C = similar(A, m, n, batch)
        compare("$T ($m, $k, $n) batch=$batch",
            () -> NNlib.batched_mul_generic!(C, A, B), () -> NNlib.batched_mul!(C, A, B); rtol=2f-2)
    end
end

function attention()
    println("\nAttention: NNlib kernel / cuDNN dispatch (µs), ratio")
    for (name, lq, lk, batch, kws) in
            (("decode", 1, 16384, 1, (; is_causal=:bottom_right)),
             ("decode batch 8", 1, 16384, 8, (; is_causal=:bottom_right)),
             ("chunked prefill", 512, 4096, 1, (; is_causal=:bottom_right)),
             ("window", 8192, 8192, 1, (; is_causal=true, window=1024)),
             ("packed", 8192, 8192, 1, (; is_causal=true, q_segments=fill(1024, 8))))
        q = Float16.(CUDA.randn(Float32, 128, 32, lq, batch) ./ 4)
        k = Float16.(CUDA.randn(Float32, 128, 8, lk, batch) ./ 4)
        v = Float16.(CUDA.randn(Float32, 128, 8, lk, batch) ./ 4)
        packed = haskey(kws, :q_segments)
        xs = packed ? map(x -> reshape(x, size(x)[1:3]), (q, k, v)) : (q, k, v)
        ci = NNlib.collapse_attention_inputs(xs..., nothing, nothing, 0, nothing, nothing,
            get(kws, :q_segments, nothing), get(kws, :q_segments, nothing))
        cq, ck, cv, bias, mask, q_offset, q_lengths, kv_lengths, segments = ci
        opts = (; fdrop=identity, mask, scale=nothing, is_causal=kws.is_causal,
                 window=get(kws, :window, nothing), q_offset, q_lengths, kv_lengths, segments)
        generic = () -> NNlib.generic_attention(cq, ck, cv, bias; opts...)
        dispatched = () -> NNlib._scaled_dot_product_attention(cq, ck, cv, bias; opts...)
        @assert isapprox(Float32.(Array(generic())), Float32.(Array(dispatched())); rtol=2f-3, atol=1f-4)
        before, after = timed(generic), timed(dispatched)
        @printf("%-65s %10.1f %10.1f %8.2f\n", name, before, after, before / after)
        flush(stdout)
        CUDA.reclaim()
    end
end

println("Julia $VERSION, ", CUDA.name(CUDA.device()))
println("CUDA ", pkgversion(CUDA), ", cuDNN ", pkgversion(cuDNN), ", AcceleratedKernels ", pkgversion(AcceleratedKernels))
println("CUDA runtime ", CUDA.runtime_version(), ", cuDNN runtime ", cuDNN.version())
PART in ("all", "reductions") && reductions()
PART in ("all", "batched_mul") && batched()
PART in ("all", "attention") && attention()
