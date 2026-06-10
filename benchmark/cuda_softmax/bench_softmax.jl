# GPU softmax / logsoftmax gradient benchmark — the end-to-end Zygote AD path.
#
# This times what Flux actually runs: the forward `softmax`/`logsoftmax` and the
# *Zygote* reverse pass (the rrule pullback), NOT the `∇softmax` kernels called
# directly. That distinction matters because the backward is only as fast as
# whatever the rrule routes to:
#
#   * before #718 the rrule called the generic broadcast `∇softmax_data(dy, y)`,
#     so the cuDNN backward (and its #513 INSTANCE-mode fix) was never reached
#     from AD — Zygote always used the broadcast kernel;
#   * after  #718 the rrule calls `∇softmax(dy, y)` → `∇softmax!`, which the cuDNN
#     extension overloads, so for a leading contiguous softmax axis (dims=1 /
#     Colon) the Zygote backward now runs cuDNN INSTANCE mode.
#
# Run the SAME script on `master` and on the PR branch to get the before/after.
#
# Run:
#     julia --project=. bench_softmax.jl
# Pin a GPU (UUID form is robust when other cards on the box are unavailable):
#     CUDA_VISIBLE_DEVICES=GPU-<uuid> julia --project=. bench_softmax.jl

using CUDA, cuDNN, NNlib, Zygote, BenchmarkTools, Printf

# Array shapes to benchmark. softmax acts along axis `dims`, so for dims=1 the
# softmax-vector length is size[1] and for dims=2 it is size[2]; the remaining
# axes are batch. The length of the softmax axis is what drives the
# cuDNN-vs-broadcast crossover on the backward pass.
const SIZES = [(256,10,32), (256,1000), (1000,1000), (100,10000),
               (10000,100), (32000,64), (1000,128)]

# time a thunk, in µs (minimum, GPU-synchronized)
us(f) = (@belapsed CUDA.@sync($f()) seconds=0.7 evals=1) * 1e6

# A representative scalar loss so the reverse pass carries a non-trivial cotangent.
sm_loss(z; dims)  = sum(abs2, softmax(z; dims))
ls_loss(z; dims)  = sum(abs2, logsoftmax(z; dims))

function bench_all(sizes, dims)
    map(sizes) do s
        x = CUDA.randn(Float32, s...)

        # Build the Zygote pullbacks once; timing `back(dȳ)` isolates the reverse
        # pass (the part #718 changes) from the forward. `back` runs the generated
        # adjoint, which calls NNlib's `softmax_pullback` → `∇softmax` — i.e. it
        # goes through Zygote, never the kernel directly.
        ysm, sm_back = Zygote.pullback(z -> softmax(z; dims), x)
        yls, ls_back = Zygote.pullback(z -> logsoftmax(z; dims), x)
        d̄ = CUDA.randn(Float32, size(ysm)...)

        (; size = s,
           # forward
           f_sm = us(() -> softmax(x; dims)),
           f_ls = us(() -> logsoftmax(x; dims)),
           # Zygote reverse pass (pullback only)
           b_sm = us(() -> sm_back(d̄)),
           b_ls = us(() -> ls_back(d̄)),
           # full Zygote.gradient of a scalar loss (forward + reverse)
           g_sm = us(() -> Zygote.gradient(z -> sm_loss(z; dims), x)),
           g_ls = us(() -> Zygote.gradient(z -> ls_loss(z; dims), x)))
    end
end

function check_finite(dims)
    x = CUDA.randn(Float32, 512, 700)
    gsm = Zygote.gradient(z -> sm_loss(z; dims), x)[1]
    gls = Zygote.gradient(z -> ls_loss(z; dims), x)[1]
    @printf("sanity dims=%d  softmax grad finite=%s  logsoftmax grad finite=%s\n",
            dims, all(isfinite, Array(gsm)), all(isfinite, Array(gls)))
end

function show_table(res)
    @printf("%-14s %9s %9s %9s %9s %9s %9s\n",
            "size", "fwd-sm", "fwd-ls", "bwd-sm", "bwd-ls", "grad-sm", "grad-ls")
    for r in res
        @printf("%-14s %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n",
                string(r.size), r.f_sm, r.f_ls, r.b_sm, r.b_ls, r.g_sm, r.g_ls)
    end
end

function main()
    @assert CUDA.functional() "CUDA is not functional"
    println("GPU:   ", CUDA.name(CUDA.device()))
    println("cuDNN: ", cuDNN.version())
    for dims in (1, 2); check_finite(dims); end
    for dims in (1, 2)
        res = bench_all(SIZES, dims)
        println("\n", "#"^72, "\n### dims=$dims   (times in µs, Float32; fwd/bwd/full-grad)\n", "#"^72)
        show_table(res)
    end
    println("\nbwd-* = Zygote pullback only; grad-* = full Zygote.gradient of sum(abs2, .).")
end

main()
