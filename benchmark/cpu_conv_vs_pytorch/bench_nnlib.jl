# NNlib CPU conv benchmark — counterpart to bench_torch.py (issue #234).
#
# Run with 4 threads:
#     julia --threads=4 --project=. bench_nnlib.jl
#
# Needs NNlib + BenchmarkTools in the active project.

using NNlib
using BenchmarkTools
using LinearAlgebra
using Random

Random.seed!(1234)
# NNlib's conv parallelism is Julia-task based, so a fair "4 threads" run is
# `julia --threads=4` with BLAS pinned to 1 (otherwise julia × BLAS threads
# oversubscribe). This matches PyTorch's `torch.set_num_threads(4)`.
BLAS.set_num_threads(1)

# (name, in_ch, out_ch, kernel, stride, pad, H, W, batch)
const CASES = [
    ("issue234 7x7 s2 p3", 3, 64, 7, 2, 3, 224, 224, 2),
    ("3x3 s1 p1 c64", 64, 64, 3, 1, 1, 56, 56, 2),
    ("3x3 s1 p1 c128", 128, 128, 3, 1, 1, 28, 28, 2),
    ("1x1 s1 p0 c256", 256, 256, 1, 1, 0, 14, 14, 2),
]

function main()
    println("Julia threads: ", Threads.nthreads())
    println("BLAS threads:  ", BLAS.get_num_threads())
    println(rpad("case", 22), lpad("min fwd (ms)", 14), lpad("mem (MiB)", 12))
    for (name, ci, co, k, s, p, H, W, b) in CASES
        # NNlib uses WHCN layout (vs PyTorch NCHW)
        x = randn(Float32, W, H, ci, b)
        w = randn(Float32, k, k, ci, co)
        cdims = DenseConvDims(x, w; stride=(s, s), padding=(p, p, p, p))
        t = @benchmark conv($x, $w, $cdims) samples=80 evals=1
        min_ms = minimum(t).time / 1e6   # min ≈ uncontended time on a shared box
        mem_mib = t.memory / 2^20
        println(rpad(name, 22), lpad(round(min_ms; digits=4), 14),
                lpad(round(mem_mib; digits=3), 12))
    end
end

main()
