# The loops of `softmax`, one at a time: compiled by Julia, and as a kernel on the CPU
# backend of KernelAbstractions. Nanoseconds per element on one thread, for `Float32`
# slices of 256 elements.
#
# With KernelAbstractions 0.10 the kernel is compiled by PoCL, and this tells which loops
# it vectorizes: it is where the formulations of `ka_softmax.jl` come from. With 0.9 the
# two columns are the same code.
#
#     julia --threads=1 --project=. bench_loops.jl

using NNlib, Printf
using NNlib: _exp

include(joinpath(@__DIR__, "ka_softmax.jl"))

const N = 256        # elements of a slice
const POST = 4096    # slices

# Each loop works on the slice `k` of `x` and writes to `out`.
@inline function loop_exp_round!(out, x, k)
    @inbounds @simd ivdep for i in (k-1)*N+1:k*N
        out[i] = _exp(x[i])
    end
end
@inline function loop_exp_addsub!(out, x, k)
    @inbounds @simd ivdep for i in (k-1)*N+1:k*N
        out[i] = ka_exp(x[i])
    end
end
@inline function loop_max_fastmath!(out, x, k)
    m = -Inf32
    @inbounds @simd for i in (k-1)*N+1:k*N
        m = @fastmath max(m, x[i])
    end
    @inbounds out[k] = m
end
# as in `NNlib.softmax!`
@inline function loop_max_clamped!(out, x, k)
    hi = floatmax(Float32)
    m = -hi
    @inbounds @simd for i in (k-1)*N+1:k*N
        v = x[i]
        v = ifelse(v <= hi, v, hi)
        v = ifelse(v >= -hi, v, -hi)
        m = @fastmath max(m, v)
    end
    @inbounds out[k] = m
end
@inline function loop_max_integers!(out, x, k)
    mi = ka_ordint(-Inf32)
    @inbounds @simd for i in (k-1)*N+1:k*N
        mi = max(mi, ka_ordint(x[i]))
    end
    @inbounds out[k] = ka_ordfloat(mi)
end
@inline function loop_sum!(out, x, k)
    s = 0f0
    @inbounds @simd for i in (k-1)*N+1:k*N
        s += x[i]
    end
    @inbounds out[k] = s
end
@inline function loop_scale!(out, x, k)
    @inbounds r = x[(k-1)*N+1]
    @inbounds @simd ivdep for i in (k-1)*N+1:k*N
        out[i] *= r
    end
end
@inline loop_softmax!(out, x, k) = ka_slice!(out, x, 0, (k - 1) * N, N)

const LOOPS = [
    ("exp, round",              loop_exp_round!),
    ("exp, add and subtract",   loop_exp_addsub!),
    ("max, @fastmath",          loop_max_fastmath!),
    ("max, @fastmath, clamped", loop_max_clamped!),
    ("max, on integers",        loop_max_integers!),
    ("sum",                     loop_sum!),
    ("scale",                   loop_scale!),
    ("softmax of a slice",      loop_softmax!),
]

function julia!(loop!::L, out, x) where {L}
    for k in 1:POST
        loop!(out, x, k)
    end
end

# One work-item runs every slice: no workgroup to speak of, and no thread but one.
@kernel inbounds=true function loops_kernel!(loop!, out, x)
    for k in 1:POST
        loop!(out, x, k)
    end
end
@kernel inbounds=true function loops_inplace_kernel!(loop!, x)
    for k in 1:POST
        loop!(x, x, k)
    end
end

best(f) = (f(); minimum(@elapsed(f()) for _ in 1:50))
nspe(t) = t * 1e9 / (N * POST)

function main()
    println("CPU:           ", Sys.cpu_info()[1].model)
    println("Julia:         ", VERSION, ", ", Threads.nthreads(), " threads")
    println("KernelAbstractions: ", pkgversion(KA))
    backend = KA.CPU()
    x = randn(Float32, N, POST)
    out = similar(x)
    @printf("\n%-34s %10s %10s %10s\n", "loop (ns per element)", "Julia", "kernel", "ratio")
    for (name, loop!) in LOOPS
        tj = nspe(best(() -> julia!(loop!, out, x)))
        tk = nspe(best(() -> loops_kernel!(backend, 1)(loop!, out, x; ndrange=1)))
        @printf("%-34s %10.3f %10.3f %9.1fx\n", name, tj, tk, tk / tj)
        flush(stdout)
    end
    # in place: as a kernel taking the array once, and as one given it twice
    y = copy(x)
    tj = nspe(best(() -> julia!(loop_softmax!, y, y)))
    tk = nspe(best(() -> loops_inplace_kernel!(backend, 1)(loop_softmax!, y; ndrange=1)))
    @printf("%-34s %10.3f %10.3f %9.1fx\n", "softmax in place, one argument", tj, tk, tk / tj)
    tk = nspe(best(() -> loops_kernel!(backend, 1)(loop_softmax!, y, y; ndrange=1)))
    @printf("%-34s %10.3f %10.3f %9.1fx\n", "softmax in place, two arguments", tj, tk, tk / tj)
end

main()
