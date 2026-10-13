# CPU multi-head attention benchmark: NNlib's fused implementation against the kernel
# of the GPU backends run on the CPU, and against the materialized implementation.
#
#   * nnlib      NNlib `scaled_dot_product_attention`: the fused implementation, which
#       walks the keys in tiles and never forms the attention weights, in the forward
#       pass or in the gradient. Memory is O(seq).
#   * nnlib-ka   the KernelAbstractions kernel NNlib launches on the GPU backends,
#       launched on the CPU backend instead. Forward pass only. Needs KernelAbstractions
#       0.10, whose CPU backend runs on PoCL. Memory is O(seq).
#   * nnlib-mat  the same function kept on the materialized implementation: the full
#       (kv_len, q_len, nheads, batch) weights from two `batched_mul` and a `softmax`.
#       Memory is O(seq²).
#
# All take q,k,v = (head_dim, nheads, seq, batch) = (E, H, L, B). `bench_torch.py` is
# the PyTorch counterpart.
#
# Reports the wall time in ms and the memory allocated, for the forward pass and for the
# full Zygote gradient (forward + backward).
#
# Run (whole suite), with as many threads as the comparison is meant for:
#     julia --threads=auto --project=. bench_attention.jl
# A single combination:
#     julia --threads=16 --project=. bench_attention.jl --dtypes f32 --causal true --sizes gpt2-ish
# See all options:
#     julia --project=. bench_attention.jl --help

using NNlib, BenchmarkTools, Printf, ArgParse, LinearAlgebra

# Only the gradient pass needs Zygote, so that the forward one also runs in an
# environment without it: with it, KernelAbstractions 0.10 is not resolved yet.
const HAS_ZYGOTE = Base.find_package("Zygote") !== nothing
if HAS_ZYGOTE
    using Zygote
end

# (name, head_dim E, seq_len L, nheads H, batch B). Self-attention: q_len=kv_len=L.
# The configurations of the GPU benchmark, up to a Llama3-8B attention layer.
const CONFIGS = [
    ("tiny",          64,  128,  4, 8),
    ("small",         64,  512,  8, 8),
    ("gpt2-ish",      64, 1024, 12, 4),
    ("llama3 L=2k",  128, 2048, 32, 1),
    ("llama3 L=4k",  128, 4096, 32, 1),
    ("llama3 L=8k",  128, 8192, 32, 1),
]

const DTYPE_ALIASES = Dict(
    "f16" => Float16, "float16" => Float16, "half" => Float16,
    "f32" => Float32, "float32" => Float32,
    "f64" => Float64, "float64" => Float64,
)
const DTYPE_ORDER = (Float16, Float32, Float64)

# NNlib only takes the fused implementation without dropout: a no-op `fdrop` other than
# `identity` keeps it on the materialized one, at no extra cost.
nofuse(x) = x

const KA = NNlib.KernelAbstractions

# With KernelAbstractions 0.9 a kernel written for the GPU does not compile for the CPU.
const KA_ON_CPU = pkgversion(KA) >= v"0.10"

# `fused_attention!` dispatches CPU arrays to the Julia tasks: this calls its method for
# the other backends on the CPU one.
function nnlib_ka(q::Array{T}, k, v, causal) where {T}
    A = NNlib.attention_compute_type(T)
    o = similar(q, size(v, 1), size(q, 2), size(q, 3), size(q, 4))
    args = (o, nothing, q, k, v, nothing, nothing, nothing, nothing, inv(sqrt(A(size(q, 1)))), causal)
    invoke(NNlib.fused_attention!, Tuple{KA.Backend,map(typeof, args)...}, KA.CPU(), args...)
    return o
end

const IMPLS = Dict(
    "nnlib"     => (q, k, v, causal) -> scaled_dot_product_attention(q, k, v; is_causal=causal),
    "nnlib-ka"  => nnlib_ka,
    "nnlib-mat" => (q, k, v, causal) -> scaled_dot_product_attention(q, k, v; is_causal=causal, fdrop=nofuse),
)
const IMPL_ORDER = ["nnlib", "nnlib-ka", "nnlib-mat"]

forward(f, q, k, v, causal) = f(q, k, v, causal)
fullgradient(f, q, k, v, causal) = Zygote.gradient((q, k, v) -> sum(f(q, k, v, causal)), q, k, v)
const PASSES = Dict("forward" => ("FORWARD", forward), "gradient" => ("FULL GRADIENT, fwd+bwd", fullgradient))
const PASS_ORDER = ["forward", "gradient"]

# (minimum time in ms, allocated MiB) of a thunk, or `nothing` if it does not run
function timed(f, seconds)
    try
        f()                                 # warmup / compile
        trial = BenchmarkTools.run(@benchmarkable($f()); seconds=seconds, evals=1)
        return minimum(trial).time / 1e6, trial.memory / 2^20
    catch e
        @warn "unsupported path → n/s: $(first(split(sprint(showerror, e), '\n')))"
        return nothing
    end
end

function sanity(T, impls)
    q = randn(T, 32, 4, 70, 2) ./ 4
    k = randn(T, 32, 4, 150, 2) ./ 4
    v = randn(T, 32, 4, 150, 2)
    ref = IMPLS[first(impls)](q, k, v, true)
    parts = map(impls) do name
        d = maximum(abs, Float64.(ref) .- Float64.(IMPLS[name](q, k, v, true)))
        @sprintf("%s=%.2e", name, d)
    end
    @printf("sanity %-8s (vs %s, fused: %s)  %s\n", string(T), first(impls),
            NNlib.is_fused_attention(q, k, v; is_causal=true), join(parts, "  "))
end

fmt(x) = x === nothing ? "n/s" : @sprintf("%.3f", x)

function run(T, causal, impls, configs, seconds, pass)
    title, op = PASSES[pass]
    println("\n-- $title --")
    header = rpad("config", 13) * lpad("L", 6) * lpad("H", 4) * " |"
    for name in impls; header *= lpad(name * " ms", 15); end
    header *= " |"
    for name in impls; header *= lpad(name * " MiB", 15); end
    length(impls) > 1 && (header *= " |")
    for name in impls[2:end]; header *= lpad(name * "/" * impls[1], 18); end
    println(header)
    for (name, E, L, H, B) in configs
        q = randn(T, E, H, L, B); k = randn(T, E, H, L, B); v = randn(T, E, H, L, B)
        res = map(impls) do nm
            pass == "gradient" && nm == "nnlib-ka" && return nothing
            timed(() -> op(IMPLS[nm], q, k, v, causal), seconds)
        end
        line = rpad(name, 13) * lpad(string(L), 6) * lpad(string(H), 4) * " |"
        for r in res; line *= lpad(fmt(r === nothing ? nothing : r[1]), 15); end
        line *= " |"
        for r in res; line *= lpad(fmt(r === nothing ? nothing : r[2]), 15); end
        length(impls) > 1 && (line *= " |")
        for r in res[2:end]
            line *= lpad((res[1] === nothing || r === nothing) ? "-" : @sprintf("%.1fx", r[1] / res[1][1]), 18)
        end
        println(line)
        flush(stdout)
    end
end

function parse_cli(argv)
    s = ArgParseSettings(description="CPU multi-head attention benchmark: NNlib fused vs the GPU kernel on the CPU vs materialized.")
    @add_arg_table! s begin
        "--dtypes"
            help = "comma list of dtypes: f16,f32,f64. Default: f32."
            default = "f32"
        "--impls"
            help = "comma list of implementations: $(join(IMPL_ORDER, ',')). Default: all."
            default = join(IMPL_ORDER, ',')
        "--causal"
            help = "comma list of causal modes: false,true. Default: both."
            default = "false,true"
        "--sizes"
            help = "comma list of config names (see --list). Default: all."
            default = "all"
        "--passes"
            help = "comma list of what to time: forward,gradient. Default: both."
            default = "forward,gradient"
        "--seconds"
            help = "per-measurement time budget."
            arg_type = Float64
            default = 2.0
        "--list"
            help = "list available configs/dtypes/impls and exit."
            action = :store_true
    end
    return parse_args(argv, s)
end

splitcsv(s) = strip.(split(s, ','; keepempty=false))

function main(argv)
    opts = parse_cli(argv)
    if opts["list"]
        println("configs: ", join(first.(CONFIGS), ", "))
        println("dtypes:  f16, f32, f64")
        println("impls:   ", join(IMPL_ORDER, ", "))
        return
    end
    dtypes = [DTYPE_ALIASES[lowercase(d)] for d in splitcsv(opts["dtypes"])]
    dtypes = filter(in(dtypes), collect(DTYPE_ORDER))
    impls = filter(in(map(lowercase, splitcsv(opts["impls"]))), IMPL_ORDER)
    if "nnlib-ka" in impls && !KA_ON_CPU
        println("nnlib-ka needs KernelAbstractions 0.10 (found ", pkgversion(KA), "): skipped")
        filter!(!=("nnlib-ka"), impls)
    end
    causals = unique(c == "true" for c in splitcsv(lowercase(opts["causal"])))
    names = lowercase(strip(opts["sizes"])) == "all" ? first.(CONFIGS) : splitcsv(opts["sizes"])
    configs = filter(c -> c[1] in names, CONFIGS)
    isempty(configs) && error("no configs matched --sizes=$(opts["sizes"])")
    passes = filter(in(map(lowercase, splitcsv(opts["passes"]))), PASS_ORDER)
    if "gradient" in passes && !HAS_ZYGOTE
        println("the gradient pass needs Zygote, which this environment does not have: skipped")
        filter!(!=("gradient"), passes)
    end

    println("CPU:           ", Sys.cpu_info()[1].model, " (", Sys.CPU_THREADS, " hardware threads)")
    println("Julia:         ", VERSION, ", ", Threads.nthreads(), " threads")
    println("BLAS:          ", BLAS.get_config().loaded_libs[1].libname |> basename, ", ", BLAS.get_num_threads(), " threads")
    println("NNlib:         ", pkgversion(NNlib))
    println("KernelAbstractions: ", pkgversion(KA))
    println("selection:     dtypes=", join(string.(dtypes), ","), "  impls=", join(impls, ","),
            "  causal=", join(string.(causals), ","), "  sizes=", join(first.(configs), ","))

    for T in dtypes; sanity(T, impls); end
    for T in dtypes, causal in causals
        @printf("\n%s\n### dtype=%s  causal=%s\n%s\n", "#"^92, string(T), causal, "#"^92)
        for pass in passes
            run(T, causal, impls, configs, opts["seconds"], pass)
        end
    end
    length(impls) > 1 && println("\nx/$(impls[1]) = time relative to $(impls[1]) (>1 → $(impls[1]) faster).")
end

main(ARGS)
