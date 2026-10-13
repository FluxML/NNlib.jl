# GPU multi-head attention benchmark: NNlib (fused / materialized) vs LuxLib.
#
# Implementations of the SAME scaled dot-product (multi-head) attention (scale 1/√E):
#
#   * nnlib      NNlib `scaled_dot_product_attention`, as it dispatches: cuDNN's fused
#       kernel for Float16/BFloat16, NNlib's own fused kernel for the other eltypes.
#       Neither materializes the attention weights, in the forward pass or in the
#       gradient: memory is O(seq).
#   * nnlib-ka   NNlib's own fused kernel (KernelAbstractions) for every eltype, that is
#       also where cuDNN would take the call.
#   * nnlib-mat  the same function kept on the materialized implementation: the full
#       (kv_len, q_len, nheads, batch) weights from two `batched_mul` and a `softmax`.
#       Memory is O(seq²).
#   * lux        LuxLib `scaled_dot_product_attention`, also a materialized softmax path.
#
# All of them take q,k,v = (head_dim, nheads, seq, batch) = (E, H, L, B). `bench_torch.py`
# is the PyTorch counterpart.
#
# Reports forward and full Zygote-gradient (forward+backward) wall time in ms,
# GPU-synchronized, and the GPU memory each of the two allocates. "OOM" = ran out of GPU
# memory; "n/s" = that path can't run the config (echoed once to stderr).
#
# Run (whole suite):
#     julia --project=. bench_attention.jl
# Benchmark a single dtype/impl combination without rerunning everything:
#     julia --project=. bench_attention.jl --dtypes f32 --impls nnlib,nnlib-mat --causal false
#     julia --project=. bench_attention.jl --dtypes f16,bf16 --impls nnlib,nnlib-ka --sizes tiny,small
# Pin a GPU:
#     CUDA_VISIBLE_DEVICES=GPU-<uuid> julia --project=. bench_attention.jl ...
# See all options:
#     julia --project=. bench_attention.jl --help

using CUDA, cuDNN, NNlib, LuxLib, Zygote, BenchmarkTools, Printf, ArgParse
using BFloat16s: BFloat16

# ---------------------------------------------------------------------------
# registries: configs, dtypes, implementations
# ---------------------------------------------------------------------------

# (name, head_dim E, seq_len L, nheads H, batch B). Self-attention: q_len=kv_len=L.
# Climbs from tiny up to a single Llama3-8B attention layer (E=128, H=32).
const CONFIGS = [
    ("tiny",          64,  128,  4, 8),
    ("small",         64,  512,  8, 8),
    ("gpt2-ish",      64, 1024, 12, 4),
    ("llama3 L=2k",  128, 2048, 32, 1),
    ("llama3 L=4k",  128, 4096, 32, 1),
    ("llama3 L=8k",  128, 8192, 32, 1),
]

const DTYPE_ALIASES = Dict(
    "f16" => Float16,  "float16" => Float16,  "fp16" => Float16,  "half" => Float16,
    "bf16" => BFloat16, "bfloat16" => BFloat16,
    "f32" => Float32,  "float32" => Float32,  "fp32" => Float32,
)
const DTYPE_ORDER = (Float16, BFloat16, Float32)   # canonical print order

# An implementation: how to run forward / full-gradient on the inputs (q, k, v, causal).
struct Impl
    name::String
    fwd::Function    # (inputs) -> output
    grad::Function   # (inputs) -> gradient w.r.t. q,k,v
end

# NNlib only takes a fused kernel without dropout: a no-op `fdrop` other than
# `identity` keeps it on the materialized implementation, at no extra cost.
nofuse(x) = x

# NNlib's own fused kernel, also for the calls its cuDNN extension takes.
nnlib_ka(q, k, v, causal) = NNlib.generic_attention(q, k, v, nothing; fdrop=identity,
    mask=nothing, scale=nothing, is_causal=causal, q_lengths=nothing, kv_lengths=nothing)

# TODO: add CuTile.jl to the comparison.
const IMPLS = Dict(
    "nnlib" => Impl("nnlib",
        I -> NNlib.scaled_dot_product_attention(I[1], I[2], I[3]; is_causal=I[4]),
        I -> Zygote.gradient((q, k, v) ->
                sum(NNlib.scaled_dot_product_attention(q, k, v; is_causal=I[4])), I[1], I[2], I[3])),
    "nnlib-ka" => Impl("nnlib-ka",
        I -> nnlib_ka(I...),
        I -> Zygote.gradient((q, k, v) -> sum(nnlib_ka(q, k, v, I[4])), I[1], I[2], I[3])),
    "nnlib-mat" => Impl("nnlib-mat",
        I -> NNlib.scaled_dot_product_attention(I[1], I[2], I[3]; is_causal=I[4], fdrop=nofuse),
        I -> Zygote.gradient((q, k, v) ->
                sum(NNlib.scaled_dot_product_attention(q, k, v; is_causal=I[4], fdrop=nofuse)),
                I[1], I[2], I[3])),
    "lux" => Impl("lux",
        I -> LuxLib.scaled_dot_product_attention(I[1], I[2], I[3]; is_causal=I[4])[1],
        I -> Zygote.gradient((q, k, v) ->
                sum(LuxLib.scaled_dot_product_attention(q, k, v; is_causal=I[4])[1]), I[1], I[2], I[3])),
)
# canonical column order; the first selected one is the reference for the ratios
const IMPL_ORDER = ["nnlib", "nnlib-ka", "nnlib-mat", "lux"]

# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

# time a thunk in ms (minimum, GPU-synced). Returns a positive time on success,
# NaN for an out-of-memory failure ("OOM"), or -1.0 if that path can't run this
# config ("n/s"). The first reason for each distinct message is echoed to stderr
# so nothing is hidden silently.
const SEEN_FAILURES = Set{String}()
function timed(f, seconds)
    try
        CUDA.reclaim()
        CUDA.@sync f()                      # warmup / compile
        b = @benchmarkable CUDA.@sync($f())
        trial = BenchmarkTools.run(b; seconds=seconds, evals=1)
        return minimum(trial).time / 1e6    # ns -> ms
    catch e
        msg = lowercase(sprint(showerror, e))
        if occursin("out of memory", msg) || occursin("outofgpumemory", msg) ||
           occursin("out of gpu memory", msg)
            return NaN
        end
        reason = first(split(sprint(showerror, e), '\n'))
        if reason ∉ SEEN_FAILURES
            push!(SEEN_FAILURES, reason)
            @warn "unsupported path → n/s: $reason"
        end
        return -1.0
    finally
        CUDA.reclaim()
    end
end

# GPU memory a call allocates, in MiB, with the conventions of `timed`
function allocated(f)
    try
        CUDA.reclaim()
        return CUDA.@allocated(CUDA.@sync f()) / 2^20
    catch e
        msg = lowercase(sprint(showerror, e))
        return occursin("out of memory", msg) || occursin("out of gpu memory", msg) ? NaN : -1.0
    finally
        CUDA.reclaim()
    end
end

# ---------------------------------------------------------------------------
# sanity check (only over the selected implementations)
# ---------------------------------------------------------------------------

function sanity(T, impls)
    E, L, H, B = 32, 64, 4, 2
    q = CUDA.randn(T, E, H, L, B); k = CUDA.randn(T, E, H, L, B); v = CUDA.randn(T, E, H, L, B)
    outs = Dict{String,Any}()
    for name in impls
        im = IMPLS[name]
        outs[name] = try
            Float32.(Array(im.fwd((q, k, v, false))))
        catch
            nothing
        end
    end
    ran = [name for name in impls if outs[name] !== nothing]
    ref = isempty(ran) ? nothing : outs[first(ran)]
    parts = map(impls) do name
        d = (outs[name] === nothing || ref === nothing) ? "n/s" :
            @sprintf("%.2e", maximum(abs, ref .- outs[name]))
        "$name=$d"
    end
    refname = isempty(ran) ? "—" : first(ran)
    @printf("sanity %-9s  (vs %s)  %s\n", string(T), refname, join(parts, "  "))
end

# ---------------------------------------------------------------------------
# run + tabulate one (dtype, causal) block
# ---------------------------------------------------------------------------

fmt(x) = isnan(x) ? "OOM" : x < 0 ? "n/s" : @sprintf("%.3f", x)
ok(x) = !isnan(x) && x > 0
spd(num, den) = (ok(num) && ok(den)) ? @sprintf("%.2fx", num / den) : "-"

function print_table(title, impls, rows, key)
    ref, others = first(impls), impls[2:end]
    header = rpad("config", 13) * lpad("L", 6) * lpad("H", 4) * " |"
    for name in impls; header *= lpad(name, 11); end
    isempty(others) || (header *= " |"; for name in others; header *= lpad(name * "/" * ref, 17); end)
    println("\n-- $title --")
    println(header)
    for r in rows
        line = rpad(r.name, 13) * lpad(string(r.L), 6) * lpad(string(r.H), 4) * " |"
        for name in impls; line *= lpad(fmt(r.t[name][key]), 11); end
        if !isempty(others)
            line *= " |"
            for name in others; line *= lpad(spd(r.t[name][key], r.t[ref][key]), 17); end
        end
        println(line)
    end
end

function run(T, causal, impls, configs, seconds)
    @printf("\n%s\n### dtype=%s  causal=%s\n%s\n", "#"^104, string(T), causal, "#"^104)
    rows = map(configs) do (name, E, L, H, B)
        q = CUDA.randn(T, E, H, L, B); k = CUDA.randn(T, E, H, L, B); v = CUDA.randn(T, E, H, L, B)
        t = Dict{String,Any}()
        for nm in impls
            im = IMPLS[nm]
            I = (q, k, v, causal)
            t[nm] = (; fwd  = timed(() -> im.fwd(I), seconds),
                      grad = timed(() -> im.grad(I), seconds),
                      mem  = allocated(() -> im.fwd(I)),
                      gmem = allocated(() -> im.grad(I)))
        end
        q = k = v = nothing; CUDA.reclaim()
        (; name, L, H, t)
    end
    print_table("FORWARD (ms)", impls, rows, :fwd)
    print_table("FULL GRADIENT, fwd+bwd (ms)", impls, rows, :grad)
    print_table("FORWARD, GPU memory allocated (MiB)", impls, rows, :mem)
    print_table("FULL GRADIENT, GPU memory allocated (MiB)", impls, rows, :gmem)
    flush(stdout)   # show each block as it completes, also when writing to a file
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function parse_cli(argv)
    s = ArgParseSettings(description="GPU multi-head attention benchmark: NNlib (fused / materialized) vs LuxLib.")
    @add_arg_table! s begin
        "--dtypes"
            help = "comma list of dtypes: f16,bf16,f32 (aliases ok). Default: all."
            default = "f16,bf16,f32"
        "--impls"
            help = "comma list of implementations: $(join(IMPL_ORDER, ',')). Default: all."
            default = join(IMPL_ORDER, ',')
        "--causal"
            help = "comma list of causal modes: false,true (or 'both'). Default: both."
            default = "false,true"
        "--sizes"
            help = "comma list of config names (see --list). Default: all."
            default = "all"
        "--seconds"
            help = "per-measurement time budget for @belapsed."
            arg_type = Float64
            default = 1.0
        "--list"
            help = "list available configs/dtypes/impls and exit."
            action = :store_true
    end
    return parse_args(argv, s)
end

splitcsv(s) = strip.(split(s, ','; keepempty=false))

function resolve(opts)
    dtypes = map(splitcsv(opts["dtypes"])) do d
        haskey(DTYPE_ALIASES, lowercase(d)) || error("unknown dtype '$d' (try f16,bf16,f32)")
        DTYPE_ALIASES[lowercase(d)]
    end
    dtypes = filter(in(unique(dtypes)), collect(DTYPE_ORDER))   # canonical order, dedup

    impls = map(lowercase, splitcsv(opts["impls"]))
    for im in impls; im in IMPL_ORDER || error("unknown impl '$im' (try $(join(IMPL_ORDER, ',')))"); end
    impls = filter(in(impls), IMPL_ORDER)

    cstr = lowercase(strip(opts["causal"]))
    causals = cstr == "both" ? [false, true] :
              [c == "true" for c in splitcsv(cstr)]
    causals = unique(causals)

    names = lowercase(strip(opts["sizes"])) == "all" ? first.(CONFIGS) : splitcsv(opts["sizes"])
    configs = filter(c -> c[1] in names, CONFIGS)
    isempty(configs) && error("no configs matched --sizes=$(opts["sizes"])")

    return dtypes, impls, causals, configs, opts["seconds"]
end

function main(argv)
    opts = parse_cli(argv)
    if opts["list"]
        println("configs: ", join(first.(CONFIGS), ", "))
        println("dtypes:  f16, bf16, f32")
        println("impls:   ", join(IMPL_ORDER, ", "))
        return
    end
    @assert CUDA.functional() "CUDA is not functional"
    dtypes, impls, causals, configs, seconds = resolve(opts)

    println("GPU:        ", CUDA.name(CUDA.device()))
    println("NNlib:      ", pkgversion(NNlib))
    println("LuxLib:     ", pkgversion(LuxLib))
    println("CUDA.jl:    ", pkgversion(CUDA))
    println("cuDNN:      ", pkgversion(cuDNN), " (library ", cuDNN.version(), ")")
    println("selection:  dtypes=", join(string.(dtypes), ","),
            "  impls=", join(impls, ","),
            "  causal=", join(string.(causals), ","),
            "  sizes=", join(first.(configs), ","))

    for T in dtypes; sanity(T, impls); end
    for T in dtypes, causal in causals
        run(T, causal, impls, configs, seconds)
    end
    if length(impls) > 1
        ref = first(impls)
        println("\n<impl>/$ref = time relative to $ref (>1 → $ref faster).")
    end
end

main(ARGS)
