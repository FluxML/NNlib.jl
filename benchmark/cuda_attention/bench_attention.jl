# GPU multi-head attention benchmark: NNlib vs LuxLib vs NNkernels (Flash Attention).
#
# Three implementations of the SAME scaled dot-product (multi-head) attention
# (scale 1/√E):
#
#   * NNlib  `dot_product_attention`         — materialized path: forms the full
#       (kv_len, q_len, nheads, batch) score tensor, `softmax`, second batched_mul.
#       Memory is O(seq²).
#   * LuxLib `scaled_dot_product_attention`  — also a materialized softmax path,
#       with its own (E, H, L, B) layout and KV-grouping support.
#   * NNkernels `flash_attention`            — fused FlashAttention kernel; never
#       materializes the scores, so memory is O(seq).
#
# Each library wants a different layout for the SAME data, so we build all three
# from one source tensor:
#   * NNkernels: q,k,v = (head_dim, seq, nheads, batch)         = (E, L, H, B)
#   * NNlib:     q,k,v = (head_dim*nheads, seq, batch), nheads=H  = (E*H, L, B)
#   * LuxLib:    q,k,v = (head_dim, nheads, seq, batch)          = (E, H, L, B)
#
# Reports forward and full Zygote-gradient (forward+backward) wall time in ms,
# GPU-synchronized. "OOM" = ran out of GPU memory; "n/s" = that path can't run
# the config (echoed once to stderr).
#
# Run (whole suite):
#     julia --project=. bench_attention.jl
# Benchmark a single dtype/impl combination without rerunning everything:
#     julia --project=. bench_attention.jl --dtypes f32 --impls flash --causal false
#     julia --project=. bench_attention.jl --dtypes f16,bf16 --impls nnlib,lux --sizes tiny,small
# Pin a GPU:
#     CUDA_VISIBLE_DEVICES=GPU-<uuid> julia --project=. bench_attention.jl ...
# See all options:
#     julia --project=. bench_attention.jl --help

using CUDA, cuDNN, NNlib, NNkernels, LuxLib, Zygote, BenchmarkTools, Printf, ArgParse
using BFloat16s: BFloat16
using NNlib: make_causal_mask
using LuxLib: scaled_dot_product_attention

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

# An implementation: how to build its inputs from the source (E,L,H,B) tensors,
# and how to run forward / full-gradient on those inputs.
struct Impl
    name::String
    prep::Function   # (qf,kf,vf,causal) -> inputs tuple in this impl's layout
    fwd::Function    # (inputs)          -> output
    grad::Function   # (inputs)          -> gradient w.r.t. q,k,v
end

# (E,L,H,B) -> (E*H, L, B)
_tonn(x) = reshape(permutedims(x, (1, 3, 2, 4)), size(x, 1) * size(x, 3), size(x, 2), size(x, 4))
# (E,L,H,B) -> (E,H,L,B)
_tolux(x) = permutedims(x, (1, 3, 2, 4))

const IMPLS = Dict(
    "nnlib" => Impl("nnlib",
        (qf, kf, vf, causal) -> begin
            qn, kn, vn = _tonn(qf), _tonn(kf), _tonn(vf)
            mask = causal ? make_causal_mask(qn) : nothing
            (qn, kn, vn, mask, size(qf, 3))
        end,
        I -> dot_product_attention(I[1], I[2], I[3]; nheads=I[5], mask=I[4])[1],
        I -> Zygote.gradient((q, k, v) ->
                sum(dot_product_attention(q, k, v; nheads=I[5], mask=I[4])[1]), I[1], I[2], I[3])),
    "lux" => Impl("lux",
        (qf, kf, vf, causal) -> (_tolux(qf), _tolux(kf), _tolux(vf), causal),
        I -> scaled_dot_product_attention(I[1], I[2], I[3]; is_causal=I[4])[1],
        I -> Zygote.gradient((q, k, v) ->
                sum(scaled_dot_product_attention(q, k, v; is_causal=I[4])[1]), I[1], I[2], I[3])),
    "flash" => Impl("flash",
        (qf, kf, vf, causal) -> (qf, kf, vf, causal),
        I -> NNkernels.flash_attention(I[1], I[2], I[3]; causal=I[4]),
        I -> Zygote.gradient((q, k, v) ->
                sum(NNkernels.flash_attention(q, k, v; causal=I[4])), I[1], I[2], I[3])),
)
const IMPL_ORDER = ["nnlib", "lux", "flash"]   # canonical column order

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

# ---------------------------------------------------------------------------
# sanity check (only over the selected implementations)
# ---------------------------------------------------------------------------

# bring each impl's forward output to a common (E,H,L,B) layout for comparison
to_common(::Val{:nnlib}, y, E, L, H, B) = reshape(y, E, H, L, B)
to_common(::Val{:lux},   y, E, L, H, B) = y
to_common(::Val{:flash}, y, E, L, H, B) = permutedims(y, (1, 3, 2, 4))

function sanity(T, impls)
    E, L, H, B = 32, 64, 4, 2
    qf = CUDA.randn(T, E, L, H, B); kf = CUDA.randn(T, E, L, H, B); vf = CUDA.randn(T, E, L, H, B)
    outs = Dict{String,Any}()
    for name in impls
        im = IMPLS[name]
        outs[name] = try
            Array(to_common(Val(Symbol(name)), im.fwd(im.prep(qf, kf, vf, false)), E, L, H, B))
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
    haveflash = "flash" in impls && length(impls) > 1
    others = filter(!=("flash"), impls)
    header = rpad("config", 13) * lpad("L", 6) * lpad("H", 4) * " |"
    for name in impls; header *= lpad(name, 10); end
    haveflash && (header *= " |"; for name in others; header *= lpad(name * "/fl", 9); end)
    println("\n-- $title --")
    println(header)
    for r in rows
        line = rpad(r.name, 13) * lpad(string(r.L), 6) * lpad(string(r.H), 4) * " |"
        for name in impls; line *= lpad(fmt(r.t[name][key]), 10); end
        if haveflash
            line *= " |"
            for name in others; line *= lpad(spd(r.t[name][key], r.t["flash"][key]), 9); end
        end
        println(line)
    end
end

function run(T, causal, impls, configs, seconds)
    @printf("\n%s\n### dtype=%s  causal=%s\n%s\n", "#"^104, string(T), causal, "#"^104)
    rows = map(configs) do (name, E, L, H, B)
        qf = CUDA.randn(T, E, L, H, B); kf = CUDA.randn(T, E, L, H, B); vf = CUDA.randn(T, E, L, H, B)
        t = Dict{String,Any}()
        for nm in impls
            im = IMPLS[nm]
            I = try im.prep(qf, kf, vf, causal) catch; nothing end
            if I === nothing
                t[nm] = (; fwd = -1.0, grad = -1.0)
            else
                t[nm] = (; fwd  = timed(() -> im.fwd(I), seconds),
                          grad = timed(() -> im.grad(I), seconds))
            end
        end
        qf = kf = vf = nothing; CUDA.reclaim()
        (; name, L, H, t)
    end
    print_table("FORWARD (ms)", impls, rows, :fwd)
    print_table("FULL GRADIENT, fwd+bwd (ms)", impls, rows, :grad)
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function parse_cli(argv)
    s = ArgParseSettings(description="GPU multi-head attention benchmark: NNlib vs LuxLib vs NNkernels.")
    @add_arg_table! s begin
        "--dtypes"
            help = "comma list of dtypes: f16,bf16,f32 (aliases ok). Default: all."
            default = "f16,bf16,f32"
        "--impls"
            help = "comma list of implementations: nnlib,lux,flash. Default: all."
            default = "nnlib,lux,flash"
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
    for im in impls; im in IMPL_ORDER || error("unknown impl '$im' (try nnlib,lux,flash)"); end
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
    println("NNkernels:  ", pkgversion(NNkernels))
    println("CUDA.jl:    ", pkgversion(CUDA))
    println("selection:  dtypes=", join(string.(dtypes), ","),
            "  impls=", join(impls, ","),
            "  causal=", join(string.(causals), ","),
            "  sizes=", join(first.(configs), ","))

    for T in dtypes; sanity(T, impls); end
    for T in dtypes, causal in causals
        run(T, causal, impls, configs, seconds)
    end
    if "flash" in impls && length(impls) > 1
        println("\n<impl>/fl = time relative to flash (>1 → flash faster).")
    end
end

main(ARGS)
