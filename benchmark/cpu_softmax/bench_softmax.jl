# CPU softmax benchmark: the implementation NNlib has for CPU arrays, on Julia tasks,
# against the same loops written as KernelAbstractions kernels and against the generic
# implementation.
#
#   * broadcast   `NNlib._softmax!`, the generic implementation: three broadcasts on one
#       thread. It is what `softmax!` ran on the CPU up to NNlib 0.9.47.
#   * serial      `NNlib.softmax!` kept on one task.
#   * spawn       `NNlib.softmax!`: loops that the compiler vectorizes, on Julia tasks.
#   * ka          `softmax_ka!` of `ka_softmax.jl`: a kernel on the CPU backend, with a
#       work-item for each chunk of slices and a workgroup size of 1.
#   * ka-slice    `softmax_ka_slice!`: a work-item for each slice, the workgroup size left
#       to the backend. Contiguous slices only.
#   * ka-slice-1  the same with a workgroup size of 1.
#   * ka-unsafe   `ka-slice` declared with `unsafe_indices=true`. Only when every workgroup
#       is full, which here means a number of slices that is a power of two.
#   * ka-unsafe-1n  `ka-unsafe` with loops over `1:n` in every work-item, instead of over
#       the indices of its slice: the case of KernelAbstractions.jl#890.
#
# `bench_torch.py` is the PyTorch counterpart.
#
# Run (whole suite), with as many threads as the comparison is meant for:
#     julia --threads=auto --project=. bench_softmax.jl
# A single combination:
#     julia --threads=16 --project=. bench_softmax.jl --dtypes f64 --impls spawn,ka --sizes classifier
# See all options:
#     julia --project=. bench_softmax.jl --help

using NNlib, BenchmarkTools, Printf, ArgParse

include(joinpath(@__DIR__, "ka_softmax.jl"))

# (name, size, dims). `softmax` runs along `dims`.
const CONFIGS = [
    ("attention",     (256, 256, 256),    1),   # weights of 256 queries on 256 keys, 256 heads × batch
    ("long slices",   (1024, 1024, 16),   1),
    ("classifier",    (1000, 512),        1),   # 1000 classes, batch of 512
    ("few classes",   (10, 65536),        1),
    ("rows",          (2048, 512),        2),   # not contiguous: along the rows of a matrix
    ("segmentation",  (128, 128, 21, 16), 3),   # not contiguous: 21 classes for each pixel
    ("2^16 elements", (64, 1024),         1),   # below the size from which NNlib spawns tasks
    ("2^10 elements", (16, 64),           1),
]

const DTYPE_ALIASES = Dict("f32" => Float32, "float32" => Float32, "f64" => Float64, "float64" => Float64)
const DTYPE_ORDER = (Float32, Float64)

serial!(out, x, dims) = NNlib.@disallow_spawns softmax!(out, x; dims)

const IMPLS = Dict(
    "broadcast"  => (out, x, dims) -> NNlib._softmax!(out, x; dims),
    "serial"     => serial!,
    "spawn"      => (out, x, dims) -> softmax!(out, x; dims),
    "ka"         => (out, x, dims) -> softmax_ka!(out, x; dims),
    "ka-slice"   => (out, x, dims) -> softmax_ka_slice!(out, x; dims),
    "ka-slice-1" => (out, x, dims) -> softmax_ka_slice!(out, x; dims, workgroupsize=1),
    "ka-unsafe"  => (out, x, dims) -> softmax_ka_slice!(out, x; dims, unsafe_indices=true),
    "ka-unsafe-1n" => (out, x, dims) -> softmax_ka_slice!(out, x; dims, unsafe_indices=true, shared_bounds=true),
)
const IMPL_ORDER = ["broadcast", "serial", "spawn", "ka", "ka-slice", "ka-slice-1", "ka-unsafe", "ka-unsafe-1n"]

# Whether an implementation runs a configuration.
function applies(name, x, dims)
    startswith(name, "ka-") || return true
    pre, n, post = NNlib.softmax_shape(x, x, dims)
    return pre == 1 && (!startswith(name, "ka-unsafe") || ispow2(post))
end

# minimum time in ms of a thunk, or `nothing` if it does not run
function timed(f, seconds)
    try
        f()                                 # warmup / compile
        trial = BenchmarkTools.run(@benchmarkable($f()); seconds=seconds, evals=1)
        return minimum(trial).time / 1e6
    catch e
        @warn "unsupported path → n/s: $(first(split(sprint(showerror, e), '\n')))"
        return nothing
    end
end

# The time of the first call, compilation included, before anything else is compiled.
function first_calls(impls)
    x = randn(Float32, 256, 4096)
    parts = map(filter(in(impls), ["spawn", "ka"])) do name
        @sprintf("%s=%.2f s", name, @elapsed(IMPLS[name](similar(x), x, 1)))
    end
    isempty(parts) || println("first call:    ", join(parts, "  "))
end

# The largest difference from the generic implementation, out of place and in place, along
# contiguous slices and along rows.
function sanity(T, impls)
    diffs = Dict(name => 0.0 for name in impls)
    for (sz, dims) in (((96, 4096), 1), ((2048, 192), 2))
        x = randn(T, sz)
        x[3, 5] = -Inf
        ref = NNlib._softmax!(similar(x), x; dims)
        for name in impls
            applies(name, x, dims) || continue
            y = copy(x)
            diffs[name] = max(diffs[name], maximum(abs, IMPLS[name](similar(x), x, dims) .- ref),
                              maximum(abs, IMPLS[name](y, y, dims) .- ref))
        end
    end
    @printf("sanity %-8s (vs broadcast)  %s\n", string(T), join((@sprintf("%s=%.1e", name, diffs[name]) for name in impls), "  "))
end

fmt(x) = x === nothing ? "n/s" : @sprintf("%.4f", x)

function run(T, impls, configs, seconds, inplace)
    header = rpad("config", 15) * rpad("size", 17) * rpad("dims", 5) * "|"
    for name in impls; header *= lpad(name, 13); end
    ratios = [(a, b) for (a, b) in (("ka", "spawn"), ("spawn", "serial"), ("serial", "broadcast")) if a in impls && b in impls]
    isempty(ratios) || (header *= " |")
    for (a, b) in ratios; header *= lpad(a * "/" * b, 18); end
    println(header)
    for (name, sz, dims) in configs
        x = randn(T, sz)
        out = inplace ? x : similar(x)
        res = Dict(nm => (applies(nm, x, dims) ? timed(() -> IMPLS[nm](out, x, dims), seconds) : nothing) for nm in impls)
        line = rpad(name, 15) * rpad(join(sz, "x"), 17) * rpad(string(dims), 5) * "|"
        for nm in impls; line *= lpad(fmt(res[nm]), 13); end
        isempty(ratios) || (line *= " |")
        for (a, b) in ratios
            line *= lpad((res[a] === nothing || res[b] === nothing) ? "-" : @sprintf("%.2fx", res[a] / res[b]), 18)
        end
        println(line)
        flush(stdout)
    end
end

function parse_cli(argv)
    s = ArgParseSettings(description="CPU softmax benchmark: NNlib on Julia tasks vs KernelAbstractions kernels vs the generic implementation.")
    @add_arg_table! s begin
        "--dtypes"
            help = "comma list of dtypes: f32,f64. Default: f32."
            default = "f32"
        "--impls"
            help = "comma list of implementations: $(join(IMPL_ORDER, ',')). Default: all."
            default = join(IMPL_ORDER, ',')
        "--sizes"
            help = "comma list of config names (see --list). Default: all."
            default = "all"
        "--inplace"
            help = "comma list of false,true: whether the output is the input. Default: false."
            default = "false"
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
        println("dtypes:  f32, f64")
        println("impls:   ", join(IMPL_ORDER, ", "))
        return
    end
    dtypes = [DTYPE_ALIASES[lowercase(d)] for d in splitcsv(opts["dtypes"])]
    dtypes = filter(in(dtypes), collect(DTYPE_ORDER))
    impls = filter(in(map(lowercase, splitcsv(opts["impls"]))), IMPL_ORDER)
    names = lowercase(strip(opts["sizes"])) == "all" ? first.(CONFIGS) : splitcsv(opts["sizes"])
    configs = filter(c -> c[1] in names, CONFIGS)
    isempty(configs) && error("no configs matched --sizes=$(opts["sizes"])")
    inplaces = unique(c == "true" for c in splitcsv(lowercase(opts["inplace"])))

    println("CPU:           ", Sys.cpu_info()[1].model, " (", Sys.CPU_THREADS, " hardware threads)")
    println("Julia:         ", VERSION, ", ", Threads.nthreads(), " threads")
    println("NNlib:         ", pkgversion(NNlib))
    println("KernelAbstractions: ", pkgversion(KA))
    println("selection:     dtypes=", join(string.(dtypes), ","), "  impls=", join(impls, ","),
            "  sizes=", join(first.(configs), ","))
    first_calls(impls)

    for T in dtypes; sanity(T, impls); end
    for T in dtypes, inplace in inplaces
        @printf("\n%s\n### dtype=%s  %s  (ms)\n%s\n", "#"^100, string(T), inplace ? "in place" : "out of place", "#"^100)
        run(T, impls, configs, opts["seconds"], inplace)
    end
    println("\na/b = time of a relative to b (>1 → b faster).")
end

main(ARGS)
