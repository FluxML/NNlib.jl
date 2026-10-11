# cuDNN operator benchmark: which implementation should NNlib dispatch to?
#
# For each operator that NNlib accelerates on CUDA, this times every implementation it
# could dispatch to, on the same inputs:
#
#   * nnlib    what NNlib runs today for a `CuArray`: one of the candidates below, plus
#              NNlib's own dispatch overhead.
#   * generic  NNlib's device-agnostic implementation, built from broadcasts and
#              reductions that CUDA.jl compiles to GPU kernels. No cuDNN involved.
#   * fixed    cuDNN's fixed-function API (`cudnnPoolingForward` and the like), which
#              cuDNN.jl keeps under `legacy`.
#   * graph    cuDNN's graph (backend) API, through cuDNN.jl's operation wrappers.
#
# Times are the minimum over GPU-synchronized samples, in μs. Every candidate allocates
# its outputs, as NNlib's functions do.
#   "n/s"  the implementation cannot run that configuration (the reason is echoed once
#          to stderr), e.g. cuDNN has no engine for it;
#   "OOM"  it ran out of GPU memory;
#   "-"    there is no such implementation.
#
# Run (whole suite):
#     julia --project=. bench_cudnn.jl
# A subset, with the timings also written as tab-separated values:
#     julia --project=. bench_cudnn.jl --ops batchnorm,pooling --dtypes f16 --out bn.tsv
# All options, and the available operators/sizes:
#     julia --project=. bench_cudnn.jl --help
#     julia --project=. bench_cudnn.jl --list
#
# To add an operator, write a `<family>_ops(T, sz)` function returning its `Op`s and
# register it in `FAMILIES`. The candidates call cuDNN.jl through qualified names
# inside functions, so that one whose API has gone shows up as "n/s" rather than
# breaking the script.

using CUDA, cuDNN, NNlib, BenchmarkTools, Printf, ArgParse
using BFloat16s: BFloat16
using NNlib: batchnorm, ∇batchnorm

# ---------------------------------------------------------------------------
# registries
# ---------------------------------------------------------------------------

const IMPLS = ("nnlib", "generic", "fixed", "graph")
const CANDIDATES = IMPLS[2:end]

# An operator and its implementations: thunks returning the output array, or
# `nothing` where there is no such implementation.
struct Op
    name::String
    impls::NamedTuple{(:nnlib, :generic, :fixed, :graph)}
end
Op(name; nnlib, generic=nothing, fixed=nothing, graph=nothing) =
    Op(name, (; nnlib, generic, fixed, graph))

# Inputs are (width, height, channels, batch).
const SIZES = [
    ("small",  (8, 8, 16, 4)),
    ("medium", (32, 32, 64, 16)),
    ("large",  (64, 64, 128, 32)),
    ("odd-C",  (32, 32, 30, 16)),   # a number of channels that is not a multiple of 4
]

const DTYPE_ALIASES = Dict(
    "f16" => Float16,  "float16" => Float16,  "fp16" => Float16,  "half" => Float16,
    "bf16" => BFloat16, "bfloat16" => BFloat16,
    "f32" => Float32,  "float32" => Float32,  "fp32" => Float32,
    "f64" => Float64,  "float64" => Float64,  "fp64" => Float64,
)
const DTYPE_ORDER = (Float16, BFloat16, Float32, Float64)   # canonical print order

# cuDNN computes half-precision data in Float32, and wants Float32 parameters for it.
wide(::Type{T}) where {T} = T in (Float16, BFloat16) ? Float32 : T

# `CUDA.randn` does not cover every eltype: draw in Float32 and convert on the device.
gpurandn(::Type{T}, sz...) where {T} = T.(CUDA.randn(Float32, sz...))

fixed_scale(::Type{T}, v) where {T} = cuDNN.scalingParameter(T, v)

# ---------------------------------------------------------------------------
# batch normalization
# ---------------------------------------------------------------------------

bn_dims(x) = ntuple(i -> i == ndims(x) - 1 ? size(x, i) : 1, ndims(x))

bn_param_desc(x, ::Type{P}) where {P} =
    cuDNN.cudnnTensorDescriptor(cuDNN.CUDNN_TENSOR_NCHW, cuDNN.cudnnDataType(P),
                                Cint(ndims(x)),
                                cuDNN.dim4(bn_dims(x), Val(cuDNN.CUDNN_TENSOR_NCHW)))

# `saved` are the buffers for the batch statistics the backward pass can reuse.
function bn_train_fixed(g, b, x, rm, rv, momentum, eps; saved=(CU_NULL, CU_NULL))
    T = eltype(x)
    y = similar(x)
    xd = cuDNN.cudnnTensorDescriptor(x)
    cuDNN.cudnnBatchNormalizationForwardTraining(
        cuDNN.handle(), cuDNN.CUDNN_BATCHNORM_SPATIAL, fixed_scale(T, 1), fixed_scale(T, 0),
        xd, x, xd, y, bn_param_desc(x, eltype(g)), g, b, momentum, rm, rv, eps, saved...)
    return y
end

function bn_test_fixed(g, b, x, rm, rv, eps)
    T = eltype(x)
    y = similar(x)
    xd = cuDNN.cudnnTensorDescriptor(x)
    cuDNN.cudnnBatchNormalizationForwardInference(
        cuDNN.handle(), cuDNN.CUDNN_BATCHNORM_SPATIAL, fixed_scale(T, 1), fixed_scale(T, 0),
        xd, x, xd, y, bn_param_desc(x, eltype(g)), g, b, rm, rv, eps)
    return y
end

function bn_grad_fixed(g, x, dy, eps, saved)
    T = eltype(x)
    dg, db, dx = similar(g), similar(g), similar(x)
    xd = cuDNN.cudnnTensorDescriptor(x)
    cuDNN.cudnnBatchNormalizationBackward(
        cuDNN.handle(), cuDNN.CUDNN_BATCHNORM_SPATIAL,
        fixed_scale(T, 1), fixed_scale(T, 0), fixed_scale(T, 1), fixed_scale(T, 0),
        xd, x, xd, dy, xd, dx, bn_param_desc(x, eltype(g)), g, dg, db, eps, saved...)
    return dx
end

function batchnorm_ops(T, sz)
    x, dy = gpurandn(T, sz...), gpurandn(T, sz...)
    C, P = sz[end-1], wide(T)
    g = CUDA.rand(P, C) .+ P(0.5)
    b = CUDA.randn(P, C)
    rm, rv = CUDA.zeros(P, C), CUDA.ones(P, C)
    momentum, eps = 0.1f0, 1f-5

    # NNlib's generic methods, which the cuDNN ones shadow for a `CuArray`
    generic(; training) = invoke(batchnorm, Tuple{Any,Any,AbstractArray,Any,Any,Any},
                                 g, b, x, rm, rv, momentum; eps, training)
    generic_grad() = invoke(∇batchnorm, Tuple{Any,Any,AbstractArray,Any,Any,Any,Any},
                            g, b, x, dy, rm, rv, momentum; eps, training=true)[3]

    function graph_train()
        y = similar(x)
        cuDNN.batchnorm_training!(y, x, g, b; running_mean=rm, running_var=rv, momentum,
                                  epsilon=eps)
        return y
    end
    graph_test() = cuDNN.batchnorm_inference!(similar(x), x, g, b, rm, rv; epsilon=eps)

    # The cuDNN gradients are given the batch statistics saved by their own forward
    # pass, which is not timed. `generic` and `nnlib` (called without a cache, as from
    # its `rrule`) recompute them.
    fixed_saved = try
        saved = (similar(g, bn_dims(x)), similar(g, bn_dims(x)))
        bn_train_fixed(g, b, x, rm, rv, momentum, eps; saved)
        saved
    catch e
        e
    end
    graph_saved = try
        cuDNN.batchnorm_training!(similar(x), x, g, b; momentum, epsilon=eps)
    catch e
        e
    end
    function fixed_grad()
        fixed_saved isa Exception && throw(fixed_saved)
        bn_grad_fixed(g, x, dy, eps, fixed_saved)
    end
    function graph_grad()
        graph_saved isa Exception && throw(graph_saved)
        dx, dg, db = similar(x), similar(g), similar(g)
        cuDNN.batchnorm_gradient!(dx, dg, db, dy, x, g, graph_saved...; epsilon=eps)
        return dx
    end

    return [
        Op("batchnorm (training)";
           nnlib   = () -> batchnorm(g, b, x, rm, rv, momentum; eps, training=true),
           generic = () -> generic(training=true),
           fixed   = () -> bn_train_fixed(g, b, x, rm, rv, momentum, eps),
           graph   = graph_train),
        Op("batchnorm (inference)";
           nnlib   = () -> batchnorm(g, b, x, rm, rv, momentum; eps, training=false),
           generic = () -> generic(training=false),
           fixed   = () -> bn_test_fixed(g, b, x, rm, rv, eps),
           graph   = graph_test),
        Op("∇batchnorm (training)";
           nnlib   = () -> ∇batchnorm(g, b, x, dy, rm, rv, momentum; eps, training=true)[3],
           generic = generic_grad,
           fixed   = fixed_grad,
           graph   = graph_grad),
    ]
end

# ---------------------------------------------------------------------------
# pooling (window 2, stride 2)
# ---------------------------------------------------------------------------

function pooling_ops(T, sz)
    x = gpurandn(T, sz...)
    nd = ndims(x) - 2
    pdims = PoolDims(x, 2)
    kw = (window=ntuple(_ -> 2, nd), stride=ntuple(_ -> 2, nd), padding=0)
    ymax, ymean = maxpool(x, pdims), meanpool(x, pdims)
    ysz = size(ymax)
    dy = gpurandn(T, ysz...)

    fixed_desc(mode) = cuDNN.cudnnPoolingDescriptor(
        getfield(cuDNN, mode), cuDNN.CUDNN_NOT_PROPAGATE_NAN, Cint(nd),
        cuDNN.pooldims(kw.window, size(x)), cuDNN.pooldims(ntuple(_ -> 0, nd), size(x)),
        cuDNN.pooldims(kw.stride, size(x)))
    fixed_fwd(mode) = cuDNN.cudnnPoolingForward!(similar(x, ysz), x, fixed_desc(mode))
    function fixed_bwd(mode, y)
        dx = similar(x)
        xd, yd = cuDNN.cudnnTensorDescriptor(x), cuDNN.cudnnTensorDescriptor(y)
        cuDNN.cudnnPoolingBackward(cuDNN.handle(), fixed_desc(mode), fixed_scale(T, 1),
                                   yd, y, yd, dy, xd, x, fixed_scale(T, 0), xd, dx)
        return dx
    end
    maxmode, meanmode = :CUDNN_POOLING_MAX, :CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING

    return [
        Op("maxpool";
           nnlib = () -> maxpool(x, pdims),
           fixed = () -> fixed_fwd(maxmode),
           graph = () -> cuDNN.maxpool!(similar(x, ysz), x; kw...)),
        Op("∇maxpool";
           nnlib = () -> ∇maxpool(dy, ymax, x, pdims),
           fixed = () -> fixed_bwd(maxmode, ymax),
           graph = () -> cuDNN.∇maxpool!(similar(x), dy, ymax, x; kw...)),
        Op("meanpool";
           nnlib = () -> meanpool(x, pdims),
           fixed = () -> fixed_fwd(meanmode),
           graph = () -> cuDNN.meanpool!(similar(x, ysz), x; kw...)),
        Op("∇meanpool";
           nnlib = () -> ∇meanpool(dy, ymean, x, pdims),
           fixed = () -> fixed_bwd(meanmode, ymean),
           graph = () -> cuDNN.∇meanpool!(similar(x), dy, ymean, x; kw...)),
    ]
end

# ---------------------------------------------------------------------------
# activations
# ---------------------------------------------------------------------------

# cuDNN.jl has no operation wrapper for a single pointwise graph: build and cache one
# the way its wrappers do.
function pointwise_graph(mode::Symbol, x)
    y = similar(x)
    T = eltype(x)
    key = (:nnlib_bench_pointwise, mode, T, size(x), strides(x),
           cuDNN.pointer_alignment(x), cuDNN.pointer_alignment(y))
    g = cuDNN.cached_graph(key) do
        g = cuDNN.Graph(io_dtype=T, intermediate_dtype=Float32, compute_dtype=wide(T))
        tx = cuDNN.tensor!(g, x; name="X")
        ty = cuDNN.tensor!(g, y; name="Y", output=true)
        cuDNN.pointwise!(g, mode, tx; y=ty)
        cuDNN.build!(g)
    end
    cuDNN.execute!(g, cuDNN.tensor(g, "X") => x, cuDNN.tensor(g, "Y") => y)
    return y
end

function activation_ops(T, sz)
    x = gpurandn(T, sz...)
    map((("relu", NNlib.relu, :CUDNN_ACTIVATION_RELU, :relu),
         ("tanh", tanh, :CUDNN_ACTIVATION_TANH, :tanh),
         ("σ", NNlib.σ, :CUDNN_ACTIVATION_SIGMOID, :sigmoid),
         ("elu", NNlib.elu, :CUDNN_ACTIVATION_ELU, :elu))) do (name, f, fixedmode, graphmode)
        Op("$name.(x)";
           nnlib   = () -> f.(x),
           # a closure, which NNlib's cuDNN broadcast overrides do not match
           generic = () -> broadcast(z -> f(z), x),
           fixed   = () -> cuDNN.cudnnActivationForward!(similar(x), x;
                                                         mode=getfield(cuDNN, fixedmode)),
           graph   = () -> pointwise_graph(graphmode, x))
    end
end

# ---------------------------------------------------------------------------
# softmax, along the first dimension and along the channels
# ---------------------------------------------------------------------------

function softmax_ops(T, sz)
    x, dy = gpurandn(T, sz...), gpurandn(T, sz...)
    algo() = cuDNN.CUDNN_SOFTMAX_ACCURATE
    # cuDNN reduces over all of (W, H, C) per sample ("instance" mode), or over the
    # channels alone: for the first dimension, fold the others into the batch.
    instance_size = (1, 1, sz[1], length(x) ÷ sz[1])
    fixed_view(a, dims) = dims == 1 ? reshape(a, instance_size) : a
    fixed_mode(dims) = dims == 1 ? cuDNN.CUDNN_SOFTMAX_MODE_INSTANCE :
                                   cuDNN.CUDNN_SOFTMAX_MODE_CHANNEL
    function fixed_fwd(dims)
        y = similar(x)
        cuDNN.cudnnSoftmaxForward!(fixed_view(y, dims), fixed_view(x, dims);
                                   mode=fixed_mode(dims), algo=algo())
        return y
    end
    function fixed_bwd(y, dims)
        dx = similar(x)
        yv = fixed_view(y, dims)
        d = cuDNN.cudnnTensorDescriptor(yv)
        cuDNN.cudnnSoftmaxBackward(cuDNN.handle(), algo(), fixed_mode(dims),
                                   fixed_scale(T, 1), d, yv, d, fixed_view(dy, dims),
                                   fixed_scale(T, 0), d, fixed_view(dx, dims))
        return dx
    end

    ops = Op[]
    for dims in (1, 3)
        y = softmax(x; dims)
        push!(ops, Op("softmax (dims=$dims)";
                      nnlib   = () -> softmax(x; dims),
                      generic = () -> NNlib._softmax!(similar(x), x; dims),
                      fixed   = () -> fixed_fwd(dims)))
        push!(ops, Op("∇softmax (dims=$dims)";
                      nnlib   = () -> ∇softmax(dy, y; dims),
                      generic = () -> NNlib._∇softmax!(similar(y), dy, y; dims),
                      fixed   = () -> fixed_bwd(y, dims)))
    end
    return ops
end

# ---------------------------------------------------------------------------
# convolution (3x3 kernel, padding 1, as many output as input channels)
# ---------------------------------------------------------------------------

function conv_ops(T, sz)
    x = gpurandn(T, sz...)
    nd, C = ndims(x) - 2, sz[end-1]
    w = gpurandn(T, ntuple(_ -> 3, nd)..., C, C) ./ T(10)
    cdims = DenseConvDims(x, w; padding=1)
    y = conv(x, w, cdims)
    dy = gpurandn(T, size(y)...)
    kw = (padding=1, mode=:convolution)

    fixed_desc() = cuDNN.cudnnConvolutionDescriptor(
        cuDNN.convdims(ntuple(_ -> 1, nd), size(x), 0),
        cuDNN.convdims(ntuple(_ -> 1, nd), size(x), 1),
        cuDNN.convdims(ntuple(_ -> 1, nd), size(x), 1),
        cuDNN.CUDNN_CONVOLUTION, cuDNN.cudnnDataType(wide(T)), cuDNN.math_mode(),
        cuDNN.CUDNN_DEFAULT_REORDER, Cint(1))
    function fixed_data()
        dx = similar(x)
        d = fixed_desc()
        xd, yd = cuDNN.cudnnTensorDescriptor(dx), cuDNN.cudnnTensorDescriptor(dy)
        wd = cuDNN.cudnnFilterDescriptor(w)
        p = cuDNN.cudnnConvolutionBwdDataAlgoPerf(wd, w, yd, dy, d, xd, dx, false)
        cuDNN.with_workspace(p.memory) do workspace
            cuDNN.cudnnConvolutionBackwardData(
                cuDNN.handle(), fixed_scale(T, 1), wd, w, yd, dy, d, p.algo, workspace,
                sizeof(workspace), fixed_scale(T, 0), xd, dx)
        end
        return dx
    end
    function fixed_filter()
        dw = similar(w)
        d = fixed_desc()
        xd, yd = cuDNN.cudnnTensorDescriptor(x), cuDNN.cudnnTensorDescriptor(dy)
        wd = cuDNN.cudnnFilterDescriptor(dw)
        p = cuDNN.cudnnConvolutionBwdFilterAlgoPerf(xd, x, yd, dy, d, wd, dw, false)
        cuDNN.with_workspace(p.memory) do workspace
            cuDNN.cudnnConvolutionBackwardFilter(
                cuDNN.handle(), fixed_scale(T, 1), xd, x, yd, dy, d, p.algo, workspace,
                sizeof(workspace), fixed_scale(T, 0), wd, dw)
        end
        return dw
    end

    return [
        Op("conv";
           nnlib = () -> conv(x, w, cdims),
           fixed = () -> cuDNN.cudnnConvolutionForward!(similar(y), w, x, fixed_desc()),
           graph = () -> cuDNN.convolution!(similar(y), x, w; kw...)),
        Op("∇conv_data";
           nnlib = () -> ∇conv_data(dy, w, cdims),
           fixed = fixed_data,
           graph = () -> cuDNN.convolution_data_gradient!(similar(x), dy, w; kw...)),
        Op("∇conv_filter";
           nnlib = () -> ∇conv_filter(x, dy, cdims),
           fixed = fixed_filter,
           graph = () -> cuDNN.convolution_filter_gradient!(similar(w), x, dy; kw...)),
    ]
end

# family => (eltype, size) -> its operators
const FAMILIES = [
    "batchnorm"   => batchnorm_ops,
    "pooling"     => pooling_ops,
    "activations" => activation_ops,
    "softmax"     => softmax_ops,
    "conv"        => conv_ops,
]

# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

# time a thunk in μs (minimum, GPU-synced). Returns a positive time on success, NaN
# for an out-of-memory failure ("OOM"), or -1.0 if that path can't run this config
# ("n/s"). The first occurrence of each distinct reason is echoed to stderr.
const SEEN_FAILURES = Set{String}()
function timed(f, seconds)
    try
        CUDA.@sync f()                      # warmup / compile / plan selection
        CUDA.@sync f()
        b = @benchmarkable CUDA.@sync($f())
        trial = BenchmarkTools.run(b; seconds=seconds, evals=1)
        return minimum(trial).time / 1e3    # ns -> μs
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

ok(t) = t !== nothing && !isnan(t) && t > 0
fmt(t) = t === nothing ? "-" : isnan(t) ? "OOM" : t < 0 ? "n/s" : @sprintf("%.1f", t)
sizestr(sz) = join(sz, "x")

# ---------------------------------------------------------------------------
# sanity check: the implementations of each operator agree
# ---------------------------------------------------------------------------

function sanity(families, T)
    sz = (8, 8, 16, 4)
    println("sanity on ", T, " ", sizestr(sz), ": largest deviation from `nnlib`")
    for (_, makeops) in families, op in makeops(T, sz)
        ref = Float32.(Array(op.impls.nnlib()))
        parts = map(CANDIDATES) do name
            f = op.impls[Symbol(name)]
            f === nothing && return "$name=-"
            d = try
                @sprintf("%.1e", maximum(abs, ref .- Float32.(Array(f()))))
            catch
                "n/s"
            end
            "$name=$d"
        end
        @printf("  %-24s %s\n", op.name, join(parts, "  "))
    end
    flush(stdout)
end

# ---------------------------------------------------------------------------
# run + tabulate one family
# ---------------------------------------------------------------------------

tableline(op, T, sz, cells, fastest) =
    rpad(op, 25) * rpad(T, 10) * rpad(sz, 15) * "|" * join(lpad.(cells, 10)) * " | " * fastest

function run_family(family, makeops, dtypes, sizes, seconds, rows)
    println("\n", "#"^100, "\n### ", family, "   (μs)\n", "#"^100)
    println(tableline("operator", "eltype", "size", IMPLS, "fastest"))
    for T in dtypes, (_, sz) in sizes
        ops = try
            makeops(T, sz)
        catch e
            @warn "cannot set up $family for $T $(sizestr(sz))" exception=e
            continue
        end
        for op in ops
            t = map(f -> f === nothing ? nothing : timed(f, seconds), op.impls)
            candidates = [(t[Symbol(name)], name) for name in CANDIDATES if ok(t[Symbol(name)])]
            fastest = isempty(candidates) ? "-" : last(minimum(candidates))
            println(tableline(op.name, string(T), sizestr(sz), map(fmt, values(t)), fastest))
            for name in IMPLS
                push!(rows, (family, op.name, string(T), sizestr(sz), name, fmt(t[Symbol(name)])))
            end
        end
        flush(stdout)   # show each block as it completes, also when writing to a file
        CUDA.reclaim()
    end
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function parse_cli(argv)
    s = ArgParseSettings(description="cuDNN operator benchmark: NNlib's dispatch vs the " *
                                     "generic, fixed-function and graph implementations.")
    @add_arg_table! s begin
        "--ops"
            help = "comma list of operator families: $(join(first.(FAMILIES), ',')). Default: all."
            default = "all"
        "--dtypes"
            help = "comma list of dtypes: f16,bf16,f32,f64 (aliases ok)."
            default = "f32,f16"
        "--sizes"
            help = "comma list of input sizes: $(join(first.(SIZES), ',')). Default: all."
            default = "all"
        "--seconds"
            help = "per-measurement time budget."
            arg_type = Float64
            default = 0.5
        "--out"
            help = "also write the timings to this file, as tab-separated values."
            default = ""
        "--no-check"
            help = "skip the check that the implementations agree."
            action = :store_true
        "--list"
            help = "list the available operators, sizes and dtypes, and exit."
            action = :store_true
    end
    return parse_args(argv, s)
end

splitcsv(s) = strip.(split(s, ','; keepempty=false))

function select(option, available, what)
    lowercase(strip(option)) == "all" && return available
    names = splitcsv(option)
    for name in names
        name in first.(available) ||
            error("unknown $what '$name' (try $(join(first.(available), ',')))")
    end
    return filter(a -> first(a) in names, available)
end

function main(argv)
    opts = parse_cli(argv)
    if opts["list"]
        println("operator families: ", join(first.(FAMILIES), ", "))
        println("sizes:             ", join(("$n=$(sizestr(sz))" for (n, sz) in SIZES), ", "))
        println("dtypes:            f16, bf16, f32, f64")
        return
    end
    @assert CUDA.functional() "CUDA is not functional"
    families = select(opts["ops"], FAMILIES, "operator family")
    sizes = select(opts["sizes"], SIZES, "size")
    dtypes = map(splitcsv(opts["dtypes"])) do d
        haskey(DTYPE_ALIASES, lowercase(d)) || error("unknown dtype '$d' (try f16,bf16,f32,f64)")
        DTYPE_ALIASES[lowercase(d)]
    end
    dtypes = filter(in(dtypes), collect(DTYPE_ORDER))   # canonical order, dedup

    println("GPU:       ", CUDA.name(CUDA.device()))
    println("Julia:     ", VERSION)
    println("NNlib:     ", pkgversion(NNlib))
    println("CUDA.jl:   ", pkgversion(CUDA))
    println("cuDNN.jl:  ", pkgversion(cuDNN), " (library ", cuDNN.version(), ")")
    println("selection: ops=", join(first.(families), ","),
            "  dtypes=", join(string.(dtypes), ","),
            "  sizes=", join(first.(sizes), ","))

    opts["no-check"] || sanity(families, first(dtypes))

    rows = NTuple{6,String}[]
    for (family, makeops) in families
        run_family(family, makeops, dtypes, sizes, opts["seconds"], rows)
    end
    println("\nfastest = the fastest of generic, fixed and graph; `nnlib` is what NNlib runs.")

    if !isempty(opts["out"])
        open(opts["out"], "w") do io
            println(io, join(("family", "operator", "eltype", "size", "impl", "time_us"), '\t'))
            for row in rows
                println(io, join(row, '\t'))
            end
        end
        println("timings written to ", opts["out"])
    end
end

main(ARGS)
