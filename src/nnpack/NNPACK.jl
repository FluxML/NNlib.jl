include("libnnpack_types.jl")
include("error.jl")
include("libnnpack.jl")
include("performance.jl")
include("interface.jl")

const depsjl_path = joinpath(dirname(@__FILE__), "..", "..", "deps", "deps.jl")
if !isfile(depsjl_path)
    error("NNPACK not installed properly, run Pkg.build(\"NNlib\"), restart Julia and try again")
end

const shared_threadpool_dict = Dict{UInt64, Base.RefValue}()

"""
    is_nnpack_available()

Checks if the current hardware is supported by NNPACK.
"""
function is_nnpack_available()
    check_deps() isa Nothing || return false
    status = nnp_initialize()
    if status == nnp_status_unsupported_hardware
        return false
    else
        return true
    end
end

"""
    allocate_threadpool()

Allocates several threadpool based on the upper limit on the number of threads for the machine.
Allows NNPACK to intelligently choose which threadpool to use for getting the best
performance.
"""
function allocate_threadpool()
    global NNPACK_CPU_THREADS = NNPACK_CPU_THREADS > 8 ? UInt64(8) : floor(log2(NNPACK_CPU_THREADS))
    for i in 1:Int(NNPACK_CPU_THREADS)
        threads = UInt64(2^i)
        push!(shared_threadpool_dict, threads => Ref(pthreadpool_create(threads)))
    end
end

@init begin
    check_deps()
    status = nnp_initialize()
    if status == nnp_status_unsupported_hardware
        @warn "Hardware is unsupported by NNPACK so falling back to default NNlib"
    end
    try
        global NNPACK_CPU_THREADS = parse(UInt64, ENV["NNPACK_CPU_THREADS"])
    catch
        # Sys.CPU_THREADS should be a better default if we are tuning the benchmark suite on
        # a particular machine. However, we fix the runtime threadpool here to have a max of
        # 4 threads so anything above will be ignored anyways
        global NNPACK_CPU_THREADS = UInt64(4)
    end
    allocate_threadpool()
end
