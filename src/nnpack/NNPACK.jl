include("libnnpack_types.jl")
include("error.jl")
include("libnnpack.jl")

const depsjl_path = joinpath(dirname(@__FILE__), "..", "..", "deps", "deps.jl")
if !isfile(depsjl_path)
    error("NNPACK not installed properly, run Pkg.build(\"NNlib\"), restart Julia and try again")
end
include(depsjl_path)

const nnlib_interface_path = joinpath(dirname(@__FILE__), "nnlib.jl")
@init begin
    check_deps()
    status = nnp_initialize()
    if status == nnp_status_unsupported_hardware
        @warn "HARDWARE is unsupported by NNPACK so falling back to default NNlib"
    else
        include(nnlib_interface_path)
    end
    try
        global NNPACK_CPU_THREADS = parse(UInt64, ENV["JULIA_NUM_THREADS"])
    catch
        global NNPACK_CPU_THREADS = Sys.CPU_THREADS
        @warn "JULIA_NUM_THREADS not set. So taking the NNPACK default $(NNPACK_CPU_THREADS)"
    end
    global shared_threadpool = pthreadpool_create(NNPACK_CPU_THREADS)
end
