include("libnnpack_types.jl")
include("error.jl")
include("libnnpack.jl")

const depsjl_path = joinpath(dirname(@__FILE__), "..", "..", "deps", "deps.jl")
if !isfile(depsjl_path)
    error("NNPACK not installed properly, run Pkg.build(\"NNlib\"), restart Julia and try again")
end
include(depsjl_path)

@init begin
    check_deps()
    try
        nnp_initialize()
        include("nnlib.jl")
    catch
        @warn "HARDWARE is unsupported by NNPACK so falling back to default NNlib"
    end
    try
        global NNPACK_CPU_THREADS = parse(UInt64, ENV["JULIA_NUM_THREADS"])
    catch
        @warn "`JULIA_NUM_THREADS` not set. So taking the NNPACK default `4`"
        global NNPACK_CPU_THREADS = UInt64(4)
    end
end
