module NNPACK

using Libdl, Requires

include("libnnpack_types.jl")
include("error.jl")
include("libnnpack.jl")

const depsjl_path = joinpath(dirname(@__FILE__), "..", "..", "deps", "deps.jl")
if !isfile(depsjl_path)
    error("NNPACK not installed properly, run Pkg.build(\"NNPACK\"), restart Julia and try again")
end
include(depsjl_path)

function __init__()
    check_deps()
    nnp_initialize()
    global NNPACK_CPU_THREADS = parse(UInt64, ENV["JULIA_NUM_THREADS"])
    include(joinpath(dirname(@__FILE__), "nnlib.jl"))
end

end
