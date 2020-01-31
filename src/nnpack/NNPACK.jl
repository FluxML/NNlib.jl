module NNPACK
using ..NNlib
using ..NNlib: check_dims, input_size, output_size, kernel_size, padding, stride, flipkernel, flipweight
using NNPACK_jll

include("libnnpack_types.jl")
include("error.jl")
include("libnnpack.jl")
include("multithreading.jl")
include("interface.jl")

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

function __init__()
    if !is_nnpack_available()
        @warn "Hardware unsupported by NNPACK, falling back to other NNlib backends"
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

# Here we register our convolution and pooling methods with the parent NNlib module.
# We have implementations only for normal convolution and maxpooling:
import ..conv_backends, ..pooling_backends
push!(conv_backends[:conv], :nnpack)
push!(conv_backends[:∇conv_data], :nnpack)
push!(conv_backends[:∇conv_filter], :nnpack)

push!(pooling_backends[:maxpool], :nnpack)
end # module NNPACK

using .NNPACK
import .NNPACK: maxpool_nnpack!, nnpack_supported_operation,
               conv_nnpack!, ∇conv_data_nnpack!, ∇conv_filter_nnpack!