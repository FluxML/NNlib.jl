using ForwardDiff: Dual

CUDA.allowscalar(false)

const CUDAExt = @static if isdefined(Base, :get_extension)
    Base.get_extension(NNlib, :CUDAExt)
else
    NNlib.CUDAExt # Added by Requires.jl
end

@testset "CUDA" begin
    include("test_utils.jl")
    include("activations.jl")
    include("batchedadjtrans.jl")
    include("batchedmul.jl")
    include("upsample.jl")
    include("conv.jl")
    include("ctc.jl")
    include("fold.jl")
    include("pooling.jl")
    include("softmax.jl")
    include("batchnorm.jl")
    include("scatter.jl")
    include("gather.jl")
    include("sampling.jl")
end
