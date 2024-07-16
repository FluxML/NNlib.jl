module NNlibNNPACK_jllExt

using NNlib: NNlib
using NNPACK_jll, Pkg

if isdefined(NNPACK_jll, :libnnpack)
    include("NNPACK.jl")
else
    @warn "NNPACK not available for your platform: " *
          "$( Pkg.BinaryPlatforms.platform_name(Pkg.BinaryPlatforms.platform_key_abi()))" *
          "($( Pkg.BinaryPlatforms.triplet(Pkg.BinaryPlatforms.platform_key_abi())))
          You will be able to use only the default Julia NNlib backend"
end

end
