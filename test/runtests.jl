using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
const is_TRAVIS = haskey(ENV,"TRAVIS")

@time begin
    if GROUP == "All" || GROUP == "Core"
        @safetestset "Activation Functions" begin include("activation.jl") end
        @safetestset "Convolutions" begin include("conv.jl") end
        @safetestset "Pooling Functions" begin include("pooling.jl") end
        @safetestset "Inference" begin include("inference.jl") end
    end
end

if GROUP == "Downstream"
    if is_TRAVIS
      using Pkg
      Pkg.add("Flux")
    end
    Pkg.test("Flux")
end
