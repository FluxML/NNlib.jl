path = joinpath(dirname(@__FILE__), "..", "src")

if is_apple()
  download("https://externalshare.blob.core.windows.net/packages/nnlib-0.1.0.dylib",
           joinpath(@__DIR__, "nnlib.dylib"))
elseif is_windows()
  Int == Int64 || error("32-bit Windows is not currently supported. Please report this to https://github.com/FluxML/NNlib.jl")
  download("https://externalshare.blob.core.windows.net/packages/nnlib64-0.1.0.dll",
           joinpath(@__DIR__, "nnlib.dll"))
elseif is_unix()
  cd(path) do
    run(`c++ -c -fPIC -std=c++11 conv.cpp`)
    run(`c++ -shared -o nnlib.so conv.o`)
    rm("conv.o")
    mv("nnlib.so", "../deps/nnlib.so")
  end
end
