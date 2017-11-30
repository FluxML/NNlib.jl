#! /usr/bin/env julia

path = joinpath(dirname(@__FILE__), "..", "src")

if is_apple()
  download("https://www.dropbox.com/s/syt6q6qv6ehc4p0/nnlib-0.1.0.dylib?dl=1",
           joinpath(@__DIR__, "nnlib.dylib"))

elseif is_windows()

  Int == Int64 || error("32-bit Windows is not currently supported. Please report this to https://github.com/FluxML/NNlib.jl")
  download("https://www.dropbox.com/s/btxuhhpo18p9rfd/nnlib64-0.1.0.dll?dl=1",
           joinpath(@__DIR__, "nnlib.dll"))

elseif is_unix()
  cd(path) do
    run(`c++ -c -fPIC -std=c++11 conv.cpp`)
    run(`c++ -shared -o nnlib.so conv.o`)
    rm("conv.o")
    mv("nnlib.so", "../deps/nnlib.so")
  end
end
