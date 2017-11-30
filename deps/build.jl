#! /usr/bin/env julia

path = joinpath(dirname(@__FILE__), "..", "src")

version = "0.1.0"

if is_apple()
  download("https://www.dropbox.com/s/syt6q6qv6ehc4p0/nnlib-$version.dylib?dl=1",
           joinpath(@__DIR__, "nnlib.dylib"))

elseif is_windows()

    import WinRPM
    info("Compiling with WinRPM gcc-c++")
    WinRPM.install("gcc-c++"; yes = true)
    WinRPM.install("gcc"; yes = true)
    WinRPM.install("headers"; yes = true)
    WinRPM.install("winpthreads-devel"; yes = true)

    gpp = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","bin","g++")
    RPMbindir = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","bin")
    incdir = Pkg.dir("WinRPM","deps","usr","x86_64-w64-mingw32","sys-root","mingw","include")

    push!(Base.Libdl.DL_LOAD_PATH, RPMbindir)
    ENV["PATH"] = ENV["PATH"] * ";" * RPMbindir
    if success(`$gpp --version`)
        cd(path) do
            run(`$gpp -c -fPIC -std=c++11 conv.cpp -I $incdir`)
            run(`$gpp -shared -o conv.dll conv.o`)
            rm("conv.o")
            mv("conv.dll", "..\\deps\\conv.dll")
        end
    else
        error("no windows c++ compiler (cl.exe) found, and WinRPM with g++ is failing as well.")
    end

elseif is_unix()
    cd(path) do
        # Note: on Mac OS X, g++ is aliased to clang++.
        run(`c++ -c -fPIC -std=c++11 conv.cpp`)
        run(`c++ -shared -o conv.so conv.o`)
        rm("conv.o")
        mv("conv.so", "../deps/conv.so")
    end
end
