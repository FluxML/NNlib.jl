
@static if is_unix()
    const libnnlib = Libdl.find_library(["conv.so"], [dirname(@__FILE__)])
elseif is_windows()
    const libnnlib = Libdl.find_library(["conv.dll"], [dirname(@__FILE__)])
end
