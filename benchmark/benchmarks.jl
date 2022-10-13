using BenchmarkTools
using NNlib

const SUITE = BenchmarkGroup()

SUITE["activations"] = BenchmarkGroup()

x = rand(64, 64)

for f in NNlib.ACTIVATIONS
    act = @eval($f)
    SUITE["activations"][string(f)] = @benchmarkable $act.($x)
end
