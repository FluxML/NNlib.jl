using BenchmarkTools
using NNlib

const SUITE = BenchmarkGroup()

SUITE["activations"] = BenchmarkGroup()

x = rand(64, 64)

for f in NNlib.ACTIVATIONS
    SUITE["activations"][string(f)] = @benchmarkable $f.($x)
end