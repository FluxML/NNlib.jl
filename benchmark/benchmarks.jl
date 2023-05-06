using BenchmarkTools
using NNlib
using NNlib.ChainRulesCore: rrule
using Random

Random.seed!(1234567890)

const SUITE = BenchmarkGroup()

SUITE["activations"] = BenchmarkGroup()
let x = rand(64, 64)
    for f in NNlib.ACTIVATIONS
        act = @eval($f)
        SUITE["activations"][string(f)] = @benchmarkable $act.($x)
    end
end

for (fn!, fn_bw) in [(softmax!, NNlib.∇softmax_data), (logsoftmax!, NNlib.∇logsoftmax_data)]
    fn_suite = BenchmarkGroup()
    SUITE[rstrip(string(fn!), '!')] = fn_suite
    let SIZES = [
        (128, 384, 8),
        (512, 784, 8),
        (768, 1024, 4),
        (1024, 2048, 4),
        (2048, 2048, 2),
        (4096, 2048, 2),
        (4096, 4096, 2),
        (12288, 2048, 1)
    ]
        for et in (Float16, Float32)
            et_suite = BenchmarkGroup("fw" => BenchmarkGroup(), "bw" => BenchmarkGroup())
            fn_suite[string(et)] = et_suite
            for sz in SIZES
                x = randn(et, sz)
                y = similar(x)
                dy = zero(x)
                fn!(y, x)
                et_suite["fw"][string(sz)] = @benchmarkable $fn!($y, $x)
                et_suite["bw"][string(sz)] = @benchmarkable $fn_bw($dy, $y)
            end
        end
    end
end

