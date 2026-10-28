using NNlib, BenchmarkTools, Printf
using Base.Broadcast: broadcasted, instantiate

function lazy_logsumexp(x; dims)
    m = NNlib.fast_maximum(x; dims)
    terms = instantiate(broadcasted(-, x, m))
    return m .+ log.(sum(NNlib._exp, terms; dims, init=zero(NNlib._acc(float(eltype(x))))))
end

println("Julia $VERSION, $(Threads.nthreads()) CPU thread(s), logsumexp µs")
println("type, size, dims                                          current         lazy")
for (T, sz, dims) in ((Float32, (1000, 4096), 1), (Float32, (8192, 768), 2),
                      (Float32, (64, 64, 32, 16), (1, 2)), (Float32, (1000, 4096), :),
                      (Float16, (1000, 4096), 1))
    x = T.(randn(Float32, sz))
    @assert NNlib.logsumexp(x; dims) ≈ lazy_logsumexp(x; dims)
    before = minimum(@benchmark NNlib.logsumexp($x; dims=$dims) seconds=0.3 evals=1).time / 1000
    after = minimum(@benchmark lazy_logsumexp($x; dims=$dims) seconds=0.3 evals=1).time / 1000
    @printf("%-50s %12.1f %12.1f\n", "$T $sz dims=$dims", before, after)
    flush(stdout)
end
