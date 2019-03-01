# This script does within-NNlib-version comparisons

if length(ARGS) != 1
    println("Usage: compare.jl <results.jld2>")
    exit(1)
end

using NNlib, BenchmarkTools, JLD2

@load ARGS[1] results

mkpath("figures")

# We will analyze the data on a 
