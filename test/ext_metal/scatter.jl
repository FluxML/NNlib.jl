# Regression tests for https://github.com/FluxML/NNlib.jl/issues/534
# `scatter` used to fail to compile on Metal for several reduction operators
# (float `max`/`min` errored, `*`/`/` silently no-oped). Metal does not support
# Float64/Int64 arrays, so the scattered data here is always Float32.

@testset "scatter" begin
    # dims == 0 → vector destination, dims == 1 → matrix destination
    srcs = Dict(
        0 => Float32.(ones(3) * collect(1:4)'),                              # 3×4
        1 => Float32.([1, 2] .* reshape(ones(3) * collect(1:4)', 1, 3, 4)),  # 2×3×4
    )
    idx = [1 2 3 4;
           4 2 1 3;
           3 5 5 3]

    @testset "forward op=$op" for op in (+, -, *, /, max, min, mean)
        for dims in (0, 1)
            src = srcs[dims]
            # both the allocating and mutating entry points exercise the atomic
            # kernel that previously errored / silently no-oped on Metal
            gputest(DEVICE, (s, i) -> NNlib.scatter(op, s, i), src, idx; checkgrad=false)

            dstsz = (size(src)[1:dims]..., maximum(idx))
            dst = fill(NNlib.scatter_empty(op, Float32), dstsz)
            gputest(DEVICE, (d, s, i) -> NNlib.scatter!(op, d, s, i), dst, src, idx; checkgrad=false)
        end
    end

    # Gradients wrt `src` match the CPU reference. (`*`/`/` gradients are excluded:
    # the GPU `∇scatter_src` for these ops needs the ragged reverse-index buffer
    # uploaded to the device, which the CUDA extension special-cases but Metal does
    # not yet support.)
    @testset "gradient op=$op" for op in (+, -, max, min, mean)
        for dims in (0, 1)
            gputest(DEVICE, (s, i) -> NNlib.scatter(op, s, i), srcs[dims], idx; checkgrad=true)
        end
    end
end
