using CuArrays
using NNlib: @fix

"""https://github.com/FluxML/NNlib.jl/issues/37"""
@testset "fix#37" begin
  m1(x,f) = @fix x.*f.(x)

  function m2(x,f)
    return @fix x.*f.(x)
  end

  function m3(x,f)
    y = @fix x.*f.(x)
    y .+ y
  end

  @testset "$T $(length(d))" for d in [[2.0], 1.0:10], T in [Float32, Float64]
    td = T.(d)

    @testset "$m $f" for m in [m1, m2, m3], f in [log, exp, sin]
      @test m(td, f) ≈ Array(m(CuArray(td), f))
    end
  end

  # expect failure if an intrinsic for the given type (i.e Int) is not available
  @test_throws ErrorException m1(CuArray([2]), log)

  # abs is defined for 4 types
  d = -5:5
  @testset "$T $m" for m in [m1, m2, m3], T in [Float32, Float64, Int32, Int64]
    td = T.(d)
    @test m(td, abs) == m(CuArray(td), abs)
  end
end
