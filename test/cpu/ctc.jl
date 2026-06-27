using Test
using NNlib: ctc_loss
using Zygote: gradient
using LinearAlgebra

@testset "ctc_loss" begin
  x = rand(10, 50)
  y = rand(1:9, 30)
  @test test_gradients(x -> ctc_loss(x, y), x; rtol=1e-5, atol=1e-5)

  # tests using hand-calculated values
  x = [1. 2. 3.; 2. 1. 1.; 3. 3. 2.]
  y = [1, 2]
  @test ctc_loss(x, y) ≈ 3.6990738275138035

  g = [-0.317671 -0.427729 0.665241; 0.244728 -0.0196172 -0.829811; 0.0729422 0.447346 0.16457]
  ghat = gradient(ctc_loss, x, y)[1]
  @test g ≈ ghat rtol=1e-5 atol=1e-5

  x = [-3. 12. 8. 15.; 4. 20. -2. 20.; 8. -33. 6. 5.]
  y = [1, 2]
  @test ctc_loss(x, y) ≈ 8.02519869363453

  g = [-2.29294774655333e-06 -0.999662657278862 1.75500863563993e-06 0.00669284889063; 0.017985914969696 0.999662657278861 -1.9907078755387e-06 -0.006693150917307; -0.01798362202195 -2.52019580677916e-20 2.35699239251042e-07 3.02026677058789e-07]
  ghat = gradient(ctc_loss, x, y)[1]
  @test g ≈ ghat rtol=1e-5 atol=1e-5
end