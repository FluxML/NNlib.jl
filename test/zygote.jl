using Zygote, NNlib
using Random
using NNlib: conv, ∇conv_data, depthwiseconv, batched_mul

function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

function gradcheck(f, xs...; rtol = 1e-5, atol = 1e-5)
  grad_zygote = gradient(f, xs...)
  grad_finite_difference = ngradient(f, xs...)
  return all(isapprox.(grad_zygote, grad_finite_difference; rtol = rtol, atol = atol))
end

gradtest(f, xs::AbstractArray...; kw...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...; kw...)
gradtest(f, dims...; kw...) = gradtest(f, rand.(Float64, dims)...; kw...)

@testset "conv: spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
  x = rand(repeat([5], spatial_rank)..., 3, 2)
  w = rand(repeat([3], spatial_rank)..., 3, 3)
  cdims = DenseConvDims(x, w)
  @test gradtest((x, w) -> conv(x, w, cdims), x, w)
  @test gradtest((x, w) -> sum(conv(x, w, cdims)), x, w)  # https://github.com/FluxML/Flux.jl/issues/1055

  y = conv(x, w, cdims)
  @test gradtest((y, w) -> ∇conv_data(y, w, cdims), y, w)
  if spatial_rank == 3
    @test_broken gradtest((y, w) -> sum(∇conv_data(y, w, cdims)), y, w)
  else
    @test gradtest((y, w) -> sum(∇conv_data(y, w, cdims)), y, w)
  end

  dcdims = DepthwiseConvDims(x, w)
  @test gradtest((x, w) -> depthwiseconv(x, w, dcdims), x, w)

  y = depthwiseconv(x, w, dcdims)
  @test gradtest((y, w) -> ∇depthwiseconv_data(y, w, dcdims), y, w)
  if spatial_rank == 3
    @test_broken gradtest((y, w) -> sum(∇depthwiseconv_data(y, w, dcdims)), y, w)
  else
    @test gradtest((y, w) -> sum(∇depthwiseconv_data(y, w, dcdims)), y, w)
  end
end

@testset "pooling: spatial_rank=$spatial_rank" for spatial_rank in (1, 2)
  x = rand(repeat([10], spatial_rank)..., 3, 2)
  pdims = PoolDims(x, 2)
  @test gradtest(x -> maxpool(x, pdims), x)
  @test gradtest(x -> meanpool(x, pdims), x)
  @test gradtest(x -> sum(maxpool(x, pdims)), x)
  @test gradtest(x -> sum(meanpool(x, pdims)), x)

  #https://github.com/FluxML/NNlib.jl/issues/188
  k = ntuple(_ -> 2, spatial_rank)  # Kernel size of pool in ntuple format
  @test gradtest(x -> maxpool(x, k), x)
  @test gradtest(x -> meanpool(x, k), x)
  @test gradtest(x -> sum(maxpool(x, k)), x)
  @test gradtest(x -> sum(meanpool(x, k)), x)
end

@testset "batched matrix multiplication" begin
  rng, M, P, Q = MersenneTwister(123456), 13, 7, 11
  B = 3
  @test gradtest(batched_mul, randn(rng, M, P, B), randn(rng, P, Q, B))
end
