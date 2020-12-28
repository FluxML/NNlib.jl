const rng = StableRNG(123)

gradtest((x, W, b) -> relu.(W*x .+ b), 5, (2,5), 2)
gradtest((x, W, b) -> relu.(W*x .+ b), (5,3), (2,5), 2)
gradtest((x, W, b) -> selu.(W*x .+ b), 5, (2,5), 2)
gradtest((x, W, b) -> selu.(W*x .+ b), (5,3), (2,5), 2)
gradtest((x, W, b) -> elu.(W*x .+ b, 2), 5, (2,5), 2)
gradtest((x, W, b) -> elu.(W*x .+ b, 2), (5,3), (2,5), 2)

# tests for https://github.com/FluxML/Zygote.jl/issues/758
@test gradient(xs -> sum(selu.(xs)), [1_000, 10_000])[1] ≈ [1.0507009873554805, 1.0507009873554805] rtol=1e-8
@test gradient(x -> selu(x), 1_000) == (1.0507009873554805,)
@test gradient(xs -> sum(elu.(xs, 2)), [1_000, 10_000]) == ([1., 1.],)
@test gradient(x -> elu(x, 2), 1_000) == (1.,)
@test gradient(x -> elu(x, 2), -1) == (2*exp(-1),)
zygote_gradient_test(x->sum(selu.(x)),[100., 1_000.])
zygote_gradient_test(x->sum(elu.(x, 3.5)),[100., 1_000.])
zygote_gradient_test(x->sum(elu.(x, 3.5)),[1_000., 10_000.]) # for elu the tests are passing but for selu not, interesting
# numerical instability even for the linear part of such function, see:
# julia> ngradient(x->sum(selu.(x)),[1_000., 10_000.])
# ([1.0506591796875, 1.0506591796875],)
# julia> gradient(x->sum(selu.(x)),[1_000., 10_000.])
# ([1.0507009873554805, 1.0507009873554805],)
# @test_broken zygote_gradient_test(x->sum(selu.(x)),[1_000., 10_000.])
@test zygote_gradient_test(x->sum(selu.(x)),[1_000., 10_000.])

gradtest((x, W, b) -> σ.(W*x .+ b), 5, (2,5), 2)
gradtest((x, W, b) -> σ.(W*x .+ b), (5,3), (2,5), 2)
gradtest((x, W, b) -> logσ.(W*x .+ b), 5, (2,5), 2)
gradtest((x, W, b) -> logσ.(W*x .+ b), (5,3), (2,5), 2)

gradtest(x -> softmax(x).*(1:3), 3)
gradtest(x -> softmax(x).*(1:3), (3,5))
gradtest(x -> softmax(x, dims=2).*(1:3), (3,5))
gradtest(x -> logsoftmax(x).*(1:3), 3)
gradtest(x -> logsoftmax(x).*(1:3), (3,5))
gradtest(x -> logsoftmax(x, dims=2).*(1:3), (3,5))

@testset "conv: spatial_rank=$spatial_rank" for spatial_rank in (1, 2, 3)
  x = rand(rng, repeat([5], spatial_rank)..., 3, 2)
  w = rand(rng, repeat([3], spatial_rank)..., 3, 3)
  cdims = DenseConvDims(x, w)
  gradtest((x, w) -> conv(x, w, cdims), x, w)
  gradtest((x, w) -> sum(conv(x, w, cdims)), x, w)  # https://github.com/FluxML/Flux.jl/issues/1055

  y = conv(x, w, cdims)
  gradtest((y, w) -> ∇conv_data(y, w, cdims), y, w)
  # if spatial_rank == 3
  #   @test_broken gradtest((y, w) -> sum(∇conv_data(y, w, cdims)), y, w)
  # else
    gradtest((y, w) -> sum(∇conv_data(y, w, cdims)), y, w)
  # end

  dcdims = DepthwiseConvDims(x, w)
  gradtest((x, w) -> depthwiseconv(x, w, dcdims), x, w)

  y = depthwiseconv(x, w, dcdims)
  gradtest((y, w) -> ∇depthwiseconv_data(y, w, dcdims), y, w)
  # if spatial_rank == 3
  #   @test_broken gradtest((y, w) -> sum(∇depthwiseconv_data(y, w, dcdims)), y, w)
  # else
    gradtest((y, w) -> sum(∇depthwiseconv_data(y, w, dcdims)), y, w)
  # end
end

@testset "pooling: spatial_rank=$spatial_rank" for spatial_rank in (1, 2)
  x = rand(rng, repeat([10], spatial_rank)..., 3, 2)
  pdims = PoolDims(x, 2)
  gradtest(x -> maxpool(x, pdims), x; broken=spatial_rank <= 2)
  gradtest(x -> meanpool(x, pdims), x)
  gradtest(x -> sum(maxpool(x, pdims)), x)
  gradtest(x -> sum(meanpool(x, pdims)), x)

  #https://github.com/FluxML/NNlib.jl/issues/188
  k = ntuple(_ -> 2, spatial_rank)  # Kernel size of pool in ntuple format
  gradtest(x -> maxpool(x, k), x; broken=spatial_rank <= 2)
  gradtest(x -> meanpool(x, k), x)
  gradtest(x -> sum(maxpool(x, k)), x)
  gradtest(x -> sum(meanpool(x, k)), x)
end

@testset "batched matrix multiplication" begin
  M, P, Q = 13, 7, 11
  B = 3
  gradtest(batched_mul, randn(rng, M, P, B), randn(rng, P, Q, B))
end
