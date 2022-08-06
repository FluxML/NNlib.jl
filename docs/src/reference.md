# Reference

The API reference of `NNlib`.

## Activation Functions

Non-linearities that go between layers of your model. Note that, unless otherwise stated, activation functions operate on scalars. To apply them to an array you can call `σ.(xs)`, `relu.(xs)` and so on.

```@docs
celu
elu
gelu
hardsigmoid
sigmoid_fast
hardtanh
tanh_fast
leakyrelu
lisht
logcosh
logsigmoid
mish
relu
relu6
rrelu
selu
sigmoid
softplus
softshrink
softsign
swish
hardswish
tanhshrink
trelu
```

## Softmax

```@docs
softmax
logsoftmax 
```

## Pooling

```@docs
PoolDims
maxpool
meanpool
```

## Padding

```@docs
pad_reflect
pad_constant
pad_repeat
pad_zeros
```

## Convolution

```@docs
conv
ConvDims
depthwiseconv
DepthwiseConvDims
DenseConvDims
```

## Upsampling

```@docs
upsample_nearest
∇upsample_nearest
upsample_linear
∇upsample_linear
upsample_bilinear
∇upsample_bilinear
upsample_trilinear
∇upsample_trilinear
pixel_shuffle
```

## Batched Operations

```@docs
batched_mul
batched_mul!
batched_adjoint
batched_transpose
batched_vec
```

## Gather and Scatter

```@docs
NNlib.gather
NNlib.gather!
NNlib.scatter
NNlib.scatter!
```

## Sampling

```@docs
grid_sample
∇grid_sample
```

## Losses

```@docs
ctc_loss
```

## Miscellaneous

```@docs
logsumexp
```
