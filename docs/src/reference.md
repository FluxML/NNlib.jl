# Reference

The API reference of `NNlib`.

## Activation Functions

Non-linearities that go between layers of your model. Note that, unless otherwise stated, activation functions operate on scalars. To apply them to an array you can call `σ.(xs)`, `relu.(xs)` and so on.

```@docs
celu
elu
gelu
gelu_tanh
gelu_sigmoid
gelu_erf
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

## Attention 

```@docs
dot_product_attention
dot_product_attention_scores
make_causal_mask
```

## Softmax

`Flux`'s `logitcrossentropy` uses `NNlib.softmax` internally.

```@docs
softmax
logsoftmax
```

## Pooling

`Flux`'s `AdaptiveMaxPool`, `AdaptiveMeanPool`, `GlobalMaxPool`, `GlobalMeanPool`, `MaxPool`, `MeanPool` and `lpnormpool` use `NNlib.PoolDims`, `NNlib.maxpool`, `NNlib.meanpool` and `NNlib.lpnormpool` as their backend.

```@docs
PoolDims
maxpool
meanpool
lpnormpool
```

## Padding

```@docs
pad_reflect
pad_symmetric
pad_circular
pad_repeat
pad_constant
pad_zeros
```

## Convolution

`Flux`'s `Conv` and `CrossCor` layers use `NNlib.DenseConvDims` and `NNlib.conv` internally.

`NNlib.conv` supports complex datatypes on CPU and CUDA devices.

!!! note "AMDGPU MIOpen supports only cross-correlation (`flipkernel=true`)."

    Therefore for every regular convolution (`flipkernel=false`)
    kernel is flipped before calculation.
    For better performance, use cross-correlation (`flipkernel=true`)
    and manually flip the kernel before `NNlib.conv` call.
    `Flux` handles this automatically, this is only required for direct calls.

```@docs
conv
ConvDims
depthwiseconv
DepthwiseConvDims
DenseConvDims
NNlib.unfold
NNlib.fold
```

## Upsampling

`Flux`'s `Upsample` layer uses `NNlib.upsample_nearest`, `NNlib.upsample_bilinear`, and `NNlib.upsample_trilinear` as its backend. Additionally, `Flux`'s `PixelShuffle` layer uses `NNlib.pixel_shuffle` as its backend.

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

## Rotation
Rotate images in the first two dimensions of an array.

```@docs
imrotate
∇imrotate
```

## Batched Operations

`Flux`'s `Bilinear` layer uses `NNlib.batched_mul` internally.

```@docs
batched_mul
batched_mul!
batched_adjoint
batched_transpose
batched_vec
```

## Gather and Scatter

`Flux`'s `Embedding` layer uses `NNlib.gather` as its backend.

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
NNlib.glu
NNlib.within_gradient
bias_act!
```
