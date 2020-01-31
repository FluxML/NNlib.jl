## Convolution and Pooling API
#
#  We provide the following generic methods, for 3d, 4d, and 5d tensors, calculating 1d,
#  2d and 3d convolutions and pooling based on the rank of the input tensors, in both
#  mutating and non-mutating/auto-allocating variants:
#   - Convolution:
#     - conv(x, w, cdims)
#     - conv!(y, x, w, cdims)
#   - Convolution data backpropagation
#     - ∇conv_data(dy, w, cdims)
#     - ∇conv_data!(dx, dy, w, cdims)
#   - Convolution filter backpropagation
#     - ∇conv_filter(x, dy, cdims)
#     - ∇conv_filter!(dw, x, dy, cdims)
#   - Pooling:
#     - maxpool(x, pdims)
#     - maxpool!(y, x, pdims)
#     - meanpool(x, pdims)
#     - meanpool!(y, x, pdims)
#   - Pooling data backprop
#     - ∇maxpool(dy, y, x, pdims)
#     - ∇maxpool!(dx, dy, y, x, pdims)
#     - ∇meanpool(dy, y, x, pdims)
#     - ∇meanpool!(dx, dy, y, x pdims)
#
#   All methods require a `ConvDims` or `PoolDims` object to define the dimensions and
#   meta elements of the convolution (padding, stride, dilation, kernel-flipping, etc...)
#   which is easily constructable through something like `DenseConvDims(x, w)`.  All
#   methods take in the `ConvDims` of the associated normal, forward-pass convolution,
#   that is, the following is legal:
#
#       cdims = DenseConvDims(x, w; stride=2, dilation=(3,2))
#       dx = ∇conv_data(conv(x, w, cdims), w, cdims)
#
#   Note that we do provide a helper API in the case that you don't want to bother with
#   DenseConvDims and friends: you can simply do the following, however it will be less
#   performant if you run the same operation multiple times:
#
#       y = conv(x, w; stride=2, dilation=(3,2))


# We support a pluggable backend system, currently consisting of three possible backends:
#  * `nnpack`: which uses the third-party NNPACK libraries for convolution and pooling
#  * `im2col`: A Julia BLAS-based implementation of convolution
#  * `direct`: A Julia-native direct implementation of convolution and pooling
#
# We store each within a module (in the case of NNPACK, it is included only if the
# NNPACK binaries are available for the host system) and each module pushes a value
# onto these `conv_backends` lists.  Those lists are then read from in the file
# `interface_impl.jl` which generates the nice interface described above, using a mixture
# of dispatch and runtime checks to provide the convenient `conv()` -> `conv!()` ->
# `conv_nnpack!()` interface that we all know and love.
conv_backends = Dict(
    :conv                   => [],
    :∇conv_data             => [],
    :∇conv_filter           => [],
    :depthwiseconv          => [],
    :∇depthwiseconv_data    => [],
    :∇depthwiseconv_filter  => [],
)

# Same thing for pooling
pooling_backends = Dict(
    :maxpool  => [],
    :meanpool => [],
)