import numpy as np

# Pixel shuffle used in sub-pixel convolutions, in Chainer
# https://arxiv.org/pdf/1609.05158v2.pdf
#
# Example:
# Scaling factor: 3
# In shape: (1, 9, 4, 4)
# Out shape: (1, 1, 12, 12)
#
# `np` may be replaced with `cupy` (after `import cupy`) to perform
# the same computations on the GPU

upscale_factor = 2
x = np.empty((1, 4, 2, 2), dtype=np.float32)

n, c, w, h = x.shape

# Set all values in each feature map to its feature map index
for i in range(c):
    x[0, i] = i

c_out = c // upscale_factor ** 2
w_out = w * upscale_factor
h_out = h * upscale_factor

x = np.reshape(x, (n, c_out, upscale_factor, upscale_factor, w, h))
x = np.transpose(x, (0, 1, 4, 2, 5, 3))
x = np.reshape(x, (n, c_out, w_out, h_out))

# assert(x.shape == (1, 1, 12, 12))
