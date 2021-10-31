using NNlib: sigmoid
"""
    glu(x, dim = 1)

The gated linear unit.

Calculates `glu(x, dim) = a ⊗ σ(b)`, where
  `x` is split in half along given dimension `dim` to form `a` and `b`
  then the element-wise product between above matrices is computed
See ["Language Modeling with Gated Convolutional Networks"](https://arxiv.org/abs/1612.08083).
"""
function glu(x, dim = 1)
    maxdim = size(x, dim)
    @assert maxdim % 2 == 0 "Dimension must be even"
    half = maxdim ÷ 2
    a, b = selectdim(x, dim, 1:half), selectdim(x, dim, half+1:maxdim)
    a .* sigmoid.(b)
end

