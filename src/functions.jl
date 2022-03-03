"""
    glu(x, dim = 1)

The gated linear unit from the ["Language Modeling with Gated Convolutional Networks"](https://arxiv.org/abs/1612.08083) paper.

Calculates `a .* sigmoid(b)`, where `x` is split in half along given dimension `dim` to form `a` and `b`.
"""
function glu(x, dim = 1)
    maxdim = size(x, dim)
    @assert maxdim % 2 == 0 "Dimension must be even"
    half = maxdim ÷ 2
    a, b = selectdim(x, dim, 1:half), selectdim(x, dim, half+1:maxdim)
    a .* sigmoid.(b)
end

