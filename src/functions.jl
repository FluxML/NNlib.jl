# The two halves of `x` along `dim`, as views of `x` seen as `(prod(size(x)[1:dim-1]) * half, 2, prod(size(x)[dim+1:end]))`.
# Unlike `selectdim` the type of the views does not depend on `dim`, so the result is inferred.
function _split_halves(x::AbstractArray{<:Any,N}, dim) where {N}
    d = size(x, dim)
    @assert d % 2 == 0 "Dimension must be even"
    half = d ÷ 2
    pre = prod(ntuple(i -> i < dim ? size(x, i) : 1, Val(N)))
    post = prod(ntuple(i -> i > dim ? size(x, i) : 1, Val(N)))
    x3 = reshape(x, pre * half, 2, post)
    return view(x3, :, 1, :), view(x3, :, 2, :)
end

_halved(y, x, dim) = reshape(y, ntuple(i -> i == dim ? size(x, i) ÷ 2 : size(x, i), Val(ndims(x))))

"""
    glu(x, dim = 1)

The gated linear unit from the ["Language Modeling with Gated Convolutional Networks"](https://arxiv.org/abs/1612.08083) paper.

Calculates `a .* sigmoid(b)`, where `x` is split in half along given dimension `dim` to form `a` and `b`.

See also [`swiglu`](@ref).
"""
function glu(x, dim = 1)
    a, b = _split_halves(x, dim)
    _halved(a .* sigmoid.(b), x, dim)
end

"""
    swiglu(x, dim = 1)

The swish-gated linear unit from the ["GLU Variants Improve Transformer"](https://arxiv.org/abs/2002.05202) paper,
the feed-forward activation of Llama, PaLM and most language models since.

Calculates `swish.(gate) .* up`, where `x` is split in half along given dimension `dim`
to form `gate` (the first half) and `up` (the second half).
This is the layout of the fused gate and up projections of Llama-style models, and the reverse of
[`glu`](@ref), where the gate is the second half.

See also [`swish`](@ref).

# Examples

```julia
ffn = Chain(Dense(d => 2h), NNlib.swiglu, Dense(h => d))
```
"""
function swiglu(x, dim = 1)
    gate, up = _split_halves(x, dim)
    _halved(swish.(gate) .* up, x, dim)
end
