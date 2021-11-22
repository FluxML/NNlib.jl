"""
    sparsemax(x; dims = 1)

[Sparsemax](https://arxiv.org/abs/1602.02068) turns input array `x`
into sparse probability distributions that sum to 1 along the dimensions specified by `dims`.

Similar to softmax, each dimension is considered independent. For a matrix input `x` it will 
by default (`dims = 1`) treat it as a batch of vectors, with each column independent. Keyword 
`dims = 2` will instead treat rows independently.

# Examples

```jldoctest
julia> sparsemax([1 2 3; 2 2 2])  # dims=1
2×3 Matrix{Float64}:
0.0  0.5  1.0
1.0  0.5  0.0

julia> sparsemax([1 2 3; 2 2 2]; dims=2)
2×3 Matrix{Float64}:
0.0       0.0       1.0
0.333333  0.333333  0.333333
```
"""
function sparsemax(x::AbstractArray; dims::Integer=1)
    z = slicemap(x -> reverse(sort(x)), x; dims=dims)  # float is usually free, except on integers etc.
    mask = _sparsemax_mask(z, dims)
    tausum = sum(z .* mask; dims)  # no longer need z
    kay = sum(mask; dims)
    z = _relu.(x  .- (tausum .- 1) ./ kay)
end

function _sparsemax_mask(z::AbstractArray, dim::Integer)
    acc = cumsum(z; dims=dim)
    if dim == 1
        acc = 1 .+ axes(z,1) .* z .> acc
    elseif dim == 2
        acc = 1 .+ axes(z,2)' .* z .> acc
    else
        # This isn't type-stable. Writing into `acc` ensures the whole function still is:
        cnt = reshape(axes(x, dim), ntuple(_->1, dim-1)..., :)
        acc = 1 .+ cnt .* z .> acc
    end
    acc
end

_relu(x) = _ifelse(x>0, x, false)  # different gradient at zero

_ifelse(p, x, y) = ifelse(p, promote(x, y)...)

function ∇sparsemax(dy::AbstractArray, y::AbstractArray; dims::Integer=1)
    vee = sum(dy .* (y .> 0); dims)
    kay = count(>(0), y; dims)  # could also keep from forward pass?
    _ifelse.(y .> 0, dy .- vee ./ kay, 0)
end

function rrule(::typeof(sparsemax), xs; dims=1)
    y = sparsemax(xs; dims=dims)
    sparsemax_pullback(Δ) = (NoTangent(), ∇sparsemax(xs, y, dims = dims))
    return y, sparsemax_pullback
 end