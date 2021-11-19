"""
    sparsemax(x; dims = 1)

[Sparsemax](https://arxiv.org/abs/1602.02068) turns input array `x`
into sparse probability distributions that sum to 1 along the dimensions specified by `dims`.

For a matrix input `x` it will by default (`dims = 1`) treat it as a batch of vectors,
with each column independent. Keyword `dims = 2` will instead treat rows independently, and so on.

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
sparsemax(x; dims = 1) = sparsemax!(similar(x, (eltype)(x)), x; dims = dims)

sparsemax!(x; dims = 1) = sparsemax!(x, x; dims = dims)

function sparsemax!(out::AbstractArray, x::AbstractArray; dims = 1)
    # only 2D tensors are supported
    @assert dims in (1, 2)
    x = x .- maximum(x; dims=dims)

    # make ix like
    d = size(x, dims)
    if dims==1
        rhos = reshape(collect(1:d), d, 1) |> typeof(x)
    elseif dims == 2 
        rhos = reshape(collect(1:d), 1, d) |> typeof(x)
    end


    # compute threshold and support
    x_sorted = slicemap(x -> reverse(sort(x)), x; dims=dims)
    x_cumsum = cumsum(x_sorted; dims=dims) .- 1.0
    support =  rhos .* x_sorted .> x_cumsum
    support_size = vec(sum(support; dims=dims)) |> Vector{Int64}
    if dims == 1
        tau = diag(gather(transpose(x_cumsum), support_size)) ./ support_size
    elseif dims == 2
        tau = diag(gather(x_cumsum, support_size)) ./ support_size
    end

    if dims == 1
        out = clamp.(x .- transpose(tau), 0, Inf)
    elseif dims == 2
        out =  clamp.(x .- tau, 0, Inf)
    end
end


function ∇sparsemax!(Δ::AbstractArray, x::AbstractArray, y::AbstractArray; dims = 1)
   nonzeros = x[x.!=0.0]
   out .= Δ .* y
   sum = sum(out .* nonzeros; dims=dims) / sum(nonzeros; dims=dims)
   out .= nonzeros .* (out .- sum)
end

function rrule(::typeof(sparsemax), xs; dims=1)
   y = sparsemax(xs; dims=dims)
   sparsemax_pullback(Δ) = (NoTangent(), ∇sparsemax(unthunk(Δ), xs, y, dims = dims))
   return y, sparsemax_pullback
end
