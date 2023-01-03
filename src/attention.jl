const AA3{T} = AbstractArray{T,3}
const AA4{T} = AbstractArray{T,4}
const AA{N,T} = AbstractArray{T,N}

"""
    dot_product_attention(query, key, value; [bias, droput_fn, mask, num_heads])

Multihead dot product attention used in transformer architectures. 

The input arrays must have the first two dimensions given by the number of features
and the sequece length, then an arbitrary number of batch dimensions or none. 

# Arguments

- `query`: Query array of size `(qk_dim, q_len, batch_size...)`.
- `key`: Key array of size `(qk_dim, kv_len, batch_size...)`.
- `value`: Value array of size `(v_dim, kv_len, batch_size...)`.
- `bias`: Either `nothing` or an input array broadcastable to size `(kv_len, q_len, num_heads, batch_size)`. 
          Can also be set to `mask=:causal` to apply a causal mask. Default `nothing`.
- `dropout_fn`: A dropout function to apply on the attention scores. Default `nothing`. 
- `mask`: Either `nothing` or an input array broadcastable to size `(kv_len, q_len, num_heads, batch_size)`. 
          Can also be set to `mask=:causal` to apply a causal mask. Default `nothing`.
- `num_heads`: Number of heads to split the input arrays into. Default `1`.

# Examples
    
```julia
q, k, v = rand(10, 20, 2), rand(10, 30, 2), rand(20, 30, 2)
y, α = dot_product_attention(q, k, v)
```
"""
function dot_product_attention(q::AA{N}, k::AA{N}, v::AA{N}; num_heads=1, kws...) where N
    batch_size = size(q)[3:end]
    
    batch_size == size(k)[3:end] == size(v)[3:end] || throw(ArgumentError("Batch dimensions have to be the same."))
    size(q, 1) == size(k, 1) || throw(ArgumentError("First dimension in query and key has to be the same."))
    size(k, 2) == size(v, 2)  || throw(ArgumentError("Second dimension in key and value has to be the same."))
    
    q, k, v = map(x -> reshape(x, size(x, 1), size(x, 2), :), (q, k, v))

    # Multihead attention. TODO create fastpath for singlehead attention.
    q, k, v = split_heads.((q, k, v), num_heads)
    x, α = _dot_product_attention(q, k, v; kws...)
    x = join_heads(x)

    x = reshape(x, size(x, 1), size(x, 2), batch_size...)
    α = reshape(α, size(α)[1:3]..., batch_size...)
    return x, α
end

function _dot_product_attention(q::AA4, k::AA4, v::AA4; 
        dropout_fn=nothing, bias=nothing, mask=nothing)

    α = dot_product_attention_scores(q, k; dropout_fn, bias, mask)
    # [α] = [kv_len, q_len, num_heads, batch_size]
    
    # The following permutedims and batched_mul are equivalent to
    # @tullio x[d, h, i, b] := α[j, i, h, b] * v[d, h, j, b]
    vt = permutedims(v, (1, 3, 2, 4))
    x = batched_mul(vt, α)
    x = permutedims(x, (1, 3, 2, 4))
    # [x] = [kv_dim ÷ num_heads, num_heads, q_len, batch_size]
    return x, α
end

"""
    dot_product_attention_scores(query, key; [bias, droput_fn, mask])

Return the attention scores for the [`dot_product_attention`](@ref).

Input arrays must have dimensions `(num_features ÷ num_heads, num_heads, sequence_length, batch_size)`

"""
function dot_product_attention_scores(q::AA4{T}, k::AA4{T}; 
            dropout_fn=nothing, mask=nothing, bias=nothing) where T

    q  = q ./ √T(size(q, 1))
    
    # The following permutedims and batched_mul are equivalent to
    # @tullio α[j, i, h, b] := q[d, h, i, b] * k[d, h, j, b]
    kt = permutedims(k, (3, 1, 2, 4))
    qt = permutedims(q, (1, 3, 2, 4))
    α = batched_mul(kt, qt)
    # [α] = [kv_len, q_len, num_heads, batch_size]

    if bias !== nothing
        α = α .+ bias
    end

    if mask !== nothing
        if mask === :causal
            mask = make_causal_mask(α)
        end
        neginf = typemin(eltype(α))
        α = ifelse.(mask, α, neginf)
    end

    α = softmax(α, dims=1)
    return dropout_fn === nothing ? α : dropout_fn(α)
end

""" 
    make_causal_mask(x)

Return a boolean square matrix `m` of the same type as `x` and of side `size(x,2)`.
Its elements are set such that `m[i, j] == i ≤ j`. 

Can be used to mask the attention scores in [`dot_product_attention`](@ref).
"""
function make_causal_mask(x::AbstractArray)
  len = size(x, 2)
  mask = triu(trues_like(x, (len, len)))
  return mask
end

trues_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), true)
falses_like(x::AbstractArray, sz=size(x)) = fill!(similar(x, Bool, sz), false)

split_heads(x, num_heads) = reshape(x, size(x, 1) ÷ num_heads, num_heads, size(x)[2:end]...)
join_heads(x) = reshape(x, :, size(x)[3:end]...)

@non_differentiable make_causal_mask(x)
@non_differentiable trues_like(::Any...)
@non_differentiable falses_like(::Any...)

