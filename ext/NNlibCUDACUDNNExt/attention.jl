import NNlib: _scaled_dot_product_attention
using ChainRulesCore: ChainRulesCore, HasReverseMode, NoTangent, RuleConfig,
                       rrule_via_ad, unthunk

# cuDNN's fused scaled dot-product attention takes half-precision arrays in NNlib's
# `(head_dim, nheads, seq_len, batch_size)` layout, and never materializes the attention
# scores, the causal mask or the repeated key/value heads of grouped-query attention.
const CUDNNSDPAFloat = Union{Float16,BFloat16}

generic_sdpa(q, k, v, bias; kws...) = first(NNlib._attention(q, k, v, bias; kws...))

# The output buffer for the fused kernel, or `nothing` when it does not apply: it has no
# bias, dropout or arbitrary mask, and its output has the head dimension of the query.
function cudnn_sdpa_output(q, k, v, bias, fdrop, mask, is_causal)
    bias === nothing && fdrop === identity && mask === nothing || return nothing
    size(v, 1) == size(q, 1) || return nothing
    out = similar(q)
    cuDNN.attention_supported(out, q, k, v; causal=is_causal) && return out
    CUDA.unsafe_free!(out)
    return nothing
end
ChainRulesCore.@non_differentiable cudnn_sdpa_output(::Any...)

function _scaled_dot_product_attention(q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                                       v::DenseCuArray{T,4}, bias;
                                       fdrop, mask, scale, is_causal) where {T<:CUDNNSDPAFloat}
    out = cudnn_sdpa_output(q, k, v, bias, fdrop, mask, is_causal)
    out === nothing &&
        return generic_sdpa(q, k, v, bias; fdrop, mask, scale, is_causal)
    return cudnn_sdpa!(out, q, k, v, is_causal, something(scale, inv(sqrt(size(q, 1)))))
end

function cudnn_sdpa!(out, q, k, v, causal, scale)
    cuDNN.attention!(out, q, k, v; scale, causal)
    return out
end

function ChainRulesCore.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(cudnn_sdpa!),
                              out, q, k, v, causal, scale)
    # The softmax statistics that the forward pass saves for the backward one.
    stats = similar(q, Float32, (1, size(q, 3), size(q, 2), size(q, 4)))
    # The gradients have the sizes of `q`, `k`, `v` and `out`, which stand in for them.
    if !cuDNN.attention_backward_supported(q, k, v, out, q, k, v, out, stats; causal)
        # cuDNN has no engine for the backward pass: differentiate the generic one.
        CUDA.unsafe_free!(stats)
        fallback = (q, k, v) -> generic_sdpa(q, k, v, nothing; fdrop=identity,
                                             mask=nothing, scale, is_causal=causal)
        y, generic_pullback = rrule_via_ad(cfg, fallback, q, k, v)
        function fallback_pullback(Δ)
            _, dq, dk, dv = generic_pullback(Δ)
            return NoTangent(), NoTangent(), dq, dk, dv, NoTangent(), NoTangent()
        end
        return copyto!(out, y), fallback_pullback
    end
    cuDNN.attention!(out, q, k, v; scale, causal, stats)
    function cudnn_sdpa_pullback(Δ)
        dO = unthunk(Δ)
        dO isa DenseCuArray{eltype(q),4} || (dO = CuArray{eltype(q)}(dO))
        dq, dk, dv = cuDNN.attention_backward(dO, q, k, v, out, stats; scale, causal)
        return NoTangent(), NoTangent(), dq, dk, dv, NoTangent(), NoTangent()
    end
    return out, cudnn_sdpa_pullback
end
