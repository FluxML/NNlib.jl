import NNlib: _scaled_dot_product_attention, _is_fused_attention
using ChainRulesCore: ChainRulesCore, HasReverseMode, NoTangent, RuleConfig,
                       rrule_via_ad, unthunk

# cuDNN's fused scaled dot-product attention takes half-precision arrays in NNlib's
# `(head_dim, nheads, seq_len, batch_size)` layout, and never materializes the attention
# weights, the causal and padding masks or the repeated key/value heads of grouped-query
# attention. It runs on tensor cores, which NNlib's own fused kernel does not use: that
# one takes the calls cuDNN has no engine for.
const CUDNNSDPAFloat = Union{Float16,BFloat16}

# the materialized implementation, which can be differentiated
generic_sdpa(q, k, v, bias; kws...) = first(NNlib._attention(q, k, v, bias; kws...))

# cuDNN takes the lengths of both the queries and the keys, or of neither.
function cudnn_sdpa_lengths(q, k, q_lengths, kv_lengths)
    q_lengths === nothing && kv_lengths === nothing && return nothing, nothing
    q_lengths === nothing && (q_lengths = full_lengths(q))
    kv_lengths === nothing && (kv_lengths = full_lengths(k))
    return q_lengths, kv_lengths
end
full_lengths(x) = fill!(similar(x, Int32, (1, 1, 1, size(x, 4))), size(x, 3))
ChainRulesCore.@non_differentiable cudnn_sdpa_lengths(::Any...)

# The output buffer for the fused kernel, or `nothing` when it does not apply: it has no
# bias, dropout or arbitrary mask, and its output has the head dimension of the query.
function cudnn_sdpa_output(q, k, v, bias, fdrop, mask, is_causal, seq_len_q, seq_len_kv)
    bias === nothing && fdrop === identity && mask === nothing || return nothing
    size(v, 1) == size(q, 1) || return nothing
    out = similar(q)
    cuDNN.attention_supported(out, q, k, v; causal=is_causal, seq_len_q, seq_len_kv) &&
        return out
    CUDA.unsafe_free!(out)
    return nothing
end
ChainRulesCore.@non_differentiable cudnn_sdpa_output(::Any...)

function _scaled_dot_product_attention(q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                                       v::DenseCuArray{T,4}, bias;
                                       fdrop, mask, scale, is_causal,
                                       q_lengths, kv_lengths) where {T<:CUDNNSDPAFloat}
    seq_len_q, seq_len_kv = cudnn_sdpa_lengths(q, k, q_lengths, kv_lengths)
    out = cudnn_sdpa_output(q, k, v, bias, fdrop, mask, is_causal, seq_len_q, seq_len_kv)
    out === nothing &&
        return NNlib.generic_attention(q, k, v, bias; fdrop, mask, scale, is_causal,
                                       q_lengths, kv_lengths)
    return cudnn_sdpa!(out, q, k, v, is_causal, something(scale, inv(sqrt(size(q, 1)))),
                       seq_len_q, seq_len_kv)
end

function _is_fused_attention(q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                             v::DenseCuArray{T,4}, bias;
                             fdrop, mask, scale, is_causal,
                             q_lengths, kv_lengths) where {T<:CUDNNSDPAFloat}
    seq_len_q, seq_len_kv = cudnn_sdpa_lengths(q, k, q_lengths, kv_lengths)
    out = cudnn_sdpa_output(q, k, v, bias, fdrop, mask, is_causal, seq_len_q, seq_len_kv)
    out === nothing && return NNlib.fused_attention_applies(q, k, v, bias, fdrop, mask)
    CUDA.unsafe_free!(out)
    return true
end

function cudnn_sdpa!(out, q, k, v, causal, scale, seq_len_q, seq_len_kv)
    cuDNN.attention!(out, q, k, v; scale, causal, seq_len_q, seq_len_kv)
    return out
end

function ChainRulesCore.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(cudnn_sdpa!),
                              out, q, k, v, causal, scale, seq_len_q, seq_len_kv)
    # The softmax statistics that the forward pass saves for the backward one.
    stats = similar(q, Float32, (1, size(q, 3), size(q, 2), size(q, 4)))
    # The gradients have the sizes of `q`, `k`, `v` and `out`, which stand in for them.
    if !cuDNN.attention_backward_supported(q, k, v, out, q, k, v, out, stats;
                                           causal, seq_len_q, seq_len_kv)
        # cuDNN has no engine for the backward pass: differentiate the generic one.
        CUDA.unsafe_free!(stats)
        fallback = (q, k, v) -> generic_sdpa(q, k, v, nothing; fdrop=identity,
                                             mask=nothing, scale, is_causal=causal,
                                             q_lengths=seq_len_q, kv_lengths=seq_len_kv)
        y, generic_pullback = rrule_via_ad(cfg, fallback, q, k, v)
        function fallback_pullback(Δ)
            _, dq, dk, dv = generic_pullback(Δ)
            return NoTangent(), NoTangent(), dq, dk, dv, NoTangent(), NoTangent(),
                   NoTangent(), NoTangent()
        end
        return copyto!(out, y), fallback_pullback
    end
    cuDNN.attention!(out, q, k, v; scale, causal, stats, seq_len_q, seq_len_kv)
    function cudnn_sdpa_pullback(Δ)
        dO = unthunk(Δ)
        dO isa DenseCuArray{eltype(q),4} || (dO = CuArray{eltype(q)}(dO))
        dq, dk, dv = cuDNN.attention_backward(dO, q, k, v, out, stats;
                                              scale, causal, seq_len_q, seq_len_kv)
        return NoTangent(), NoTangent(), dq, dk, dv, NoTangent(), NoTangent(),
               NoTangent(), NoTangent()
    end
    return out, cudnn_sdpa_pullback
end
