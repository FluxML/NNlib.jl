import NNlib: scaled_dot_product_attention
using ChainRulesCore: ChainRulesCore, NoTangent, unthunk

const CUDNNSDPAFloat = Union{Float16,cuDNN.BFloat16}

function generic_scaled_dot_product_attention(q, k, v, bias;
                                               fdrop, mask, scale, is_causal)
    mask = mask isa AbstractArray && !(mask isa DenseCuArray) ? cu(mask) : mask
    return first(NNlib._scaled_dot_product_attention(q, k, v, bias;
                                                      fdrop, mask, scale, is_causal))
end

function cudnn_sdpa_applicable(out, q, k, v, bias, fdrop, mask, is_causal)
    bias === nothing || return false
    fdrop === identity || return false
    mask === nothing || return false
    size(q, 1) == size(v, 1) || return false
    return cuDNN.attention_supported(out, q, k, v; causal=is_causal)
end
ChainRulesCore.@non_differentiable cudnn_sdpa_applicable(::Any...)

function scaled_dot_product_attention(q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                                      v::DenseCuArray{T,4}, bias=nothing;
                                      fdrop=identity, mask=nothing, scale=nothing,
                                      is_causal::Bool=false) where {T<:CUDNNSDPAFloat}
    out = similar(q)
    cudnn_sdpa_applicable(out, q, k, v, bias, fdrop, mask, is_causal) ||
        return generic_scaled_dot_product_attention(q, k, v, bias;
                                                    fdrop, mask, scale, is_causal)
    return cudnn_sdpa!(out, q, k, v, is_causal,
                       something(scale, inv(sqrt(size(q, 1)))))
end

function cudnn_sdpa!(out, q, k, v, causal, scale)
    cuDNN.attention!(out, q, k, v; scale, causal)
    return out
end

function ChainRulesCore.rrule(cfg::ChainRulesCore.RuleConfig{>:ChainRulesCore.HasReverseMode},
                              ::typeof(cudnn_sdpa!), out, q, k, v, causal, scale)
    stats = similar(q, Float32, (1, size(q, 2), size(q, 3), size(q, 4)))
    dq, dk, dv = similar(q), similar(k), similar(v)
    if !cuDNN.attention_backward_supported(dq, dk, dv, out, q, k, v, out, stats; causal)
        fallback = (q, k, v) -> generic_scaled_dot_product_attention(q, k, v, nothing;
                                                                       fdrop=identity,
                                                                       mask=nothing, scale,
                                                                       is_causal=causal)
        y, generic_pullback = ChainRulesCore.rrule_via_ad(cfg, fallback, q, k, v)
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
        cuDNN.attention_backward!(dq, dk, dv, dO, q, k, v, out, stats; scale, causal)
        return NoTangent(), NoTangent(), dq, dk, dv, NoTangent(), NoTangent()
    end
    return out, cudnn_sdpa_pullback
end
