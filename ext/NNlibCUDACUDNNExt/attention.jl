import NNlib: _scaled_dot_product_attention, _is_fused_attention
using ChainRulesCore: ChainRulesCore, HasReverseMode, NoTangent, RuleConfig,
                       rrule_via_ad, unthunk
using KernelAbstractions: @kernel, @index, get_backend

const CUDNNSDPAFloat = Union{Float16,BFloat16}

function cudnn_sdpa_lengths(q, k, q_lengths, kv_lengths)
    q_lengths === nothing && kv_lengths === nothing && return nothing, nothing
    q_lengths === nothing && (q_lengths = full_lengths(q))
    kv_lengths === nothing && (kv_lengths = full_lengths(k))
    return q_lengths, kv_lengths
end
full_lengths(x) = fill!(similar(x, Int32, (1, 1, 1, size(x, 4))), size(x, 3))

cudnn_sdpa_equal(x::Integer, y::Integer) = x == y
cudnn_sdpa_equal(x, y) = all(x .== y)

function cudnn_sdpa_options(q, k; is_causal, window, q_offset, q_lengths, kv_lengths, segments)
    causal, offset = NNlib.causal_offsets(q, k, is_causal, q_offset, q_lengths, kv_lengths)
    @static if CUDNN_6_5
        if causal && window === nothing &&
                ((is_causal === :bottom_right &&
                  (segments === nothing ? size(q, 3) == 1 : segments.maxlen_q == 1)) ||
                 (offset isa Integer && offset + 1 >= size(k, 3)))
            causal = false
        elseif is_causal === :bottom_right
            causal = offset isa Integer && iszero(offset) ? true : :bottom_right
        elseif !(offset isa Integer && iszero(offset))
            if cudnn_sdpa_equal(offset, 0)
                causal = is_causal
            elseif causal && cudnn_sdpa_equal(offset, NNlib.bottom_right_offsets(q, k, q_lengths, kv_lengths))
                causal = :bottom_right
            elseif causal || window !== nothing
                return nothing
            end
        end
        right_bound = window !== nothing && causal === false ? window - 1 : nothing
        bounds = (; causal, window, right_bound)
        if segments !== nothing
            NNlib.fused_attention_segments(segments) || return nothing
            segments.maxlen_q > 0 && segments.maxlen_kv > 0 || return nothing
            # Ragged offsets delimit physical storage; shorter valid lengths need NNlib's kernel.
            for (lengths, sizes) in ((q_lengths, segments.q_sizes), (kv_lengths, segments.kv_sizes))
                pointer(lengths) == pointer(sizes) || cudnn_sdpa_equal(vec(lengths), sizes) || return nothing
            end
            cu_seq_len_q = vcat(CUDA.zeros(Int32, 1), segments.q_ends)
            cu_seq_len_kv = vcat(CUDA.zeros(Int32, 1), segments.kv_ends)
            return (; bounds..., cu_seq_len_q, cu_seq_len_kv,
                    max_seq_len_q=segments.maxlen_q, max_seq_len_kv=segments.maxlen_kv)
        end
        seq_len_q, seq_len_kv = cudnn_sdpa_lengths(q, k, q_lengths, kv_lengths)
        if size(q, 3) > size(k, 3) && (window !== nothing || causal === :bottom_right)
            if seq_len_q === nothing
                seq_len_q, seq_len_kv = full_lengths(q), full_lengths(k)
            end
        end
        return (; bounds..., seq_len_q, seq_len_kv)
    else
        segments === nothing && window === nothing || return nothing
        !causal || (offset isa Integer && iszero(offset)) || return nothing
        seq_len_q, seq_len_kv = cudnn_sdpa_lengths(q, k, q_lengths, kv_lengths)
        return (; causal, seq_len_q, seq_len_kv)
    end
end

cudnn_sdpa_array(x, segments::Nothing) = x
cudnn_sdpa_array(x, segments) = reshape(x, size(x, 1), size(x, 2), size(x, 3))

function cudnn_sdpa_output(q, k, v, bias, fdrop, mask, options, segments)
    options === nothing && return nothing
    bias === nothing && fdrop === identity && mask === nothing || return nothing
    size(v, 1) == size(q, 1) || return nothing
    (isempty(q) || isempty(k) || isempty(v)) && return nothing
    out = similar(q)
    arrays = map(x -> cudnn_sdpa_array(x, segments), (out, q, k, v))
    cuDNN.attention_supported(arrays...; options...) && return out
    CUDA.unsafe_free!(out)
    return nothing
end

ChainRulesCore.@non_differentiable cudnn_sdpa_options(::Any...)
ChainRulesCore.@non_differentiable cudnn_sdpa_output(::Any...)

function _scaled_dot_product_attention(q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                                       v::DenseCuArray{T,4}, bias;
                                       fdrop, mask, scale, is_causal, window,
                                       q_offset, q_lengths, kv_lengths, segments) where {T<:CUDNNSDPAFloat}
    kws = (; is_causal, window, q_offset, q_lengths, kv_lengths, segments)
    options = cudnn_sdpa_options(q, k; kws...)
    out = cudnn_sdpa_output(q, k, v, bias, fdrop, mask, options, segments)
    out === nothing && return NNlib.generic_attention(q, k, v, bias; fdrop, mask, scale, kws...)
    return cudnn_sdpa!(out, q, k, v, something(scale, inv(sqrt(size(q, 1)))), options, kws)
end

function _is_fused_attention(q::DenseCuArray{T,4}, k::DenseCuArray{T,4},
                             v::DenseCuArray{T,4}, bias;
                             fdrop, mask, scale, is_causal, window,
                             q_offset, q_lengths, kv_lengths, segments) where {T<:CUDNNSDPAFloat}
    options = cudnn_sdpa_options(q, k; is_causal, window, q_offset, q_lengths, kv_lengths, segments)
    out = cudnn_sdpa_output(q, k, v, bias, fdrop, mask, options, segments)
    out === nothing && return NNlib.fused_attention_applies(q, k, v, bias, fdrop, mask) &&
                              NNlib.fused_attention_segments(segments)
    CUDA.unsafe_free!(out)
    return true
end

# cuDNN leaves empty rows undefined. NNlib returns NaN there and zero in padding;
# backward uses zero rows and infinite log-sum-exp statistics instead.
@kernel function cudnn_sdpa_rows_kernel!(out, stats, q_lengths, kv_lengths, q_ends, kv_ends,
                                         q_offset, window, causal, kv_size, empty_value)
    i, s = @index(Global, NTuple)
    first, nq = NNlib.attention_span(q_ends, s, size(out, 3))
    _, nk = NNlib.attention_span(kv_ends, s, kv_size)
    qlen = NNlib.attention_len(q_lengths, s, nq)
    kvlen = NNlib.attention_len(kv_lengths, s, nk)
    b = NNlib.attention_batch(q_ends, s)
    off = NNlib.attention_offset(q_offset, s)
    lo = window === nothing ? 1 : max(1, i + off - window + 1)
    hi = causal ? min(kvlen, i + off) : kvlen
    live = i <= qlen && lo <= hi
    if i <= nq && !live
        value = i > qlen ? zero(eltype(out)) : eltype(out)(empty_value)
        for h in 1:size(out, 2), d in 1:size(out, 1)
            @inbounds out[d, h, first + i, b] = value
        end
    end
    if stats !== nothing && !live
        for h in 1:size(out, 2)
            @inbounds stats[1, i, h, s] = Inf32
        end
    end
end

function cudnn_sdpa_needs_rows(kws)
    (; is_causal, window, q_offset, q_lengths, kv_lengths, segments) = kws
    return q_lengths !== nothing || kv_lengths !== nothing || window !== nothing ||
           segments !== nothing || is_causal === :bottom_right ||
           !(q_offset isa Integer && q_offset >= 0)
end

function cudnn_sdpa_rows!(out, stats, q, k, kws, empty_value=NaN32)
    cudnn_sdpa_needs_rows(kws) || return out
    (; is_causal, window, q_offset, q_lengths, kv_lengths, segments) = kws
    causal, offset = NNlib.causal_offsets(q, k, is_causal, q_offset, q_lengths, kv_lengths)
    causal, offset, band = NNlib.attention_band(causal, offset, window)
    seqs = NNlib.attention_seqs(q, k, segments)
    cudnn_sdpa_rows_kernel!(get_backend(out), 256)(out, stats, NNlib.attention_vec(q_lengths),
        NNlib.attention_vec(kv_lengths), seqs.q_ends, seqs.kv_ends, NNlib.attention_vec_offsets(offset),
        band, causal, size(k, 3), empty_value; ndrange=(seqs.maxlen_q, seqs.n))
    return out
end

function cudnn_sdpa!(out, q, k, v, scale, options, kws)
    arrays = map(x -> cudnn_sdpa_array(x, kws.segments), (out, q, k, v))
    cuDNN.attention!(arrays...; scale, options...)
    return cudnn_sdpa_rows!(out, nothing, q, k, kws)
end

function ChainRulesCore.rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(cudnn_sdpa!),
                              out, q, k, v, scale, options, kws)
    seqs = NNlib.attention_seqs(q, k, kws.segments)
    stats = similar(q, Float32, (1, seqs.maxlen_q, size(q, 2), seqs.n))
    arrays = map(x -> cudnn_sdpa_array(x, kws.segments), (out, q, k, v))
    o, cq, ck, cv = arrays
    if !cuDNN.attention_backward_supported(cq, ck, cv, o, cq, ck, cv, o, stats; options...)
        CUDA.unsafe_free!(stats)
        fallback = (q, k, v) -> NNlib.generic_attention(q, k, v, nothing;
                                                      fdrop=identity, mask=nothing, scale, kws...)
        y, generic_pullback = rrule_via_ad(cfg, fallback, q, k, v)
        function fallback_pullback(Δ)
            _, dq, dk, dv = generic_pullback(Δ)
            return NoTangent(), NoTangent(), dq, dk, dv, NoTangent(), NoTangent(), NoTangent()
        end
        return copyto!(out, y), fallback_pullback
    end
    cuDNN.attention!(arrays...; scale, stats, options...)
    cudnn_sdpa_rows!(out, stats, q, k, kws)
    function cudnn_sdpa_pullback(Δ)
        dO = unthunk(Δ)
        dO isa DenseCuArray{eltype(q),4} || (dO = CuArray{eltype(q)}(dO))
        masked = cudnn_sdpa_needs_rows(kws)
        safe_out = masked ? cudnn_sdpa_rows!(copy(out), nothing, q, k, kws, 0) : out
        safe_dO = masked ? cudnn_sdpa_rows!(copy(dO), nothing, q, k, kws, 0) : dO
        dq, dk, dv = map(x -> masked ? CUDA.zeros(eltype(x), size(x)) : similar(x), (q, k, v))
        grads = map(x -> cudnn_sdpa_array(x, kws.segments), (dq, dk, dv))
        cuDNN.attention_backward!(grads..., cudnn_sdpa_array(safe_dO, kws.segments), cq, ck, cv,
                                  cudnn_sdpa_array(safe_out, kws.segments), stats; scale, options...)
        cudnn_sdpa_rows!(dq, nothing, q, k, kws, 0)
        # Gradients outside the valid keys are undefined in cuDNN.
        for grad in (dk, dv)
            cudnn_sdpa_rows!(grad, nothing, k, q,
                (; is_causal=false, window=nothing, q_offset=0, q_lengths=kws.kv_lengths,
                   kv_lengths=nothing, segments=cudnn_sdpa_reverse_segments(kws.segments)), 0)
        end
        return NoTangent(), NoTangent(), dq, dk, dv, NoTangent(), NoTangent(), NoTangent()
    end
    return out, cudnn_sdpa_pullback
end

cudnn_sdpa_reverse_segments(::Nothing) = nothing
cudnn_sdpa_reverse_segments(s) = NNlib.AttentionSegments(s.kv_sizes, s.kv_ends, s.q_sizes, s.q_ends,
                                                        s.maxlen_kv, s.maxlen_q)
