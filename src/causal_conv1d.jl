"""
    NNlib.causal_conv1d(x, w; bias=nothing, state=nothing, lengths=nothing) -> (y, new_state)

The causal depthwise convolution of the sequences `x`, of size
`(channels, seq_len, batch_size)`, with the filters `w`, of size `(kernel_size, channels)`:
each channel has its own filter, and the output at a position depends on the inputs up to
that position only,

    y[c, t, b] = bias[c] + sum(w[j, c] * x[c, t - kernel_size + j, b] for j in 1:kernel_size)

This is the short convolution that Mamba-2, Gated DeltaNet and LFM2 apply before their
sequence mixing (see [`NNlib.ssd_scan`](@ref) and [`NNlib.gated_delta_rule`](@ref)). As in
PyTorch the filter is not flipped: `w[kernel_size, c]` multiplies the current input. The
weight of PyTorch's `Conv1d(channels, channels, kernel_size; groups=channels)` has size
`(channels, 1, kernel_size)`, which read in Julia's column-major order is
`(kernel_size, 1, channels)`: drop the middle dimension.

The `kernel_size - 1` inputs before the first position are the columns of `state`, of size
`(channels, kernel_size - 1, batch_size)`, or zeros. The returned `new_state` holds the
last `kernel_size - 1` inputs of `state` continued by `x`. Convolving a sequence in
pieces, each starting from the `new_state` of the previous one, gives the convolution of
the whole sequence: this is how a decoder passes its tokens one at a time. On the GPU
backends, a single token outside of differentiation is one kernel.

Returns `y`, of the size of `x`, and `new_state`.

# Keywords

- `bias`: Either `nothing` or a vector of length `channels`. Default `nothing`.
- `state`: Either `nothing`, meaning zeros, or the inputs before `x`, of size
          `(channels, kernel_size - 1, batch_size)`. Default `nothing`.
- `lengths`: Either `nothing` or the number of valid positions of each sequence of a batch
          padded on the right: an integer, or an integer array with one entry for each
          batch element. `new_state` then holds the last inputs before the end of each
          sequence, and the outputs past it are not meaningful. Default `nothing`, meaning
          all the positions.

# Examples

The convolution of a prompt followed by the decoding of one token, with Mamba-2's
`kernel_size = 4` and its SiLU:

```julia
w, bias = randn(Float32, 4, 64), randn(Float32, 64)
x = randn(Float32, 64, 10, 2)                  # (channels, seq_len, batch_size)
y, state = NNlib.causal_conv1d(x, w; bias)
y = swish.(y)

x1 = randn(Float32, 64, 1, 2)                  # the next token
y1, state = NNlib.causal_conv1d(x1, w; bias, state)
```
"""
function causal_conv1d(x::AbstractArray{<:Any,3}, w::AbstractMatrix;
                       bias=nothing, state=nothing, lengths=nothing)
    check_causal_conv1d(x, w, bias, state)
    if size(x, 2) == 1 && !within_gradient(x) && scan_kernels_apply(x, w, bias, state)
        return causal_conv_step(scan_dense.((x, w, bias, state))..., lengths)
    end
    K = size(w, 1)
    xs = state === nothing ? pad_zeros(x, (K - 1, 0); dims=2) : cat(state, x; dims=2)
    y = causal_conv(xs, w)
    y = bias === nothing ? y : y .+ bias
    return y, causal_conv_state(xs, K, lengths)
end

function check_causal_conv1d(x, w, bias, state)
    K, C = size(w)
    K >= 1 || throw(ArgumentError("The kernel size has to be at least 1, got $K."))
    size(x, 1) == C || throw(DimensionMismatch(
        "The filters of size $(size(w)) are for $C channels, the input of size \
         $(size(x)) has $(size(x, 1))."))
    bias === nothing || length(bias) == C || throw(DimensionMismatch(
        "The bias has to have one entry per channel ($C), got $(length(bias))."))
    state === nothing || size(state) == (C, K - 1, size(x, 3)) || throw(DimensionMismatch(
        "The state has to have size $((C, K - 1, size(x, 3))), got $(size(state))."))
    return nothing
end

@non_differentiable check_causal_conv1d(::Any...)

# The convolution of `xs`, of size `(channels, kernel_size - 1 + seq_len, batch_size)`,
# without padding. On CUDA this is faster than cuDNN's grouped convolution, which needs
# the channels as the second dimension.
function causal_conv(xs::AbstractArray{<:Any,3}, w::AbstractMatrix)
    K = size(w, 1)
    n = size(xs, 2) - K + 1
    y = similar(xs, promote_type(eltype(xs), eltype(w)), size(xs, 1), n, size(xs, 3))
    return shifted_sum!(y, w, xs, Val(K))
end

function rrule(::typeof(causal_conv), xs::AbstractArray{<:Any,3}, w::AbstractMatrix)
    y = causal_conv(xs, w)
    function causal_conv_pullback(Δ)
        Δ = unthunk(Δ)
        K, n = size(w, 1), size(Δ, 2)
        # the input `j` taps the outputs `j - K + 1` to `j`, with the filter reversed
        dxs = shifted_sum!(similar(xs), w[end:-1:1, :],
                           pad_zeros(Δ, (K - 1, K - 1); dims=2), Val(K))
        dw = similar(w)
        for j in 1:K
            view(dw, j, :) .= vec(sum(Δ .* view(xs, :, j:(j + n - 1), :); dims=(2, 3)))
        end
        return NoTangent(), dxs, dw
    end
    return y, causal_conv_pullback
end

# `y = sum(w[j, :] .* x[:, j:(j + size(y, 2) - 1), :] for j in 1:K)`, in one broadcast.
function shifted_sum!(y, w, x, ::Val{K}) where K
    n = size(y, 2)
    taps = ntuple(j -> view(x, :, j:(j + n - 1), :), Val(K))
    y .= dot_taps.(ntuple(j -> view(w, j, :), Val(K))..., taps...)
    return y
end

@inline dot_taps(args::Vararg{Any,M}) where M = sum(ntuple(j -> args[j] * args[M ÷ 2 + j], Val(M ÷ 2)))

# The last `K - 1` inputs of `xs`, or those before the end of each sequence.
causal_conv_state(xs, K, ::Nothing) = xs[:, (end - K + 2):end, :]

function causal_conv_state(xs, K, lengths)
    K == 1 && return xs[:, 1:0, :]
    return gather(reshape(xs, size(xs, 1), :), causal_state_columns(xs, K, lengths))
end

# The columns of `reshape(xs, channels, :)` that hold the state after each sequence:
# `(K - 1, batch_size)`, on the device of `xs`.
function causal_state_columns(xs, K, lengths)
    m, b = size(xs, 2), size(xs, 3)
    len = sequence_lengths(xs, lengths, m - K + 1, b)
    return (1:(K - 1)) .+ len .+ m .* (0:(b - 1))'
end

@non_differentiable causal_state_columns(::Any...)

# The `lengths` of `b` sequences of `n` positions, as an `Int32` array of size `(1, b)` on
# the device of `x`, clamped to `0:n`.
function sequence_lengths(x::AbstractArray, lengths, n::Int, b::Int)
    out = similar(x, Int32, (1, b))
    lengths isa Number && return fill!(out, clamp(lengths, 0, n))
    length(lengths) == b || throw(ArgumentError(
        "Expected a sequence length for each of the $b sequences, got $(length(lengths))."))
    return copyto!(out, Int32.(clamp.(vec(lengths), 0, n)))
end

@non_differentiable sequence_lengths(::Any...)
