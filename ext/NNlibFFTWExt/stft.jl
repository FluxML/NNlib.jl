function NNlib.stft(x;
    n_fft::Int, hop_length::Int = n_fft ÷ 4, window = nothing,
    center::Bool = true, normalized::Bool = false,
)
    kab = get_backend(x)
    use_window = !isnothing(window)

    use_window && kab != get_backend(window) && throw(ArgumentError(
        "`window` must be on the same device as stft input `x` ($kab), \
        instead: `$(get_backend(window))`."))
    use_window && !(0 < length(window) ≤ n_fft) && throw(ArgumentError(
        "Expected `0 < length(window) ≤ n_fft=$n_fft`, \
        but got `length(window)=$(length(window))`."))
    hop_length < 0 && throw(ArgumentError(
        "Expected `hop_length > 0`, but got `hop_length=$hop_length`."))

    # Pad window on both sides with `0` to `n_fft` length if needed.
    if use_window && length(window) < n_fft
        left = ((n_fft - length(window)) ÷ 2) + 1
        tmp = KernelAbstractions.zeros(kab, eltype(window), n_fft)
        tmp[left:left + length(window) - 1] .= window
        window = tmp
    end

    if center
        pad_amount = n_fft ÷ 2
        x = pad_reflect(x, pad_amount; dims=1)
    end

    n = size(x, 1)
    (0 < n_fft ≤ n) || throw(ArgumentError(
        "Expected `0 < n_fft ≤ size(x, 1)=$n`, but got `n_fft=$n_fft`."))

    n_frames = 1 + (n - n_fft) ÷ hop_length

    # time2col.
    # Reshape `x` to (n_fft, n_frames, B) if needed.
    # Each row in `n_frames` is shifted by `hop_length`.
    if n_frames > 1
        # TODO can be more efficient if we support something like torch.as_strided
        ids = [
            row + hop_length * col
            for row in 1:n_fft, col in 0:(n_frames - 1)]
        x = @inbounds x[ids, ntuple(_ -> Colon(), ndims(x) - 1)...]
    end

    region = 1
    use_window && (x = x .* window;)
    y = eltype(x) <: Complex ? fft(x, region) : rfft(x, region)

    normalized && (y = y .* eltype(y)(n_fft^-0.5);)
    return y
end

function NNlib.istft(y;
    n_fft::Int, hop_length::Int = n_fft ÷ 4, window = nothing,
    center::Bool = true, normalized::Bool = false,
    return_complex::Bool = false,
    original_length::Union{Nothing, Int} = nothing,
)
    kab = get_backend(y)
    use_window = !isnothing(window)

    use_window && kab != get_backend(window) && throw(ArgumentError(
        "`window` must be on the same device as istft input `y` ($kab), \
        instead: `$(get_backend(window))`."))
    use_window && !(0 < length(window) ≤ n_fft) && throw(ArgumentError(
        "Expected `0 < length(window) ≤ n_fft=$n_fft`, \
        but got `length(window)=$(length(window))`."))
    hop_length < 0 && throw(ArgumentError(
        "Expected `hop_length > 0`, but got `hop_length=$hop_length`."))

    # TODO check `y` eltype is complex

    n_frames = size(y, 2)

    # Pad window on both sides with `0` to `n_fft` length if needed.
    if use_window && length(window) < n_fft
        left = ((n_fft - length(window)) ÷ 2) + 1
        tmp = KernelAbstractions.zeros(kab, eltype(window), n_fft)
        tmp[left:left + length(window) - 1] .= window
        window = tmp
    end

    # Denormalize.
    normalized && (y = y .* eltype(y)(n_fft^0.5);)

    region = 1
    x = return_complex ? ifft(y, region) : irfft(y, n_fft, region)

    # De-apply window.
    use_window && (x = x ./ window;)

    # col2time.
    expected_output_len = n_fft + hop_length * (n_frames - 1)

    ids = Vector{Int}(undef, expected_output_len)
    in_idx, out_idx = 0, 0
    prev_e, v = 0, 0

    for col in 0:(n_frames - 1)
        for row in 1:n_fft
            in_idx += 1
            v = row + hop_length * col
            v > prev_e || continue

            out_idx += 1
            ids[out_idx] = in_idx
        end
        prev_e = v
    end

    # In case of batched input, reshaped it (n_fft, n_frames, batch) -> (:, batch).
    nd = ntuple(_ -> Colon(), ndims(x) - 2)
    ndims(x) == 3 && (x = reshape(x, (:, size(x, 3)));)
    x = @inbounds x[ids, nd...]

    # Trim padding.
    left = center ? (n_fft ÷ 2 + 1) : 1
    right = if isnothing(original_length)
        center ? (size(x, 1) - n_fft ÷ 2) : expected_output_len
    else
        left + original_length - 1
    end
    x = x[left:right, nd...]
    return x
end
