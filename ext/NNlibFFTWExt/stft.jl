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
    # Reshape `x` to (n_fft, n_frames, B). Each column in `n_frames` is
    # shifted by `hop_length`. This is done unconditionally (even when
    # `n_frames == 1`) so that the output rank does not depend on the runtime
    # value of `n_frames`, keeping `stft` type-stable (see issue #702).
    # TODO can be more efficient if we support something like torch.as_strided
    ids = [
        row + hop_length * col
        for row in 1:n_fft, col in 0:(n_frames - 1)]
    x = @inbounds x[ids, ntuple(_ -> Colon(), ndims(x) - 1)...]

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

    # Least-squares estimate (Griffin & Lim, 1984): overlap-add the windowed frames
    # and divide by the overlap-added squared window. Dividing each frame by the
    # window instead amplifies rounding errors where the window is close to zero.
    w = use_window ? window : KernelAbstractions.ones(kab, real(eltype(x)), n_fft)
    x = _overlap_add(reshape(x .* w, n_fft, n_frames, :), hop_length)
    envelope = _overlap_add(repeat(w .^ 2, 1, n_frames, 1), hop_length)
    # Samples outside the support of every window cannot be recovered and stay `0`.
    x = x ./ ifelse.(envelope .> floatmin(eltype(envelope)), envelope, one(eltype(envelope)))

    # (len, batch) -> (len,) for unbatched input.
    x = reshape(x, size(x, 1), Base.tail(Base.tail(size(y)))...)
    nd = ntuple(_ -> Colon(), ndims(x) - 1)

    # Trim padding.
    left = center ? (n_fft ÷ 2 + 1) : 1
    right = if isnothing(original_length)
        center ? (size(x, 1) - n_fft ÷ 2) : size(x, 1)
    else
        left + original_length - 1
    end
    x = x[left:right, nd...]
    return x
end

# Sum the frames `(n_fft, n_frames, batch)`, `hop_length` apart, into `(len, batch)`.
function _overlap_add(x::AbstractArray{<:Real, 3}, hop_length::Int)
    n_fft, n_frames, batch = size(x)
    len = n_fft + hop_length * (n_frames - 1)
    y = NNlib.fold(permutedims(x, (2, 1, 3)), (len, 1, batch), (n_fft, 1, 1);
                   stride=hop_length)
    return reshape(y, len, batch)
end

# `fold` accumulates with atomics, which GPUs support only for real numbers.
_overlap_add(x::AbstractArray{<:Complex, 3}, hop_length::Int) =
    complex.(_overlap_add(real(x), hop_length), _overlap_add(imag(x), hop_length))
