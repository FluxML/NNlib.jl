function hann_window(
    window_length::Integer, ::Type{T} = Float32; periodic::Bool = true,
) where T <: Real
    window_length < 1 && throw(ArgumentError(
        "`window_length` must be > 0, instead: `$window_length`."))

    n::T = ifelse(periodic, window_length, window_length - 1)
    scale = T(2) * π / n
    [T(0.5) * (T(1) - cos(scale * T(k))) for k in 0:(window_length - 1)]
end

function hamming_window(
    window_length::Integer, ::Type{T} = Float32; periodic::Bool = true,
    α::T = T(0.54), β::T = T(0.46),
) where T <: Real
    window_length < 1 && throw(ArgumentError(
        "`window_length` must be > 0, instead: `$window_length`."))

    n::T = ifelse(periodic, window_length, window_length - 1)
    scale = T(2) * π / n
    [α - β * cos(scale * T(k)) for k in 0:(window_length - 1)]
end

function stft(x::AbstractArray{T};
    n_fft::Int, hop_length::Int = n_fft ÷ 4, window = hann_window(n_fft, T),
    center::Bool = true, normalized::Bool = false,
) where T
    # TODO:
    # - check args are valid
    # - for now input is only 1D time sequence
    # - support 2D batch of time sequences
    # - if window < n_fft - pad on both sides before applying

    kab = get_backend(x)
    _window = adapt(kab, window)

    if center
        pad_amount = n_fft ÷ 2
        x = pad_reflect(x, pad_amount; dims=1)
    end

    if length(_window) < n_fft
        left = ((n_fft - length(_window)) ÷ 2) + 1
        tmp = KernelAbstractions.zeros(kab, eltype(_window), n_fft)
        tmp[left:left + length(_window) - 1] .= _window
        _window = tmp
    end

    n = size(x, 1)
    n_frames = 1 + (n - n_fft) ÷ hop_length

    if n_frames > 1
        # TODO if we support something like torch.as_strided we can reduce memory
        ids = [
            row + hop_length * col
            for row in 1:n_fft, col in 0:(n_frames - 1)]
        x = x[ids]
    end
    display(x); println()

    # TODO dispatch for GPU/CPU implementations
    x .*= window
    region = 1
    y = rfft(x, region)
    # TODO if not onesided, use `fft` instead of `rfft`

    @show size(x)
    @show size(y)
    @show typeof(y)
    display(y); println()
    return
end
