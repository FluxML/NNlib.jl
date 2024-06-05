"""

Create a spectrogram or a batch of spectrograms from a raw audio signal.

# Arguments

- `pad::Int`:
    Then amount of padding to apply on both sides.
    Default is `0`.
- `window_normalized::Bool`:
    Whether to normalize the waveform by the window’s L2 energy.
    Default is `false`.
- `power::Real`:
    Exponent for the magnitude spectrogram (must be ≥ 0)
    e.g., `1` for magnitude, `2` for power, etc.
    If `0`, complex spectrum is returned instead.

See [`stft`](@ref) for other arguments.

# Returns

Spectrogram in the shape `(T, F, B)`, where
`T` is the number of window hops and `F = n_fft ÷ 2 + 1`.

# Example

```julia
julia> waveform, sampling_rate = load("test.flac");

julia> spec = spectrogram(waveform;
    n_fft=1024, hop_length=128, window=hann_window(1024));

julia> spec_db = NNlib.power_to_db(spec);

julia> Makie.heatmap(spec_db[:, :, 1])
```
"""
function spectrogram(waveform;
    pad::Int = 0, n_fft::Int, hop_length::Int, window,
    center::Bool = true, power::Real = 2.0,
    normalized::Bool = false, window_normalized::Bool = false,
)
    pad > 0 && (waveform = pad_zeros(waveform, pad; dims=1);)

    # Pack batch dimensions.
    sz = size(waveform)
    spec_ = stft(reshape(waveform, (sz[1], :));
        n_fft, hop_length, window, center, normalized)
    # Unpack batch dimensions.
    spec = reshape(spec_, (size(spec_)[1:2]..., sz[2:end]...))
    window_normalized && (spec .*= inv(norm(window));)

    if power > 0
        p = real(eltype(spec)(power))
        spec = abs.(spec).^p
    end
    return spec
end

"""
    power_to_db(s; ref::Real = 1f0, amin::Real = 1f-10, top_db::Real = 80f0)

Convert a power spectrogram (amplitude squared) to decibel (dB) units.

# Arguments

- `s`: Input power.
- `ref`: Scalar w.r.t. which the input is scaled. Default is `1`.
- `amin`: Minimum threshold for `s`. Default is `1f-10`.
- `top_db`: Threshold the output at `top_db` below the peak:
    `max.(s_db, maximum(s_db) - top_db)`. Default is `80`.

# Returns

`s_db ~= 10 * log10(s) - 10 * log10(ref)`
"""
function power_to_db(s; ref::Real = 1f0, amin::Real = 1f-10, top_db::Real = 80f0)
    log_spec = 10f0 .* (log10.(max.(amin, s)) .- log10.(max.(amin, ref)))
    return max.(log_spec, maximum(log_spec) - top_db)
end

"""
    db_to_power(s_db; ref::Real = 1f0)

Inverse of [`power_to_db`](@ref).
"""
function db_to_power(s_db; ref::Real = 1f0)
    return ref .* 10f0.^(s_db .* 0.1f0)
end
