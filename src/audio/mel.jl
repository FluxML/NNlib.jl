"""
    melscale_filterbanks(;
        n_freqs::Int, n_mels::Int, sample_rate::Int,
        fmin::Float32 = 0f0, fmax::Float32 = Float32(sample_rate ÷ 2))

Create triangular Mel scale filter banks
(ref: [Mel scale - Wikipedia](https://en.wikipedia.org/wiki/Mel_scale)).
Each column is a filterbank that highlights its own frequency.

# Arguments:

- `n_freqs::Int`: Number of frequencies to highlight.
- `n_mels::Int`: Number of mel filterbanks.
- `sample_rate::Int`: Sample rate of the audio waveform.
- `fmin::Float32`: Minimum frequency in Hz.
- `fmax::Float32`: Maximum frequency in Hz.

# Returns:

Filterbank matrix of shape `(n_freqs, n_mels)` where each column is a filterbank.

```jldoctest
julia> n_mels = 8;

julia> fb = melscale_filterbanks(; n_freqs=200, n_mels, sample_rate=16000);

julia> plot = lineplot(fb[:, 1]);

julia> for i in 2:n_mels
           lineplot!(plot, fb[:, i])
       end

julia> plot
     ┌────────────────────────────────────────┐
   1 │⠀⡀⢸⠀⢸⠀⠀⣧⠀⠀⢸⡄⠀⠀⠀⣷⠀⠀⠀⠀⠀⣷⠀⠀⠀⠀⠀⠀⢀⣿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⡇⢸⡆⢸⡇⠀⣿⠀⠀⡜⡇⠀⠀⢰⠋⡆⠀⠀⠀⢰⠁⡇⠀⠀⠀⠀⠀⡸⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⣿⢸⡇⡇⡇⢰⠹⡄⠀⡇⢱⠀⠀⢸⠀⢣⠀⠀⠀⡜⠀⢸⡀⠀⠀⠀⢀⠇⠀⠈⡇⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⣿⡇⡇⡇⡇⢸⠀⡇⢀⠇⠸⡀⠀⡇⠀⠸⡀⠀⢀⠇⠀⠀⢇⠀⠀⠀⡸⠀⠀⠀⠸⡄⠀⠀⠀⠀⠀⠀⠀│
     │⢠⢻⡇⡇⡇⢱⢸⠀⢇⢸⠀⠀⡇⢀⠇⠀⠀⡇⠀⢸⠀⠀⠀⠸⡀⠀⢠⠇⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀│
     │⢸⢸⡇⢱⡇⢸⡇⠀⢸⢸⠀⠀⢣⢸⠀⠀⠀⢸⠀⡇⠀⠀⠀⠀⢇⠀⡜⠀⠀⠀⠀⠀⠈⢇⠀⠀⠀⠀⠀⠀│
     │⢸⢸⡇⢸⠀⢸⡇⠀⢸⡇⠀⠀⢸⡎⠀⠀⠀⠈⣶⠁⠀⠀⠀⠀⠸⣤⠃⠀⠀⠀⠀⠀⠀⠘⡆⠀⠀⠀⠀⠀│
     │⢸⠀⡇⢸⠀⠀⡇⠀⠀⡇⠀⠀⠀⡇⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⢱⡀⠀⠀⠀⠀│
     │⢸⢸⡇⢸⠀⢸⡇⠀⢸⡇⠀⠀⢸⢇⠀⠀⠀⢀⠿⡀⠀⠀⠀⠀⢰⠛⡄⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀│
     │⢸⢸⡇⡸⡇⢸⡇⠀⢸⢸⠀⠀⡜⢸⠀⠀⠀⢸⠀⡇⠀⠀⠀⠀⡎⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠘⡆⠀⠀⠀│
     │⢸⢸⡇⡇⡇⡸⢸⠀⡎⢸⠀⠀⡇⠈⡆⠀⠀⡇⠀⢸⠀⠀⠀⢰⠁⠀⠘⡆⠀⠀⠀⠀⠀⠀⠀⠀⠸⡄⠀⠀│
     │⡇⢸⡇⡇⡇⡇⢸⠀⡇⠈⡆⢰⠁⠀⡇⠀⢰⠁⠀⠈⡆⠀⠀⡎⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀│
     │⡇⢸⢸⡇⡇⡇⠸⣰⠃⠀⡇⡸⠀⠀⢸⠀⡜⠀⠀⠀⢣⠀⢸⠁⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠈⢇⠀│
     │⡇⡇⢸⠇⢸⡇⠀⣿⠀⠀⢣⡇⠀⠀⠸⣄⠇⠀⠀⠀⠸⡀⡇⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⡄│
   0 │⣇⣇⣸⣀⣸⣀⣀⣟⣀⣀⣸⣃⣀⣀⣀⣿⣀⣀⣀⣀⣀⣿⣀⣀⣀⣀⣀⣀⣈⣇⣀⣀⣀⣀⣀⣀⣀⣀⣀⣱│
     └────────────────────────────────────────┘
     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀200⠀
```
"""
function melscale_filterbanks(;
    n_freqs::Int, n_mels::Int, sample_rate::Int,
    fmin::Float32 = 0f0, fmax::Float32 = Float32(sample_rate ÷ 2),
)
    mel_min, mel_max = _hz_to_mel(fmin), _hz_to_mel(fmax)
    mel_points = range(mel_min, mel_max; length=n_mels + 2)

    all_freqs = collect(range(0f0, Float32(sample_rate ÷ 2); length=n_freqs))
    freq_points = _mel_to_hz.(mel_points)
    filter_banks = _triangular_filterbanks(freq_points, all_freqs)

    if any(maximum(filter_banks; dims=1) .≈ 0f0)
        @warn """At least one mel filterbank has all zero values.
        The value for `n_mels=$n_mels` may be set too high.
        Or the value for `n_freqs=$n_freqs` may be set too low.
        """
    end
    return filter_banks
end

_hz_to_mel(freq::T) where T = T(2595) * log10(T(1) + (freq / T(700)))

_mel_to_hz(mel::T) where T = T(700) * (T(10)^(mel / T(2595)) - T(1))

"""
    _triangular_filterbanks(
        freq_points::Vector{Float32}, all_freqs::Vector{Float32})

Create triangular filter banks.

# Arguments:

- `freq_points::Vector{Float32}`: Filter midpoints of size `n_filters`.
- `all_freqs::Vector{Float32}`: Frequency points of size `n_freqs`.

# Returns:

Array of size `(n_freqs, n_filters)`.
"""
function _triangular_filterbanks(
    freq_points::Vector{Float32}, all_freqs::Vector{Float32},
)
    diff = @view(freq_points[2:end]) .- @view(freq_points[1:end - 1])
    slopes = transpose(reshape(freq_points, :, 1) .- reshape(all_freqs, 1, :))

    down_slopes = -(@view(slopes[:, 1:end - 2]) ./ reshape(@view(diff[1:end - 1]), 1, :))
    up_slopes = @view(slopes[:, 3:end]) ./ reshape(@view(diff[2:end]), 1, :)
    return max.(0f0, min.(down_slopes, up_slopes))
end
