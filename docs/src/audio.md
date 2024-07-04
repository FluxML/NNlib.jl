# Reference

!!! note
    Spectral functions require importing `FFTW` package to enable them.

## Window functions

```@docs
hann_window
hamming_window
```

## Spectral

```@docs
stft
istft
NNlib.power_to_db
NNlib.db_to_power
```

## Spectrogram

```@docs
melscale_filterbanks
spectrogram
```

Example:

```@example 1
using FFTW # <- required for STFT support.
using NNlib
using FileIO
using Makie, CairoMakie
CairoMakie.activate!()

waveform, sampling_rate = load("./assets/jfk.flac")
fig = lines(reshape(waveform, :))
save("waveform.png", fig)

# Spectrogram.

n_fft = 1024
spec = spectrogram(waveform; n_fft, hop_length=n_fft ÷ 4, window=hann_window(n_fft))
fig = heatmap(transpose(NNlib.power_to_db(spec)[:, :, 1]))
save("spectrogram.png", fig)

# Mel-scale spectrogram.

n_freqs = n_fft ÷ 2 + 1
fb = melscale_filterbanks(; n_freqs, n_mels=128, sample_rate=Int(sampling_rate))
mel_spec = permutedims(spec, (2, 1, 3)) ⊠ fb # (time, n_mels)
fig = heatmap(NNlib.power_to_db(mel_spec)[:, :, 1])
save("mel-spectrogram.png", fig)
nothing # hide
```

|Waveform|Spectrogram|Mel Spectrogram|
|:---:|:---:|:---:|
|![](waveform.png)|![](spectrogram.png)|![](mel-spectrogram.png)|
