# Reference

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
spectrogram
```

Example:

```@example 1
using NNlib
using FileIO
using Makie, CairoMakie
CairoMakie.activate!()

waveform, sampling_rate = load("./assets/jfk.flac")
fig = lines(reshape(waveform, :))
save("waveform.png", fig)
nothing # hide
```

![](waveform.png)

```@example 1
n_fft = 1024
spec = spectrogram(waveform; n_fft, hop_length=n_fft ÷ 4, window=hann_window(n_fft))
fig = heatmap(transpose(NNlib.power_to_db(spec)[:, :, 1]))
save("spectrogram.png", fig)
nothing # hide
```

![](spectrogram.png)
