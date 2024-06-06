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
using UnicodePlots

waveform, sampling_rate = load("./assets/jfk.flac")
lineplot(reshape(waveform, :); width=50, color=:white)
```

```@example 1
n_fft = 256
spec = spectrogram(waveform; n_fft, hop_length=n_fft ÷ 4, window=hann_window(n_fft))
heatmap(NNlib.power_to_db(spec)[:, :, 1]; width=80, height=30)
```
