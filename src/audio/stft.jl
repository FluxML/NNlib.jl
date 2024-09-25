"""
    hamming_window(
        window_length::Int, ::Type{T} = Float32; periodic::Bool = true,
        α::T = T(0.54), β::T = T(0.46),
    ) where T <: Real

Hamming window function
(ref: [Window function § Hann and Hamming windows - Wikipedia](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)).
Generalized version of `hann_window`.

``w[n] = \\alpha - \\beta \\cos(\\frac{2 \\pi n}{N - 1})``

Where ``N`` is the window length.

```julia-repl
julia> lineplot(hamming_window(100); width=30, height=10)
     ┌──────────────────────────────┐
   1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠚⠉⠉⠉⠢⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠁⠀⠀⠀⠀⠀⠈⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⡀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⢰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⣠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⡀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡄⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⡰⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀│
     │⠀⠀⠀⢀⠴⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢄⠀⠀⠀│
     │⠀⢀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⣀⠀│
   0 │⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉│
     └──────────────────────────────┘
     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀100⠀
```

# Arguments:

- `window_length::Int`: Size of the window.
- `::Type{T}`: Elemet type of the window.

# Keyword Arguments:

- `periodic::Bool`: If `true` (default), returns a window to be used as
    periodic function. If `false`, return a symmetric window.

    Following always holds:

```jldoctest
julia> N = 256;

julia> hamming_window(N; periodic=true) ≈ hamming_window(N + 1; periodic=false)[1:end - 1]
true
```
- `α::Real`: Coefficient α in the equation above.
- `β::Real`: Coefficient β in the equation above.

# Returns:

Vector of length `window_length` and eltype `T`.
"""
function hamming_window(
    window_length::Int, ::Type{T} = Float32; periodic::Bool = true,
    α::T = T(0.54), β::T = T(0.46),
) where T <: Real
    window_length < 1 && throw(ArgumentError(
        "`window_length` must be > 0, instead: `$window_length`."))

    n::T = ifelse(periodic, window_length, window_length - 1)
    scale = T(2) * π / n
    return [α - β * cos(scale * T(k)) for k in 0:(window_length - 1)]
end

"""
    hann_window(
        window_length::Int, ::Type{T} = Float32; periodic::Bool = true,
    ) where T <: Real

Hann window function
(ref: [Window function § Hann and Hamming windows - Wikipedia](https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows)).

``w[n] = \\frac{1}{2}[1 - \\cos(\\frac{2 \\pi n}{N - 1})]``

Where ``N`` is the window length.

```julia-repl
julia> lineplot(hann_window(100); width=30, height=10)
     ┌──────────────────────────────┐
   1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠚⠉⠉⠉⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡔⠁⠀⠀⠀⠀⠀⠘⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠞⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⢀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢣⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢦⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⢀⠞⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢆⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⢀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢇⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢦⠀⠀⠀⠀│
     │⠀⠀⠀⢠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠣⡀⠀⠀│
   0 │⣀⣀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢤⣀│
     └──────────────────────────────┘
     ⠀0⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀100⠀
```

# Arguments:

- `window_length::Int`: Size of the window.
- `::Type{T}`: Elemet type of the window.

# Keyword Arguments:

- `periodic::Bool`: If `true` (default), returns a window to be used as
    periodic function. If `false`, return a symmetric window.

    Following always holds:

```jldoctest
julia> N = 256;

julia> hann_window(N; periodic=true) ≈ hann_window(N + 1; periodic=false)[1:end - 1]
true

julia> hann_window(N) ≈ hamming_window(N; α=0.5f0, β=0.5f0)
true
```

# Returns:

Vector of length `window_length` and eltype `T`.
"""
function hann_window(
    window_length::Int, ::Type{T} = Float32; periodic::Bool = true,
) where T <: Real
    hamming_window(window_length, T; periodic, α=T(0.5), β=T(0.5))
end

"""
    stft(x;
        n_fft::Int, hop_length::Int = n_fft ÷ 4, window = nothing,
        center::Bool = true, normalized::Bool = false,
    )

Short-time Fourier transform (STFT).

The STFT computes the Fourier transform of short overlapping windows of the input,
giving frequency components of the signal as they change over time.

``Y[\\omega, m] = \\sum_{k = 0}^{N - 1} \\text{window}[k] \\text{input}[m \\times \\text{hop length} + k] \\exp(-j \\frac{2 \\pi \\omega k}{\\text{n fft}})``

where ``N`` is the window length,
``\\omega`` is the frequency ``0 \\le \\omega < \\text{n fft}``
and ``m`` is the index of the sliding window.

# Arguments:

- `x`: Input, must be either a 1D time sequence (`(L,)` shape)
    or a 2D batch of time sequence (`(L, B)` shape).

# Keyword Arguments:

- `n_fft::Int`: Size of Fourier transform.
- `hop_length::Int`: Distance between neighboring sliding window frames.
- `window`: Optional window function to apply.
    Must be 1D vector `0 < length(window) ≤ n_fft`.
    If window is shorter than `n_fft`, it is padded with zeros on both sides.
    If `nothing` (default), then no window is applied.
- `center::Bool`: Whether to pad input on both sides so that ``t``-th frame
    is centered at time ``t \\times \\text{hop length}``.
    Padding is done with `pad_reflect` function.
- `normalized::Bool`: Whether to return normalized STFT,
    i.e. multiplied with ``\\text{n fft}^{-0.5}``.

# Returns:

Complex array of shape `(n_fft, n_frames, B)`,
where `B` is the optional batch dimension.
"""
function stft end

"""
    istft(y;
        n_fft::Int, hop_length::Int = n_fft ÷ 4, window = nothing,
        center::Bool = true, normalized::Bool = false,
        return_complex::Bool = false,
        original_length::Union{Nothing, Int} = nothing,
    )

Inverse Short-time Fourier Transform.

Return the least squares estimation of the original signal

# Arguments:

- `y`: Input complex array in the `(n_fft, n_frames, B)` shape.
    Where `B` is the optional batch dimension.

# Keyword Arguments:

- `n_fft::Int`: Size of Fourier transform.
- `hop_length::Int`: Distance between neighboring sliding window frames.
- `window`: Window function that was applied to the input of `stft`.
    If `nothing` (default), then no window was applied.
- `center::Bool`: Whether input to `stft` was padded on both sides
    so that ``t``-th frame is centered at time ``t \\times \\text{hop length}``.
    Padding is done with `pad_reflect` function.
- `normalized::Bool`: Whether input to `stft` was normalized.
- `return_complex::Bool`: Whether the output should be complex,
    or if the input should be assumed to derive from a real signal and window.
- `original_length::Union{Nothing, Int}`: Optional size of the first dimension
    of the input to `stft`. Helps restoring the exact `stft` input size.
    Otherwise, the array might be a bit shorter.
"""
function istft end
