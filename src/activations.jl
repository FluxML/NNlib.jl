## Activation functions
#
# Some of activation functions have its wrapper function for GPU in NNlibCUDAExt.jl.
# https://github.com/JuliaGPU/CuArrays.jl/issues/614

ACTIVATIONS = [
    :σ, :hardσ, :hardtanh, :relu,
    :leakyrelu, :relu6, :rrelu, :elu, :gelu_tanh, :gelu_sigmoid, :gelu_erf, :swish, :hardswish, :selu,
    :celu, :softplus, :softsign, :logσ, :logcosh,
    :mish, :tanhshrink, :softshrink, :trelu, :lisht,
    :tanh_fast, :sigmoid_fast,
]

# of type float (to allow for integer inputs)
oftf(x, y) = oftype(float(x), y)

# oftype contains control flow on 1.10+, which can lead to type instabilities under AD 
function rrule(::typeof(oftf), x, y)
    proj_y = ChainRulesCore.ProjectTo(y)
    oftf_pullback(Δ) = (NoTangent(), NoTangent(), proj_y(Δ))
    return oftf(x, y), oftf_pullback
end

## `exp`, `expm1` and `log1p` that the compiler can vectorize
#
# A broadcast of `exp`, `expm1` or `log1p` from Base is never compiled to SIMD code,
# because they branch on their argument and `exp` is marked `@noinline`. The versions
# below are inlined and free of branches, so that a loop calling them is vectorized. They
# exist for `Float32` and `Float64` only, any other number type falls back to Base.
# They are accurate to less than 1 ulp, `_expm1` for `x ≤ 0` only (1.6 ulp above), and
# `_exp` covers the whole range of `exp`, subnormal results included, unlike `@fastmath exp`.

_exp(x) = exp(x)
_expm1(x) = expm1(x)

# As in Base: `x = N log(2) + r` with `abs(r) ≤ log(2)/2`, then `exp(x) = 2^N exp(r)`
# with a polynomial for `exp(r)`. The coefficients are those of Base.
@inline function _exp(x::Float32)
    x = clamp(x, -104f0, 89f0)  # beyond these the result is 0 or Inf anyway
    N = round(x * 1.442695f0)
    r = muladd(N, -0.69314575f0, x)  # log(2) is split into high and low bits
    r = muladd(N, -1.4286068f-6, r)
    p = evalpoly(r, (1f0, 1f0, 0.5f0, 0.16666667f0, 0.041666217f0,
                     0.008333249f0, 0.001394858f0, 0.00019924171f0))
    _mul_pow2(p, unsafe_trunc(Int32, N))
end

# Base reads `exp(r)` from a table for Float64, which cannot be vectorized. This is instead
# the algorithm of FreeBSD's msun (`e_exp.c`): `exp(r) = 1 + r + r c / (2 - c)` with
# `c = r - r^2 P(r^2)`.
@inline function _exp(x::Float64)
    N, e = _expm1_reduced(clamp(x, -746.0, 710.0))
    _mul_pow2(1 + e, unsafe_trunc(Int64, N))
end

# With the `N` and `r` of `_exp`, `expm1(x) = 2^N expm1(r) + (2^N - 1)`. Nothing cancels
# for small `x`, where `N = 0`, unlike in `exp(x) - 1`.
@inline function _expm1(x::Float32)
    N, e = _expm1_reduced(clamp(x, -18f0, 89f0))  # beyond these the result is -1 or Inf
    _expm1_scale(e, unsafe_trunc(Int32, N))
end
@inline function _expm1(x::Float64)
    N, e = _expm1_reduced(clamp(x, -38.0, 710.0))
    _expm1_scale(e, unsafe_trunc(Int64, N))
end

# `N` and `expm1(r)`, for `x = N log(2) + r` with `abs(r) ≤ log(2)/2`.
@inline function _expm1_reduced(x::Float32)
    N = round(x * 1.442695f0)
    r = muladd(N, -0.69314575f0, x)
    r = muladd(N, -1.4286068f-6, r)
    q = evalpoly(r, (0.5f0, 0.16666667f0, 0.041666217f0, 0.008333249f0, 0.001394858f0,
                     0.00019924171f0))
    N, muladd(r, r * q, r)
end
@inline function _expm1_reduced(x::Float64)
    N = round(x * 1.4426950408889634)
    hi = muladd(N, -0.6931471803691238, x)
    lo = N * 1.9082149292705877e-10
    r = hi - lo
    r2 = r * r
    c = r - r2 * evalpoly(r2, (0.16666666666666602, -0.0027777777777015593,
                               6.613756321437934e-5, -1.6533902205465252e-6,
                               4.1381367970572385e-8))
    N, hi - (lo - r * c / (2 - c))
end

# `2^N e + (2^N - 1)`. For the largest `N` the power of two overflows while the result
# need not, but there `2^N (1 + e)` is just as good.
@inline function _expm1_scale(e::Float32, N::Int32)
    p = reinterpret(Float32, (N + Int32(127)) << Int32(23))
    ifelse(N == Int32(128), _mul_pow2(1 + e, N), muladd(p, e, p - 1))
end
@inline function _expm1_scale(e::Float64, N::Int64)
    p = reinterpret(Float64, (N + 1023) << 52)
    ifelse(N == 1024, _mul_pow2(1 + e, N), muladd(p, e, p - 1))
end

# `p * 2^N`, with the power of two applied in two halves: neither of them can overflow or
# underflow, so the product is rounded only once, also when it is subnormal.
@inline function _mul_pow2(p::Float32, N::Int32)
    M = N >> 1
    p * reinterpret(Float32, (M + Int32(127)) << Int32(23)) *
        reinterpret(Float32, (N - M + Int32(127)) << Int32(23))
end
@inline function _mul_pow2(p::Float64, N::Int64)
    M = N >> 1
    p * reinterpret(Float64, (M + 1023) << 52) * reinterpret(Float64, (N - M + 1023) << 52)
end

# `log1p(t)` for `0 ≤ t ≤ 1` only, which is all that `softplus` needs. This is the
# algorithm of FreeBSD's msun (`s_log1p.c`) restricted to that interval:
# `1 + t = 2^k (1 + f)` with `k` either 0 or 1, and
# `log1p(f) = f - f^2/2 + s (f^2/2 + R(s^2))` where `s = f / (2 + f)`.
@inline function _log1p_unit(t::T) where {T<:Union{Float32,Float64}}
    halve = t ≥ T(0.41421356237309503)  # 1 + t ≥ √2
    u = 1 + t
    f = ifelse(halve, u / 2 - 1, t)
    c = ifelse(halve, (t - (u - 1)) / u, zero(T))  # makes up for the rounding of `1 + t`
    k = ifelse(halve, one(T), zero(T))
    s = f / (2 + f)
    z = s * s
    h = f * f / 2
    R = z * _log1p_poly(z)
    log2hi, log2lo = _log2_hi_lo(T)
    k * log2hi - ((h - (s * (h + R) + (k * log2lo + c))) - f)
end

_log1p_poly(z::Float32) = evalpoly(z, (0.6666667f0, 0.4f0, 0.2857143f0, 0.22222199f0,
                                       0.18183573f0, 0.15313838f0, 0.14798199f0))
_log1p_poly(z::Float64) = evalpoly(z, (0.6666666666666735, 0.3999999999940942,
                                       0.2857142874366239, 0.22222198432149784,
                                       0.1818357216161805, 0.15313837699209373,
                                       0.14798198605116586))

_log2_hi_lo(::Type{Float32}) = (0.6931381f0, 9.058001f-6)
_log2_hi_lo(::Type{Float64}) = (0.6931471803691238, 1.9082149292705877e-10)

@scalar_rule _exp(x) Ω
@scalar_rule _expm1(x) Ω + 1
@scalar_rule _log1p_unit(x) inv(1 + x)

"""
    σ(x) = 1 / (1 + exp(-x))

Classic [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation
function.
Unicode `σ` can be entered as `\\sigma` then tab, in many editors.
The ascii name `sigmoid` is also exported.

See also [`sigmoid_fast`](@ref).

```julia-repl
julia> using UnicodePlots

julia> lineplot(-5, 5, sigmoid, height=7)
          ┌────────────────────────────────────────┐     
        1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⣀⡠⠤⠖⠒⠒⠋⠉⠉⠉⠉⠉⠉│ σ(x)
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢀⡠⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⣀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
   f(x)   │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⡏⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡔⠋⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠊⠁⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
        0 │⣀⣀⣀⣀⣀⣀⣀⠤⠤⠤⠒⠊⠉⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
          └────────────────────────────────────────┘     
          ⠀-5⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀5⠀     
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     

julia> sigmoid === σ
true
```
"""
@inline function σ(x)
    # Not `-abs(x)`, whose derivative at `-0.0` is that of the other branch.
    t = _exp(ifelse(x ≥ 0, -x, x))
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

# Complex inputs (e.g. complex-valued networks): σ is holomorphic, so we compute the
# same function via the overflow-safe rearrangement. `1/(1+exp(-z))` and
# `exp(z)/(1+exp(z))` are equal everywhere; we branch on `real(z)` to pick the one
# whose `exp` cannot overflow (overflow depends only on the real part).
function σ(z::Complex)
    real(z) ≥ 0 ? inv(1 + exp(-z)) : (t = exp(z); t / (1 + t))
end

const sigmoid = σ

"""
    hardσ(x) = max(0, min(1, (x + 3) / 6))

Piecewise linear approximation of [`sigmoid`](@ref).

```julia-repl
julia> lineplot(-5, 5, hardsigmoid, height=7)
          ┌────────────────────────────────────────┐         
        1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⢀⡠⠖⠋⠉⠉⠉⠉⠉⠉⠉⠉│ hardσ(x)
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⣀⡤⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡠⠔⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
   f(x)   │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⡗⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠊⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠖⠋⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
        0 │⣀⣀⣀⣀⣀⣀⣀⣀⣀⠤⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          └────────────────────────────────────────┘         
          ⠀-5⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀5⠀         
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀         

julia> lineplot(-5, 5, sigmoid, height=7)
          ┌────────────────────────────────────────┐     
        1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⣀⡠⠤⠖⠒⠒⠋⠉⠉⠉⠉⠉⠉│ σ(x)
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢀⡠⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⣀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
   f(x)   │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⡏⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡔⠋⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠊⠁⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
        0 │⣀⣀⣀⣀⣀⣀⣀⠤⠤⠤⠒⠊⠉⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│     
          └────────────────────────────────────────┘     
          ⠀-5⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀5⠀     
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀     
```
"""
hardσ(x) = clamp((x + 3) / 6, 0, 1)

# https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html

const hardsigmoid = hardσ

"""
    logσ(x)

Return `log(σ(x))` which is computed in a numerically stable way.

```julia-repl
julia> lineplot(-5, 5, logsigmoid, height=7)
           ┌────────────────────────────────────────┐        
         0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡧⠤⠔⠒⠒⠒⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉│ logσ(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠖⠊⠉⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⠉⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
   f(x)    │⠀⠀⠀⠀⠀⠀⢀⡤⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⠀⠀⠀⣀⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⡤⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
        -6 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           └────────────────────────────────────────┘        
           ⠀-5⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀5⠀        
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀        
```
"""
@inline logσ(x) = -softplus(-x)

const logsigmoid = logσ

"""
    hardtanh(x) = max(-1, min(1, x))

Segment-wise linear approximation of `tanh`, much cheaper to compute.
See ["Large Scale Machine Learning"](https://ronan.collobert.com/pub/matos/2004_phdthesis_lip6.pdf).

See also [`tanh_fast`](@ref).
```julia-repl
julia> lineplot(-2, 2, hardtanh, height=7)
           ┌────────────────────────────────────────┐            
         1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⣀⠔⠋⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉│ hardtanh(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⣀⡤⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢀⡤⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
   f(x)    │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⡤⡷⠥⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│            
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠖⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠖⠋⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
        -1 │⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⠔⠋⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
           └────────────────────────────────────────┘            
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀            
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x

julia> lineplot(-2, 2, tanh, height=7)
           ┌────────────────────────────────────────┐        
         1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⣀⡠⠤⠤⠒⠒⠒⠊⠉⠉⠉│ tanh(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⢀⡠⠔⠊⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢀⡤⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
   f(x)    │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⡤⡷⠥⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠖⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡠⠔⠊⠁⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
        -1 │⣀⣀⣀⡠⠤⠤⠤⠖⠒⠊⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           └────────────────────────────────────────┘        
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀        
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀        
```
"""
hardtanh(x) = clamp(x, oftype(x, -1), oftype(x, 1))  # clamp(x, -1, 1) is type-stable, but would promote Int32, for which we have tests

"""
    relu(x) = max(0, x)

[Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function.

```julia-repl
julia> lineplot(-2, 2, relu, height=7)
          ┌────────────────────────────────────────┐        
        2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠋│ relu(x)
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠊⠁⠀⠀│        
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠊⠁⠀⠀⠀⠀⠀│        
   f(x)   │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⢀⡤⠖⠁⠀⠀⠀⠀⠀⠀⠀⠀│        
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⡠⠖⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⡠⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
        0 │⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣇⠔⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
          └────────────────────────────────────────┘        
          ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀        
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀        
```
"""
relu(x) = ifelse(x<0, zero(x), x)  # faster than max(zero(x), x), still preserves NaN

"""
    leakyrelu(x, a=0.01) = max(a*x, x)

Leaky [Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function.
You can also specify the coefficient explicitly, e.g. `leakyrelu(x, 0.01)`.

```julia-repl
julia> lineplot(-2, 2, x -> leakyrelu(x, 0.5), height=7)
           ┌────────────────────────────────────────┐       
         2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⠉│ #42(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊⠉⠀⠀⠀⠀│       
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⣀⡤⠖⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀│       
   f(x)    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣀⠤⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│       
           │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⣤⣤⡤⡧⠶⠭⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│       
           │⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠤⠤⠒⠒⠋⠉⠁⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│       
        -1 │⣀⣀⠤⠤⠒⠒⠊⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│       
           └────────────────────────────────────────┘       
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀       
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀       

julia> leakyrelu(-10f0, 0.2)
-2.0f0

julia> leakyrelu(-10f0, 0.02)
-0.5f0
```
"""
leakyrelu(x, a=oftf(x, leakyrelu_a)) = ifelse(x>0, float(x), oftf(x, a*x))  # max(a*x, x) is 3x slower

const leakyrelu_a = 0.01  # also used in gradient below

"""
    relu6(x) = min(max(0, x), 6)

[Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function capped at 6.
See ["Convolutional Deep Belief Networks"](https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf) from CIFAR-10.

```julia-repl
julia> lineplot(-10, 10, relu6, height=7)
          ┌────────────────────────────────────────┐         
        6 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠎⠉⠉⠉⠉⠉⠉⠉⠉│ relu6(x)
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⢀⡔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⡤⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
   f(x)   │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⡠⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢀⠖⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡔⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
        0 │⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⡧⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          └────────────────────────────────────────┘         
          ⠀-10⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀10⠀         
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀         
```
"""
relu6(x) = clamp(x, oftype(x, 0), oftype(x, 6))  # clamp promotes, but clamp(x, 0, 6) would promote x::Int32

"""
    rrelu(x, lo=1/8, hi=1/3) = max(a*x, x)
    # where `a` is randomly sampled from uniform distribution `U(lo, hi)`

Randomized Leaky Rectified Linear Unit activation function.
See ["Empirical Evaluation of Rectified Activations"](https://arxiv.org/abs/1505.00853)
You can also specify the bound explicitly, e.g. `rrelu(x, 0.0, 1.0)`.

```julia-repl
julia> lineplot(-20, 10, rrelu, height=7)
            ┌────────────────────────────────────────┐         
         10 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠖⠋│ rrelu(x)
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⢀⡠⠖⠋⠁⠀⠀⠀│         
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⢀⡠⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀│         
   f(x)     │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⡤⠤⣤⣤⢤⣤⣤⠤⠤⠤⢼⠮⠥⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│         
            │⣰⢀⣆⡄⣄⡄⡠⡰⠦⠷⡜⢢⠷⠳⠢⠊⠉⠉⠀⠀⠁⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
            │⠃⠉⠙⠘⠃⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
        -10 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
            └────────────────────────────────────────┘         
            ⠀-20⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀10⠀         
            ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀         

julia> extrema(rrelu.(fill(-10f0, 1000)))
(-3.3316886f0, -1.2548422f0)
```
"""
function rrelu(x::T, l=oftf(x,1/8), u=oftf(x,1/3)) where T<:Number
    a = (u - l) * rand(float(T)) + l
    return leakyrelu(x, a)
end

"""
    elu(x, α=1) = x > 0 ? x : α * (exp(x) - 1)

Exponential Linear Unit activation function.
See ["Fast and Accurate Deep Network Learning by Exponential Linear Units"](https://arxiv.org/abs/1511.07289).
You can also specify the coefficient explicitly, e.g. `elu(x, 1)`.

```julia-repl
julia> lineplot(-2, 2, elu, height=7)
           ┌────────────────────────────────────────┐       
         2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⠉│ elu(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊⠉⠀⠀⠀⠀│       
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⣀⡤⠖⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀│       
   f(x)    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣀⠤⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│       
           │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⡤⡧⠶⠭⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│       
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠤⠔⠒⠋⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│       
        -1 │⠤⠤⠤⠤⠔⠒⠒⠒⠊⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│       
           └────────────────────────────────────────┘       
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀       
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀       

julia> elu(-10f0)
-0.9999546f0

julia> elu(-10f0, 2)
-1.9999092f0
```
"""
@inline elu(x, α=1) = ifelse(x ≥ 0, float(x), oftf(x, α) * _expm1(x))

deriv_elu(Ω, α=1) = ifelse(Ω ≥ 0, one(Ω), Ω + oftype(Ω, α))

"""
    gelu_tanh(x) = 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))

Activation function from ["Gaussian Error Linear Units"](https://arxiv.org/abs/1606.08415) using tanh approximation.

This implementation uses `tanh` which allows for better pattern matching and fusion in optimizing 
compilers compared to the sigmoid-based implementation. For a potentially faster implementation 
that uses `sigmoid_fast`, see [`gelu_sigmoid`](@ref).

```julia-repl
julia> lineplot(-2, 2, gelu_tanh, height=7)
           ┌────────────────────────────────────────┐        
         2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊│ gelu_tanh(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠊⠁⠀⠀⠀│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀│        
   f(x)    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⣀⡠⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⣤⣤⣤⣤⣤⣤⣤⣤⡤⠤⠤⠤⠤⠤⠤⠤⣤⣤⣤⡤⡧⠶⠶⠭⠥⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠉⠉⠉⠉⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
        -1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           └────────────────────────────────────────┘        
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀        
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀        

julia> lineplot(-5, 0, gelu_tanh, height=7);

julia> lineplot!(ans, swish)
             ┌────────────────────────────────────────┐         
           0 │⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠒⠒⠤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸│ gelu_tanh(x) 
             │⠑⠒⠢⠤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇│ swish(x)
             │⠀⠀⠀⠀⠀⠈⠉⠒⠤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢆⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⠁│         
   f(x)      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠒⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⠀⢠⡇⠀│         
             │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⢄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠓⣄⠀⠀⠀⠀⠀⢠⡞⠀⠀│         
             │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠓⢄⣀⣀⡤⢣⠃⠀⠀│         
        -0.2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠇⠀⠀⠀│         
             └────────────────────────────────────────┘         
             ⠀-5⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀0⠀         
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀         
```
"""
@inline function gelu_tanh(x)
    α = oftf(x, 0.044715)
    λ = oftf(x, gelu_λ)
    x/2 * (1 + tanh_fast(λ * (x + α * x^3)))
end

const gelu_λ = √(2 / π)
const gelu_2λ = √(8 / π)

@inline function deriv_gelu_tanh(x)
    α = oftf(x, 0.044715)
    α2 = oftf(x, 0.08943)
    λ = oftf(x, gelu_λ)
    x2 = x * x
    t = muladd(x2, α, one(x))
    z = λ * x * t
    Ω = tanh_fast(z)
    sech2 = 1 - Ω^2
    (1 + Ω)/2 + x * λ * muladd(x2, α2, t) * sech2 / 2
end

"""
    gelu_sigmoid(x) = x * σ(√(8/π) * (x + 0.044715x^3))

Alternative implementation of the GELU activation function using `sigmoid` instead of `tanh`.
This is mathematically equivalent to [`gelu_tanh`](@ref) but may be faster in some cases.

The sigmoid-based implementation may prevent pattern matching and fusion in some optimizing 
compilers. Use [`gelu_tanh`](@ref) if you need better compiler optimization support.

See ["Gaussian Error Linear Units"](https://arxiv.org/abs/1606.08415).
"""
@inline function gelu_sigmoid(x)
    α = oftf(x, 0.044715)
    λλ = oftf(x, gelu_2λ)
    x * sigmoid_fast(λλ * x * muladd(x^2, α, one(x)))
end

@inline function deriv_gelu_sigmoid(x)
    α = oftf(x, 0.044715)
    α2 = oftf(x, 0.08943)
    λλ = oftf(x, gelu_2λ)
    x2 = x * x
    t = muladd(x2, α, one(x))
    Ω = sigmoid_fast(λλ * x * t)
    dσ = conj(Ω * (1 - Ω))
    muladd(dσ * λλ * muladd(x2, α2, t), x, Ω)
end

"""
    gelu_erf(x) = xΦ(x) = 0.5x * (1 + erf(x/√2))

Activation function from ["Gaussian Error Linear Units"](https://arxiv.org/abs/1606.08415) without approximation.
The SpecialFunctions.jl package needs to be loaded to use this function.
"""
function gelu_erf end
function deriv_gelu_erf end

"""
    gelu(x) = gelu_tanh(x)

Activation function from ["Gaussian Error Linear Units"](https://arxiv.org/abs/1606.08415). 
See [`gelu_tanh`](@ref).
"""
const gelu = gelu_tanh
# Need to alias the type as well to ensure serialization libraries still work
# See https://github.com/FluxML/NNlib.jl/issues/631
const var"#gelu" = typeof(gelu_tanh)
const deriv_gelu = deriv_gelu_tanh

"""
    swish(x) = x * σ(x)

Self-gated activation function.
See ["Swish: a Self-Gated Activation Function"](https://arxiv.org/abs/1710.05941).

```julia-repl
julia> lineplot(-2, 2, swish, height=7)
           ┌────────────────────────────────────────┐         
         2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤│ swish(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠖⠋⠁⠀│         
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠖⠋⠁⠀⠀⠀⠀⠀│         
   f(x)    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⢀⣀⡤⠔⠊⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
           │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⣤⣤⡤⡧⠴⠶⠯⠥⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│         
           │⠉⠑⠒⠒⠒⠒⠒⠒⠒⠒⠒⠒⠉⠉⠉⠉⠁⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
        -1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
           └────────────────────────────────────────┘         
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀         
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀         
```
"""
@inline swish(x) = x * sigmoid_fast(x)

@inline function deriv_swish(x)
    s = sigmoid_fast(x)
    s * (1 + x * (1 - s))
end

"""
    hardswish(x) = x * hardσ(x)

Hard-Swish activation function.
See ["Searching for MobileNetV3"](https://arxiv.org/abs/1905.02244).

```julia-repl
julia> lineplot(-2, 5, hardswish, height = 7)
           ┌────────────────────────────────────────┐             
         5 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠔⠒⠉│ hardswish(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡠⠔⠒⠉⠁⠀⠀⠀⠀│             
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠖⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀│             
   f(x)    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│             
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⢀⣀⠤⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│             
           │⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣇⣤⣤⣖⣚⣉⣁⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀│             
        -1 │⠉⠒⠒⠒⠒⠉⠉⠉⠉⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│             
           └────────────────────────────────────────┘             
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀5⠀             
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀             

julia> lineplot(-4, 0, hardswish, height = 7);

julia> lineplot!(ans, swish)
             ┌────────────────────────────────────────┐             
           0 │⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⢣⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡜│ hardswish(x)
             │⠒⠒⠢⠤⢄⣀⡀⠀⠀⠀⠀⠱⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠀│ swish(x)    
             │⠀⠀⠀⠀⠀⠀⠈⠉⠑⠒⠦⢄⣘⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡴⠃⠀⠀│             
   f(x)      │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠑⡖⠦⢄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⢔⠏⠁⠀⠀⠀│             
             │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠣⣄⠀⠉⠑⠒⠦⠤⢄⣀⣀⣀⣀⡠⠤⠖⣊⠕⠁⠀⠀⠀⠀⠀│             
             │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠓⠤⡀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠖⠁⠀⠀⠀⠀⠀⠀⠀│             
        -0.4 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠒⠢⠤⠤⠔⠒⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│             
             └────────────────────────────────────────┘             
             ⠀-4⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀0⠀             
             ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀             

julia> hardswish.(-5:5)'
1×11 adjoint(::Vector{Float64}) with eltype Float64:
 -0.0  -0.0  -0.0  -0.333333  -0.333333  0.0  0.666667  1.66667  3.0  4.0  5.0
```
"""
@inline hardswish(x) = x * hardσ(x)

deriv_hardswish(x) = ifelse(x < -3, oftf(x,0), ifelse(x > 3, oftf(x,1), x/3 + oftf(x,1/2)))

"""
    lisht(x) = x * tanh(x)

Activation function from 
["LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent ..."](https://arxiv.org/abs/1901.05894)

```julia-repl
julia> lineplot(-2, 2, lisht, height=7)
          ┌────────────────────────────────────────┐         
        2 │⠢⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔│ lisht(x)
          │⠀⠈⠑⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠊⠁⠀│         
          │⠀⠀⠀⠀⠈⠣⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠁⠀⠀⠀⠀│         
   f(x)   │⠀⠀⠀⠀⠀⠀⠀⠑⢆⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠁⠀⠀⠀⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠢⡄⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⢀⠔⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⢄⡀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⢀⡠⠖⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
        0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⠦⣄⣀⣀⣇⣀⣀⠤⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          └────────────────────────────────────────┘         
          ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀         
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀         

julia> lineplot!(ans, logcosh)
          ┌────────────────────────────────────────┐           
        2 │⠢⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔│ lisht(x)  
          │⠀⠈⠑⢦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡤⠊⠁⠀│ logcosh(x)
          │⠢⣄⠀⠀⠈⠣⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠁⠀⠀⣀⠔│           
   f(x)   │⠀⠈⠑⠢⣀⠀⠀⠑⢆⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠁⠀⣀⠔⠊⠁⠀│           
          │⠀⠀⠀⠀⠀⠉⠢⢄⡀⠉⠢⡄⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⢀⠔⠋⠀⡠⠔⠋⠁⠀⠀⠀⠀│           
          │⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⠦⣌⡓⢄⡀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⢀⡠⠖⣁⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀│           
        0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠓⠪⠷⣦⣄⣀⣀⣇⣀⣀⣤⠶⠕⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│           
          └────────────────────────────────────────┘           
          ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀           
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀           
```
"""
@inline lisht(x) = x * tanh_fast(x)

@inline function deriv_lisht(x)
    t = tanh_fast(x)
    t + x * (1 - t^2)
end

"""
    selu(x) = λ * (x ≥ 0 ? x : α * (exp(x) - 1))

    λ ≈ 1.05070...
    α ≈ 1.67326...

Scaled exponential linear units.
See ["Self-Normalizing Neural Networks"](https://arxiv.org/abs/1706.02515).

```julia-repl
julia> lineplot(-3, 2, selu, height=7)
           ┌────────────────────────────────────────┐        
         3 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ selu(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠤⠒│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⢀⣀⠤⠖⠊⠉⠀⠀⠀⠀│        
   f(x)    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⣀⡠⠤⠒⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⣉⠭⠛⡏⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⡤⠤⠒⠊⠉⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
        -2 │⠤⠤⠖⠒⠒⠒⠒⠒⠒⠒⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           └────────────────────────────────────────┘        
           ⠀-3⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀        
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀        

julia> selu(-10f0)
-1.7580194f0
```
"""
@inline function selu(x)
    λ = oftf(x, selu_λ)
    α = oftf(x, selu_α)
    λ * ifelse(x > 0, x, α * _expm1(x))
end

const selu_λ = 1.0507009873554804934193349852946
const selu_α = 1.6732632423543772848170429916717

function deriv_selu(Ω)
    λ = oftf(Ω, selu_λ)
    α = oftf(Ω, selu_α)
    ifelse(Ω > 0, λ, Ω + α * λ)
end

"""
    celu(x, α=1) = x ≥ 0 ? x : α * (exp(x/α) - 1)

Activation function from ["Continuously Differentiable Exponential Linear Units"](https://arxiv.org/abs/1704.07483).

```julia-repl
julia> lineplot(-2, 2, celu, height=7)
           ┌────────────────────────────────────────┐        
         2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⠉│ celu(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊⠉⠀⠀⠀⠀│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⣀⡤⠖⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀│        
   f(x)    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣀⠤⠖⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⡤⡧⠶⠭⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⠤⠔⠒⠋⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
        -1 │⠤⠤⠤⠤⠔⠒⠒⠒⠊⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           └────────────────────────────────────────┘        
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀        
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀        

julia> celu(-10f0)
-0.9999546f0
```
"""
@inline celu(x, α=1) = ifelse(x ≥ 0, float(x), oftf(x,α) * _expm1(x/oftf(x,α)))

deriv_celu(Ω, α=1) = ifelse(Ω > 0, oftf(Ω, 1), Ω / oftf(Ω, α) + 1)

"""
    trelu(x, theta=1) = x > theta ? x : 0

Threshold gated rectified linear activation function.
See ["Zero-bias autoencoders and the benefits of co-adapting features"](https://arxiv.org/abs/1402.3337)

```julia-repl
julia> lineplot(-2, 4, trelu, height=7)
          ┌────────────────────────────────────────┐         
        4 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠖⠋│ trelu(x)
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠖⠋⠁⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀│         
   f(x)   │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠴⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⣠⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
        0 │⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣇⣀⣀⣀⣀⣀⣀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│         
          └────────────────────────────────────────┘         
          ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀4⠀         
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀         
```
"""
trelu(x, theta=1) = ifelse(x <= theta, zero(x), x)

const thresholdrelu = trelu

"""
    softsign(x) = x / (1 + |x|)

See ["Quadratic Polynomials Learn Better Image Features"](http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/205) (2009).

```julia-repl
julia> lineplot(-5, 5, softsign, height=7)
           ┌────────────────────────────────────────┐            
         1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⣀⣀⠤⠤⠤⠤⠤│ softsign(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⣀⡤⠖⠒⠋⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⡔⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
   f(x)    │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⡯⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│            
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠁⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⠤⠤⠒⠋⠁⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
        -1 │⠒⠒⠒⠒⠒⠊⠉⠉⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
           └────────────────────────────────────────┘            
           ⠀-5⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀5⠀            
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀            

julia> lineplot!(ans, tanh)
           ┌────────────────────────────────────────┐            
         1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢀⡤⠖⠊⠉⠉⠉⣉⣉⣉⣉⣉⠭⠭⠭⠭⠭│ softsign(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⡔⣃⡤⠖⠒⠋⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀│ tanh(x)    
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⡞⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
   f(x)    │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⡯⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│            
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡴⠃⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⠤⠤⠒⢋⠕⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
        -1 │⣒⣒⣒⣒⣒⣊⣉⣉⣉⣉⣁⣀⣀⡠⠤⠒⠁⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
           └────────────────────────────────────────┘            
           ⠀-5⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀5⠀            
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀            

julia> softsign(1f0)
0.5f0

julia> softsign(100f0)
0.990099f0
```
"""
softsign(x) = x / (1 + abs(x))

deriv_softsign(x) = 1 / (1 + abs(x))^2

"""
    softplus(x) = log(exp(x) + 1)

See ["Deep Sparse Rectifier Neural Networks"](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf), JMLR 2011.

```julia-repl
julia> lineplot(-3, 3, softplus, height=7)
          ┌────────────────────────────────────────┐            
        4 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ softplus(x)
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠│            
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊⠁⠀│            
   f(x)   │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠔⠊⠁⠀⠀⠀⠀⠀│            
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⣀⡠⠤⠒⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⡧⠤⠒⠊⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
        0 │⣀⣀⣀⣀⣀⣀⣀⡠⠤⠤⠤⠤⠔⠒⠒⠚⠉⠉⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
          └────────────────────────────────────────┘            
          ⠀-3⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀3⠀            
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀            

julia> lineplot!(ans, relu)
          ┌────────────────────────────────────────┐            
        4 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ softplus(x)
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠│ relu(x)    
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡴⠞⠋⠁│            
   f(x)   │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⡴⠞⠋⠁⠀⠀⠀⠀│            
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⣀⡠⢤⡲⠝⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀│            
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣀⡧⠤⠒⠊⣉⠥⠚⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
        0 │⣀⣀⣀⣀⣀⣀⣀⣠⣤⣤⣤⣤⣔⣒⣒⣚⣉⣉⣁⣀⣇⠴⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│            
          └────────────────────────────────────────┘            
          ⠀-3⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀3⠀            
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀            

julia> softplus(16f0)
16.0f0
```
"""
function softplus(x)
    # Branching on `real(x)` keeps `exp` from overflowing and also makes this work for
    # complex inputs: softplus(z) = log(1 + exp(z)) on the principal branch (the two
    # rearrangements agree for `abs(imag(z)) ≤ π`, which covers typical use).
    real(x) < zero(real(x)) ? log1p(exp(x)) : x + log1p(exp(-x))
end

# The same without branches, so that it can be vectorized. Both terms switch at `x < 0`
# for the derivative at zero to be right when an AD differentiates through this code.
@inline function softplus(x::Union{Float32,Float64})
    neg = x < 0
    ifelse(neg, zero(x), x) + _log1p_unit(_exp(ifelse(neg, x, -x)))
end

"""
    logcosh(x)

Return `log(cosh(x))` which is computed in a numerically stable way.

```julia-repl
julia> lineplot(-5, 5, logcosh, height=7)
          ┌────────────────────────────────────────┐           
        5 │⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ logcosh(x)
          │⠉⠢⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠋│           
          │⠀⠀⠀⠑⠢⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠊⠁⠀⠀│           
   f(x)   │⠀⠀⠀⠀⠀⠀⠑⠦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠊⠁⠀⠀⠀⠀⠀│           
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠦⡀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⢀⡤⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀│           
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠓⠦⡀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⢀⡤⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│           
        0 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠢⢄⣀⣀⣇⣀⡠⠔⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│           
          └────────────────────────────────────────┘           
          ⠀-5⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀5⠀           
          ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀           
```
"""
@inline logcosh(x) = x + softplus(-2x) - oftf(x, log2)

const log2 = log(2)

"""
    mish(x) = x * tanh(softplus(x))

Activation function from ["Mish: A Self Regularized Non-Monotonic Neural Activation Function"](https://arxiv.org/abs/1908.08681).

```julia-repl
julia> lineplot(-5, 5, mish, height=7)
           ┌────────────────────────────────────────┐        
         5 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠖⠋│ mish(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡤⠒⠁⠀⠀⠀│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠔⠋⠁⠀⠀⠀⠀⠀⠀│        
   f(x)    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⢀⡠⠖⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢀⡤⠖⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣧⣔⣊⣁⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀│        
        -1 │⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           └────────────────────────────────────────┘        
           ⠀-5⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀5⠀        
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀        
```
"""
@inline mish(x) = x * _tanh_softplus(x)

_tanh_softplus(x) = tanh(softplus(x))

# With `t = exp(-abs(x))`, `tanh(softplus(x))` is a ratio of polynomials in `t`:
# a single `exp`, no `log1p` or `tanh`, and no overflow.
@inline function _tanh_softplus(x::Union{Float32,Float64})
    t = _exp(-abs(x))
    a = ifelse(x ≥ 0, 1 + 2t, t * (t + 2))
    b = ifelse(x ≥ 0, 2 * t * t, oftype(x, 2))
    a / (a + b)
end

@scalar_rule _tanh_softplus(x) (1 - Ω^2) * sigmoid_fast(x)

@inline function deriv_mish(x)
    t = _tanh_softplus(x)
    t + x * (1 - t^2) * sigmoid_fast(x)
end

"""
    tanhshrink(x) = x - tanh(x)

See ["Tanhshrink Activation Function"](https://www.gabormelli.com/RKB/Tanhshrink_Activation_Function).

```julia-repl
julia> lineplot(-3, 3, tanhshrink, height=7)
           ┌────────────────────────────────────────┐              
         3 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ tanhshrink(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡠⠤⠖⠊│              
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⢀⣀⡠⠤⠒⠊⠉⠁⠀⠀⠀⠀│              
   f(x)    │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⢤⣤⡤⠤⠤⠤⠤⠤⠤⡷⠶⠶⠶⠶⠶⠮⠭⠥⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│              
           │⠀⠀⠀⠀⠀⣀⡠⠴⠒⠊⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│              
           │⡠⠴⠒⠊⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│              
        -3 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│              
           └────────────────────────────────────────┘              
           ⠀-3⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀3⠀              
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀              

julia> tanhshrink.((-10f0, 10f0))
(-9.0f0, 9.0f0)
```
"""
@inline tanhshrink(x) = x - tanh_fast(x)

deriv_tanhshrink(x, Ω) = (x - Ω)^2

"""
    softshrink(x, λ=0.5) =
        (x ≥ λ ? x - λ : (-λ ≥ x ? x + λ : 0))

See ["Softshrink Activation Function"](https://www.gabormelli.com/RKB/Softshrink_Activation_Function).

```julia-repl
julia> lineplot(-2, 2, softshrink, height=7)
           ┌────────────────────────────────────────┐              
         2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀│ softshrink(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡤⠔⠒⠉⠁│              
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⣀⡤⠤⠒⠋⠁⠀⠀⠀⠀⠀⠀│              
   f(x)    │⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⣤⡤⠤⠤⠤⠤⠤⠤⡧⠤⠤⠤⠤⠶⠮⠭⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│              
           │⠀⠀⠀⠀⠀⠀⢀⣀⠤⠖⠒⠉⠁⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│              
           │⠀⣀⠤⠔⠒⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│              
        -2 │⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│              
           └────────────────────────────────────────┘              
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀              
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀              

julia> lineplot!(ans, tanhshrink)
           ┌────────────────────────────────────────┐              
         2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀│ softshrink(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡤⠔⠒⣉⡡│ tanhshrink(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⣀⡤⠤⣒⣋⠥⠤⠒⠊⠉⠁⠀│              
   f(x)    │⠤⠤⠤⠤⠤⠤⠤⠤⠤⣤⣤⣤⣤⡤⠤⠤⠤⠤⠤⠤⡷⠶⠶⠶⠶⠶⠾⠿⠯⠭⠭⠤⠤⠤⠤⠤⠤⠤⠤⠤│              
           │⠀⢀⣀⡠⠤⠖⢒⣋⠭⠗⠒⠉⠁⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│              
           │⠊⣉⠤⠔⠒⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│              
        -2 │⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│              
           └────────────────────────────────────────┘              
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀              
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀

julia> softshrink.((-10f0, 10f0))
(-9.5f0, 9.5f0)
```
"""
function softshrink(x, λ = 0.5)
    lo = x - oftf(x, λ)
    hi = x + oftf(x, λ)
    ifelse(hi > 0, ifelse(lo < 0, zero(hi), lo), hi)
end

# Define broadcasts for activation functions on arrays
for f in ACTIVATIONS
  @eval $(f)(x::AbstractArray, args...) = $(f).(x, args...)
end

# Base does not specialize a call `(f ∘ g)(x...)` on the number of arguments, which costs
# nothing as long as that call is inlined. With an inlined activation inside it is too
# large for that, and every call would box its arguments, or fail to compile on a GPU
# (#597). These methods take such calls instead, for numbers, and are inlined.
# `act ∘ +` is what `bias_act!` broadcasts.
const Activation = Union{map(f -> typeof(getfield(@__MODULE__, f)), ACTIVATIONS)...}
const ComposedActivation = Union{ComposedFunction{<:Activation}, ComposedFunction{<:Any,<:Activation}}

@inline (c::ComposedActivation)(x::Vararg{Number,N}) where {N} = c.outer(c.inner(x...))

## Faster, less accurate, versions of some.

"""
    tanh_fast(x)

This is a faster but slighly less accurate version of `tanh`.

Where Julia's `tanh` function has an error under 2 eps, this
may be wrong by 5 eps, a reduction by less than one decimal digit. 

For `x::Float32` this is usually about 10 times faster,
with a smaller speedup for `x::Float64`.
`Float16` and `BFloat16` are computed in `Float32`.
For any other number types, it just calls `tanh`.

See also [`sigmoid_fast`](@ref).

```julia-repl
julia> tanh(0.5f0)
0.46211717f0

julia> tanh_fast(0.5f0)
0.46211714f0

julia> hard_tanh(0.5f0)
0.5f0
```
"""
@inline function tanh_fast(x::Float32)
    # This method added in NNlib.jl#345 by @mcabbott and @oscardssmith,
    # with coeffiecients found using Remez.jl
    x2 = abs2(x)
    n = evalpoly(x2, (1.0f0, 0.1346604f0, 0.0035974074f0, 2.2332108f-5, 1.587199f-8))
    d = evalpoly(x2, (1.0f0, 0.4679937f0, 0.026262015f0, 0.0003453992f0, 8.7767893f-7))
    ifelse(x2 < 66f0, x * (n / d), sign(x))
end

@inline function tanh_fast(x::Float64)
    exp2x = _exp(x + x)
    y = (exp2x - 1) / (exp2x + 1) 
    # That has large errors near zero; using `expm1` would more accurate, but about as slow as `tanh`.
    # Instead, we switch to a polynomial, which is very accurate within its range:
    x2 = x * x
    ypoly = x * evalpoly(x2, (1.0, -0.33333333333324583, 0.13333333325511604, -0.05396823125794372, 0.02186660872609521, -0.008697141630499953))
    ifelse(x2 > 900.0, sign(x), ifelse(x2 < 0.017, ypoly, y))
end

# These approximations are a bit slower with ForwardDiff.Dual numbers, let's use Base.
# Float16 and BFloat16 take the Float32 one, see "Half precision" below.
tanh_fast(x::Number) = Base.tanh(x)

"""
    sigmoid_fast(x)

This used to be a faster, and very slightly less accurate, version of `sigmoid`.
For `Float32` and `Float64` the two are now computed in the same way, and differ
only in that `sigmoid_fast(x)` is exactly zero for `x < -80`.

See also [`tanh_fast`](@ref).

```julia-repl
julia> sigmoid(0.2f0)
0.54983395f0

julia> sigmoid_fast(0.2f0)
0.54983395f0

julia> hardσ(0.2f0)
0.53333336f0
```
"""
@inline function sigmoid_fast(x::Real)
    t = _exp(ifelse(x ≥ 0, -x, x))
    y = ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
    ifelse(x > 40, one(y), ifelse(x < -80, zero(y), y))
end
# For x::Float32, this is not as quick as the rational tanh_fast(x) above,
# but that polynomial has poor relative accuracy for negative x.

# Complex inputs: the `_fast` clamped approximation is real-only, so fall back to the
# holomorphic `σ` (which has its own overflow-safe complex method). This keeps `swish`,
# `logσ`, and the `softplus`/`logσ` gradients holomorphic and GPU-compilable; using a
# plain method (rather than the deprecation path below) avoids the `depwarn` string
# interpolation that cannot compile in a GPU kernel.
sigmoid_fast(x::Complex) = σ(x)

function sigmoid_fast(x::Number)
    Base.depwarn("sigmoid only makes sense on real numbers, got $(typeof(x))", :sigmoid_fast)
    sigmoid(x)
end

## Half precision
#
# For `Float16` and `BFloat16` the activations below are computed in `Float32` and rounded
# once at the end, as PyTorch does in its kernels (`OpMathType.h`) and Base in
# `exp(::Float16)`. Half precision arithmetic rounds after every operation, and most CPUs
# have to convert to `Float32` and back each time. The activations left out only select or
# clamp (`relu`, `relu6`, `hardtanh`, `trelu`), do a single operation (`softshrink`) or
# call `leakyrelu` (`rrelu`).

const HalfFloat = Union{Float16,BFloat16}

for f in (:σ, :hardσ, :logσ, :gelu_tanh, :gelu_sigmoid, :gelu_erf, :swish, :hardswish,
          :lisht, :selu, :softsign, :softplus, :logcosh, :mish, :tanhshrink,
          :tanh_fast, :sigmoid_fast, :deriv_gelu_tanh, :deriv_gelu_sigmoid, :deriv_gelu_erf,
          :deriv_swish, :deriv_hardswish, :deriv_lisht, :deriv_softsign, :deriv_mish)
    @eval @inline $f(x::T) where {T<:HalfFloat} = T($f(Float32(x)))
end

# `x - Ω` has few correct digits in half precision, and none for large `x`.
@inline deriv_tanhshrink(x::T, Ω::T) where {T<:HalfFloat} = T(tanh_fast(Float32(x))^2)

for f in (:elu, :celu)
    @eval @inline $f(x::T, α=1) where {T<:HalfFloat} = T($f(Float32(x), α))
end

@inline leakyrelu(x::T, a=Float32(leakyrelu_a)) where {T<:HalfFloat} =
    T(leakyrelu(Float32(x), a))

"""
    NNlib.fast_act(f, [x::AbstractArray])

Replaces `f == tanh` with [`tanh_fast`](@ref), etc.

Takes an optional 2nd argument, so that you can disable
this replacement for some array or element types.
"""
@inline fast_act(f::F, ::AbstractArray = 1:0) where {F<:Function} = f
@inline fast_act(::typeof(tanh), ::AbstractArray = 1:0) = tanh_fast
@inline fast_act(::typeof(sigmoid), ::AbstractArray = 1:0) = sigmoid_fast

## Define rrules for some activation functions, along with the
## broadcasted rrule activation functions.

## This is a performance hack specifically for Zygote, because it doesn't handle fused
## broadcasts well; but it generally should be good (or at least harmless) for any AD, as
## it saves ADing the broadcasting machinery.
## Related Issue https://github.com/JuliaDiff/ChainRulesCore.jl/issues/271

UNARY_ACTS = [ # f, dfdx
    ## In the same order as above!
    ## Derivatives are the plain (un-conjugated) f'; the broadcasted rrule below applies
    ## `conj`, and the scalar `@scalar_rule` conjugates internally, so both paths yield the
    ## ChainRules-convention `conj(f')` needed for correct complex (holomorphic) gradients.
    (:σ,            :(Ω * (1 - Ω))),
    (:hardσ,        :(ifelse((Ω>0)&(Ω<1), oftf(Ω, 1)/oftf(Ω, 6), oftf(Ω, 0)))),
    (:logσ,         :(sigmoid_fast(-x))),
    (:hardtanh,     :((Ω>-1) & (Ω<1))),
    (:relu,         :(Ω > 0)),
    (:leakyrelu,    :(ifelse(Ω > 0, oftf(Ω, 1), oftf(Ω, 1)/oftf(Ω, 100)))),  # 1/100 == leakyrelu_a; built from Ints to stay Float64-free for GPU (Metal) kernels
    (:relu6,        :((Ω>0) & (Ω<6))),
    # rrelu is random, can't write a rule.
    (:elu,          :(deriv_elu(Ω))),
    (:gelu_tanh,    :(deriv_gelu_tanh(x))),
    (:gelu_sigmoid, :(deriv_gelu_sigmoid(x))),
    (:gelu_erf,     :(deriv_gelu_erf(x))),
    (:swish,        :(deriv_swish(x))),
    (:hardswish,    :(deriv_hardswish(x))),
    (:lisht,        :(deriv_lisht(x))),
    (:selu,         :(deriv_selu(Ω))),
    (:celu,         :(deriv_celu(Ω))),
    (:trelu,        :(Ω > 0)),
    (:softsign,     :(deriv_softsign(x))),
    (:softplus,     :(sigmoid_fast(x))),
    # (:softplus,     :(1 - @fastmath exp(-Ω))),  # slightly faster, check accuracy?
    (:logcosh,      :(tanh_fast(x))),
    (:mish,         :(deriv_mish(x))),
    (:tanhshrink,   :(deriv_tanhshrink(x, Ω))),
    (:softshrink,    :(Ω != 0)),
    ## Fast variants are the same!
    (:tanh_fast,    :(1 - Ω^2)),
    (:sigmoid_fast, :(Ω * (1 - Ω))),
]

for (f, dfdx) in UNARY_ACTS
    @eval @scalar_rule($f(x), $dfdx)

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function rrule(::typeof(broadcasted),
                         ::typeof($f), x::Union{Numeric, Broadcast.Broadcasted})
        Ω = $f.(x)
        function $pullback(dΩ)
            x_thunk = InplaceableThunk(
                dx -> @.(dx += dΩ * conj($dfdx)),
                @thunk @.(dΩ * conj($dfdx))
            )
            NoTangent(), NoTangent(), x_thunk
        end
        return Ω, $pullback
    end
end

# NO_ACT_GRAD = ChainRulesCore.@not_implemented "for simplicitly NNlib assumes the 2nd argument of this activation function is a constant"
NO_ACT_GRAD = NaN  ## Still reminds you not to use this, but is perhaps more GPU friendly.

BINARY_ACTS = [ # f, dfdx1, dfdx2
    ## In the same order as above!
    (:leakyrelu,   :(ifelse(Ω > 0, oftf(Ω, 1), oftf(Ω, x2))), NO_ACT_GRAD),
    (:elu,         :(deriv_elu(Ω, x2)),      NO_ACT_GRAD),
    (:celu,        :(deriv_celu(Ω, x2)),     NO_ACT_GRAD),
    (:trelu,       :(Ω > 0),                 ZeroTangent()),
    (:softshrink,  :(Ω != 0),                NO_ACT_GRAD),
]

for (f, dfdx1, dfdx2) in BINARY_ACTS
    @eval @scalar_rule($f(x1, x2), ($dfdx1, $dfdx2))

    pullback = Symbol(:broadcasted_, f, :_pullback_2arg)
    @eval function rrule(::typeof(broadcasted),
                         ::typeof($f), 
                         x1::Union{Numeric, Broadcast.Broadcasted}, x2::Number)
        Ω = $f.(x1, x2)
        ## Allowing x2::Array would allow size(Ω) != size(x1), which is not handled here:
        $pullback(dΩ) = (NoTangent(), NoTangent(), @.(dΩ * conj($dfdx1)), NO_ACT_GRAD)
        return Ω, $pullback
    end
end
