doc"""
    σ(x) = 1 / (1 + exp(-x))

Classic [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation
function.

```
       ┌────────────────────────────────────────┐
     1 │                    |                .__│ σ(x)
       │                    |           ._-/'`  │
       │                    |        .r/`       │
       │                    |     .r/`          │
       │                    |   .r/             │
       │                    |  r`               │
       │                    |.'                 │
f(x)   │                   .F                   │
       │                 ./`|                   │
       │               .-`  |                   │
       │             .-`    |                   │
       │          ../`      |                   │
       │       ._-'         |                   │
       │  ._r-/`            |                   │
     0 │''`                 |                   │
       └────────────────────────────────────────┘
       -3                                       3
                           x
```
""" σ
σ(x) = one(x) / (one(x) + exp(-x))

# ForwardDiff numerical stability hack
σ(x::Float32) = ifelse(x < -80.0f0, zero(x), one(x) / (one(x) + exp(-x)))


doc"""
    relu(x) = max(0, x)

[Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function.

```
       ┌────────────────────────────────────────┐
     3 │                    |                  /│ relu(x)
       │                    |                ./ │
       │                    |               .`  │
       │                    |              /`   │
       │                    |            ./     │
       │                    |           r`      │
       │                    |          /        │
f(x)   │                    |        ./         │
       │                    |       .`          │
       │                    |     ./`           │
       │                    |    ./             │
       │                    |   .`              │
       │                    |  /`               │
       │                    |,'                 │
     0 │____________________D`                  │
       └────────────────────────────────────────┘
       -3                                       3
                           x
```
""" relu
relu(x) = max(zero(x), x)


doc"""
    leakyrelu(x) = max(0.01x, x)

Leaky [Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function.
You can also specify the coefficient explicitly, e.g. `leakyrelu(x, 0.01)`.

```
        ┌────────────────────────────────────────┐
      3 │                    |                 ./│ leakyrelu(x, 0.2)
        │                    |               ./` │
        │                    |              r`   │
        │                    |            ./     │
        │                    |          ,'       │
        │                    |        ./`        │
        │                    |      .r`          │
f(x)    │                    |     J`            │
        │                    |   ./              │
        │                    | _/`               │
        │                    |/                  │
        │'''''''''''''===PPP/F''''''''''''''''''`│
        │    .__.--/''`      |                   │
        │-/''`               |                   │
     -1 │                    |                   │
        └────────────────────────────────────────┘
        -3                                       3
                            x
```
""" leakyrelu
leakyrelu(x::T, a::T) where T = max(a * x, x)
leakyrelu(x::T, a::S) where T<:AbstractFloat where S<:AbstractFloat = max(T(a) * x, x)

function leakyrelu(x::T, a::S) where {T, S}
    _T = promote_type(T, S)
    max(_T(a) * _T(x), _T(x))
end

leakyrelu(x) = leakyrelu(x, 0.01)


doc"""
    elu(x) = x > 0 ? x : α * (exp(x) - 1)

    α = 1

Exponential Linear Unit activation function. 
See [Fast and Accurate Deep Network Learning by Exponential Linear Units](https://arxiv.org/abs/1511.07289)
You can also specify the coefficient explicitly, e.g. `elu(x, 1)`.

```
        ┌────────────────────────────────────────┐
      3 │                    |                 ./│ elu(x)
        │                    |               ./` │
        │                    |              r`   │
        │                    |            ./     │
        │                    |          ,'       │
        │                    |        ./`        │
        │                    |      .r`          │
f(x)    │                    |     J`            │
        │                    |   ./              │
        │                    | _/`               │
        │                    |/                  │
        │''''''''''''''''''=7F''''''''''''''''''`│
        │               ._-` |                   │
        │           __r/'    |                   │
     -1 │___.----/''         |                   │
        └────────────────────────────────────────┘
        -3                                       3
                            x
```
""" elu
elu(x::T, α::T) where T = ifelse(x ≥ zero(x), x, α * (exp(x) - one(x)))
elu(x::T, α::S) where T<:AbstractFloat where S<:AbstractFloat = ifelse(x ≥ zero(x), x, T(α) * (exp(x) - one(x)))

function elu(x::T, α::S) where {T, S}
    _T = promote_type(T, S)
    ifelse(x ≥ zero(x), _T(x), _T(α) * (exp(_T(x)) - one(_T(x))))
end

elu(x) = elu(x, 1.0)


doc"""
    swish(x) = x * σ(x)

Self-gated actvation function.
See [Swish: a Self-Gated Activation Function](https://arxiv.org/pdf/1710.05941.pdf).

```
        ┌────────────────────────────────────────┐
      3 │                    |                  ,│ swish(x)
        │                    |                .r`│
        │                    |               .`  │
        │                    |             ./    │
        │                    |            r`     │
        │                    |          .'       │
        │                    |        ./`        │
f(x)    │                    |      .r`          │
        │                    |     J`            │
        │                    |  .r/              │
        │                    |u-`                │
        │====7'''''''''''==7PF''''''''''''''''''`│
        │    ''''''''''''`   |                   │
        │                    |                   │
     -1 │                    |                   │
        └────────────────────────────────────────┘
        -3                                       3
                            x
```
""" swish
swish(x) = x * σ(x)


doc"""
    selu(x) = λ * (x ≥ 0 ? x : α * (exp(x) - 1))

    λ ≈ 1.0507
    α ≈ 1.6733

Scaled exponential linear units.
See [Self-Normalizing Neural Networks](https://arxiv.org/pdf/1706.02515.pdf).

```
        ┌────────────────────────────────────────┐
      4 │                    |                   │ selu(x)
        │                    |                   │
        │                    |                 ./│
        │                    |              ./'  │
        │                    |            ./`    │
        │                    |         .f'       │
        │                    |      ../`         │
f(x)    │                    |    .f`            │
        │                    | ../`              │
        │                    Lf`                 │
        │'''''''''''''''''')FF''''''''''''''''''`│
        │               ._/` |                   │
        │            ..-'    |                   │
        │   .____--/'`       |                   │
     -2 │'''`                |                   │
        └────────────────────────────────────────┘
        -3                                       3
                            x
```
""" selu
function selu(x::T) where T <: AbstractFloat
    T(1.0507009873554804934193349852946) * ifelse(x ≥ zero(x), x, T(1.6732632423543772848170429916717) * (exp(x) - one(x)))
end

function selu(x)
    1.0507009873554804934193349852946 * ifelse(x ≥ zero(x), Float64(x), 1.6732632423543772848170429916717 * (exp(x) - 1.0))
end


doc"""
    softsign(x) = x / (1 + |x|)

See [Quadratic Polynomials Learn Better Image Features](http://www.iro.umontreal.ca/~lisa/publications2/index.php/attachments/single/205).

```
        ┌────────────────────────────────────────┐
      1 │                    |                   │ softsign(x)
        │                    |                 ._│
        │                    |         ._---'''` │
        │                    |     u-/'`         │
        │                    |  .r'              │
        │                    |.-`                │
        │                    |/                  │
f(x)    │-------------------nP------------------*│
        │                  ./|                   │
        │                 r` |                   │
        │              ./'   |                   │
        │          ._-/`     |                   │
        │ .___r-f''`         |                   │
        │''                  |                   │
     -1 │                    |                   │
        └────────────────────────────────────────┘
        -3                                       3
                            x
```
""" softsign
softsign(x) = x / (one(x) + abs(x))


doc"""
    softplus = log(exp(x) + 1)

See [Deep Sparse Rectifier Neural Networks](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf).

```
       ┌────────────────────────────────────────┐
     4 │                    |                   │ softplus(x)
       │                    |                   │
       │                    |                   │
       │                    |                  ,│
       │                    |                _/`│
       │                    |              ./   │
       │                    |            .r`    │
f(x)   │                    |          .-`      │
       │                    |        ./`        │
       │                    |     .r/`          │
       │                    |   .r/             │
       │                    |../`               │
       │                 ._rF`                  │
       │            ._r-'`  |                   │
     0 │____r----/''`       |                   │
       └────────────────────────────────────────┘
       -3                                       3
                           x
```
""" softplus
softplus(x) = log1p(exp(x))