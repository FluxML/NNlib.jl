σ(x) = 1 / (1 + exp(-x))

# ForwardDiff numerical stability hack
σ(x::Float32) = ifelse(x < -80, zero(x), 1 / (1 + exp(-x)))

relu(x) = max(0, x)

"""
    elu(x; α = 1.)

Exponential Linear Unit

# Reference
- [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)]
  (https://arxiv.org/abs/1511.07289)
"""
elu(x::Real; α::AbstractFloat = 1.) = ifelse((x >= 0), x, α * (exp(x) - one(x)))

function softmax!(out::AbstractVecOrMat, xs::AbstractVecOrMat)
  # out[end, :] .= maximum(xs, 1)
  for j = 1:size(xs, 2)
    out[end, j] = 0
    for i = 1:size(xs, 1)
      @inbounds out[end, j] = max(out[end, j], xs[i, j])
    end
  end
  # out .= exp(xs .- out[end, :])
  for j = 1:size(out, 2), i = 1:size(out, 1)
    @inbounds out[i, j] = exp(xs[i, j] - out[end, j])
  end
  # out ./= sum(out, 1)
  for j = 1:size(out, 2)
    s = zero(eltype(out))
    for i = 1:size(out, 1)
      @inbounds s += out[i, j]
    end
    for i = 1:size(out, 1)
      @inbounds out[i, j] /= s
    end
  end
  return out
end

softmax!(xs) = softmax!(xs, xs)
softmax(xs) = softmax!(similar(xs), xs)

function ∇softmax!(out::AbstractVecOrMat, Δ::AbstractVecOrMat, xs::AbstractVecOrMat)
  s = sum(exp, xs, 1)
  out .= exp.(xs)./s.*(Δ .- sum(Δ .* exp.(xs), 1)./s)
end

∇softmax!(Δ, xs) = ∇softmax!(Δ, Δ, xs)
∇softmax(Δ, xs) = ∇softmax!(similar(Δ), Δ, xs)
