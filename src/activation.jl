σ(x) = 1 / (1 + exp(-x))

# ForwardDiff numerical stability hack
σ(x::Float32) = ifelse(x < -80, zero(x), 1 / (1 + exp(-x)))

relu(x) = max(0, x)

function softmax!(out::AbstractVecOrMat, xs::AbstractVecOrMat)
  out .= xs .- maximum(xs)
  out .= exp.(out) ./ sum(exp, out, 1)
end

softmax!(xs) = softmax!(xs, xs)
softmax(xs) = softmax!(similar(xs), xs)

function ∇softmax!(out::AbstractVecOrMat, Δ::AbstractVecOrMat, xs::AbstractVecOrMat)
  s = sum(exp, xs, 1)
  out .= exp.(xs)./s.*(Δ .- sum(Δ .* exp.(xs), 1)./s)
end

∇softmax!(Δ, xs) = ∇softmax!(Δ, Δ, xs)
∇softmax(Δ, xs) = ∇softmax!(similar(Δ), Δ, xs)
