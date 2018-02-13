
"""
    logσ(x)

Return `log(σ(x))` which is computed in a numerically stable way.

    julia> logσ(0.)
    -0.6931471805599453
    julia> logσ.([-100, -10, 100.])
    3-element Array{Float64,1}:
     -100.0
      -10.0
       -0.0
"""
function logσ(x)
  max_v = max(zero(x), -x)
  z = exp(-max_v) + exp(-x-max_v)
  -(max_v + log(z))
end

∇logσ(Δ, x) = Δ * (1 - σ(x))

const logsigmoid = logσ
const ∇logsigmoid = ∇logσ
