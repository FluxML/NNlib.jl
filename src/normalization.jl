# TODO: add CPU implementation
function batchnorm end

function ∇batchnorm end


function ChainRulesCore.rrule(::typeof(batchnorm), g, b, x, running_mean, running_var, momentum; kw...)
  y = batchnorm(g, b, x, running_mean, running_var, momentum; kw...) 
  function batchnorm_pullback(Δ)
    grad = ∇batchnorm(g, b, x, unthunk(Δ), running_mean, running_var, momentum; kw...)
    (NoTangent(), grad..., NoTangent(), NoTangent(), NoTangent())
  end
  y, batchnorm_pullback
end
