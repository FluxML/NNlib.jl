_tanh(x) = tanh(x)
Base.broadcasted(::typeof(tanh), x::AbstractArray) = @turbo _tanh.(x)

_softsign(x) = x / (1 + abs(x))
Base.broadcasted(::typeof(NNlib.softsign), x::AbstractArray) = @turbo _softsign.(x)

_softplus(x) = log1p(exp(-abs(x)))
Base.broadcasted(::typeof(NNlib.softplus), x::AbstractArray) = (@turbo _softplus.(x)) .+ NNlib.relu.(x)

function _sigmoid(x)
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end
Base.broadcasted(::typeof(NNlib.sigmoid), x::AbstractArray) = @turbo _sigmoid.(x)
Base.broadcasted(::typeof(NNlib.sigmoid_fast), x::AbstractArray) = @turbo _sigmoid.(x) # don't do the same for tanh_fast, it would be slower

function _hardsigmoid(x)
    clamp((x + 3) / 6, 0, 1)
end
Base.broadcasted(::typeof(NNlib.hardsigmoid), x::AbstractArray) = @turbo _hardsigmoid.(x)

_logsigmoid(x) = -_softplus(-x)
Base.broadcasted(::typeof(NNlib.logsigmoid), x::AbstractArray) = @turbo _logsigmoid.(x)

_swish(x) = x * _sigmoid(x)
Base.broadcasted(::typeof(NNlib.swish), x::AbstractArray) = @turbo _swish.(x)