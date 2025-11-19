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

"""
    σ(x) = 1 / (1 + exp(-x))

Classic [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation
function.
Unicode `σ` can be entered as `\\sigma` then tab, in many editors.
The ascii name `sigmoid` is also exported.

See also [`sigmoid_fast`](@ref).

```julia-repl
julia> using UnicodePlots

julia> lineplot(sigmoid, -5, 5, height=7)
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
function σ(x)
    t = exp(-abs(x))
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
end

const sigmoid = σ

"""
    hardσ(x) = max(0, min(1, (x + 3) / 6))

Piecewise linear approximation of [`sigmoid`](@ref).

```julia-repl
julia> lineplot(hardsigmoid, -5, 5, height=7)
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

julia> lineplot(sigmoid, -5, 5, height=7)
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
julia> lineplot(logsigmoid, -5, 5, height=7)
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
logσ(x) = -softplus(-x)

const logsigmoid = logσ

"""
    hardtanh(x) = max(-1, min(1, x))

Segment-wise linear approximation of `tanh`, much cheaper to compute.
See ["Large Scale Machine Learning"](https://ronan.collobert.com/pub/matos/2004_phdthesis_lip6.pdf).

See also [`tanh_fast`](@ref).
```julia-repl
julia> lineplot(hardtanh, -2, 2, height=7)
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

julia> lineplot(tanh, -2, 2, height=7)
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
julia> lineplot(relu, -2, 2, height=7)
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
julia> lineplot(x -> leakyrelu(x, 0.5), -2, 2, height=7)
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
julia> lineplot(relu6, -10, 10, height=7)
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
julia> lineplot(rrelu, -20, 10, height=7)
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
julia> lineplot(elu, -2, 2, height=7)
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
elu(x, α=1) = ifelse(x ≥ 0, float(x), @fastmath oftf(x, α) * (exp(x) - 1))

deriv_elu(Ω, α=1) = ifelse(Ω ≥ 0, one(Ω), Ω + oftype(Ω, α))

"""
    gelu_tanh(x) = 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))

Activation function from ["Gaussian Error Linear Units"](https://arxiv.org/abs/1606.08415) using tanh approximation.

This implementation uses `tanh` which allows for better pattern matching and fusion in optimizing 
compilers compared to the sigmoid-based implementation. For a potentially faster implementation 
that uses `sigmoid_fast`, see [`gelu_sigmoid`](@ref).

```julia-repl
julia> lineplot(gelu_tanh, -2, 2, height=7)
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

julia> lineplot(gelu_tanh, -5, 0, height=7);

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
function gelu_tanh(x)
    α = oftf(x, 0.044715)
    λ = oftf(x, gelu_λ)
    x/2 * (1 + tanh_fast(λ * (x + α * x^3)))
end

const gelu_λ = √(2 / π)
const gelu_2λ = √(8 / π)

function deriv_gelu_tanh(x)
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
function gelu_sigmoid(x)
    α = oftf(x, 0.044715)
    λλ = oftf(x, gelu_2λ)
    x * sigmoid_fast(λλ * x * muladd(x^2, α, one(x)))
end

function deriv_gelu_sigmoid(x)
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
julia> lineplot(swish, -2, 2, height=7)
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

"""
    hardswish(x) = x * hardσ(x)

Hard-Swish activation function.
See ["Searching for MobileNetV3"](https://arxiv.org/abs/1905.02244).

```julia-repl
julia> lineplot(hardswish, -2, 5, height = 7)
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

julia> lineplot(hardswish, -4, 0, height = 7);

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
julia> lineplot(lisht, -2, 2, height=7)
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
lisht(x) = x * tanh_fast(x)

"""
    selu(x) = λ * (x ≥ 0 ? x : α * (exp(x) - 1))

    λ ≈ 1.05070...
    α ≈ 1.67326...

Scaled exponential linear units.
See ["Self-Normalizing Neural Networks"](https://arxiv.org/abs/1706.02515).

```julia-repl
julia> lineplot(selu, -3, 2, height=7)
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
function selu(x)
    λ = oftf(x, selu_λ)
    α = oftf(x, selu_α)
    λ * ifelse(x > 0, x, @fastmath α * (exp(x) - 1))
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
julia> lineplot(celu, -2, 2, height=7)
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
celu(x, α=1) = ifelse(x ≥ 0, float(x), oftf(x,α) * (exp(x/oftf(x,α)) - 1))

deriv_celu(Ω, α=1) = ifelse(Ω > 0, oftf(Ω, 1), Ω / oftf(Ω, α) + 1)

"""
    trelu(x, theta=1) = x > theta ? x : 0

Threshold gated rectified linear activation function.
See ["Zero-bias autoencoders and the benefits of co-adapting features"](https://arxiv.org/abs/1402.3337)

```julia-repl
julia> lineplot(trelu, -2, 4, height=7)
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
julia> lineplot(softsign, -5, 5, height=7)
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
julia> lineplot(softplus, -3, 3, height=7)
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
softplus(x) = log1p(exp(-abs(x))) + relu(x)

"""
    logcosh(x)

Return `log(cosh(x))` which is computed in a numerically stable way.

```julia-repl
julia> lineplot(logcosh, -5, 5, height=7)
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
logcosh(x) = x + softplus(-2x) - oftf(x, log2)

const log2 = log(2)

"""
    mish(x) = x * tanh(softplus(x))

Activation function from ["Mish: A Self Regularized Non-Monotonic Neural Activation Function"](https://arxiv.org/abs/1908.08681).

```julia-repl
julia> lineplot(mish, -5, 5, height=7)
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
mish(x) = x * tanh(softplus(x))

"""
    tanhshrink(x) = x - tanh(x)

See ["Tanhshrink Activation Function"](https://www.gabormelli.com/RKB/Tanhshrink_Activation_Function).

```julia-repl
julia> lineplot(tanhshrink, -3, 3, height=7)
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
tanhshrink(x) = x - tanh_fast(x)

"""
    softshrink(x, λ=0.5) =
        (x ≥ λ ? x - λ : (-λ ≥ x ? x + λ : 0))

See ["Softshrink Activation Function"](https://www.gabormelli.com/RKB/Softshrink_Activation_Function).

```julia-repl
julia> lineplot(softshrink, -2, 2, height=7)
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

## Faster, less accurate, versions of some.

"""
    tanh_fast(x)

This is a faster but slighly less accurate version of `tanh`.

Where Julia's `tanh` function has an error under 2 eps, this
may be wrong by 5 eps, a reduction by less than one decimal digit. 

For `x::Float32` this is usually about 10 times faster,
with a smaller speedup for `x::Float64`.
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
    exp2x = @fastmath exp(x + x)
    y = (exp2x - 1) / (exp2x + 1) 
    # That has large errors near zero; using `expm1` would more accurate, but about as slow as `tanh`.
    # Instead, we switch to a polynomial, which is very accurate within its range:
    x2 = x * x
    ypoly = x * evalpoly(x2, (1.0, -0.33333333333324583, 0.13333333325511604, -0.05396823125794372, 0.02186660872609521, -0.008697141630499953))
    ifelse(x2 > 900.0, sign(x), ifelse(x2 < 0.017, ypoly, y))
end

# These approximations are very badly behaved for Float16; none are fast.
# They are also a bit slower with ForwardDiff.Dual numbers, let's use Base:
tanh_fast(x::Number) = Base.tanh(x)

"""
    sigmoid_fast(x)

This is a faster, and very slightly less accurate, version of `sigmoid`.
For `x::Float32`, perhaps 3 times faster, and maximum errors 2 eps instead of 1.

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
function sigmoid_fast(x::Real)
    @static if VERSION ≥ v"1.11-"
        @inline
    end
    t = @fastmath exp(-abs(x))
    y = ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
    ifelse(x > 40, one(y), ifelse(x < -80, zero(y), y))
end
# For x::Float32, this is not as quick as the rational tanh_fast(x) above,
# but that polynomial has poor relative accuracy for negative x.

sigmoid_fast(x::Float16) = sigmoid(x)  # sigmoid_fast is extremely badly behaved at large x

function sigmoid_fast(x::Number)
    Base.depwarn("sigmoid only makes sense on real numbers, got $(typeof(x))", :sigmoid_fast)
    sigmoid(x)
end

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

## TODO: add to the lists below all activations.

UNARY_ACTS = [ # f, dfdx
    ## In the same order as above!
    (:σ,            :(conj(Ω * (1 - Ω)))),
    (:hardσ,        :(ifelse((Ω>0)&(Ω<1), oftf(Ω, 1/6), oftf(Ω, 1)))),
    (:logσ,         :(sigmoid_fast(-x))),
    (:hardtanh,     :((Ω>-1) & (Ω<1))),
    (:relu,         :(Ω > 0)),
    (:leakyrelu,    :(ifelse(Ω > 0, oftf(Ω, 1), oftf(Ω, leakyrelu_a)))),
    (:relu6,        :((Ω>0) & (Ω<6))),
    # rrelu is random, can't write a rule.
    (:elu,          :(deriv_elu(Ω))),
    (:gelu_tanh,    :(deriv_gelu_tanh(x))),
    (:gelu_sigmoid, :(deriv_gelu_sigmoid(x))),
    (:gelu_erf,     :(deriv_gelu_erf(x))),
    (:swish,        :(Ω + sigmoid_fast(x) * (1 - Ω))),
    (:hardswish,    :(deriv_hardswish(x))),
    # lisht
    (:selu,         :(deriv_selu(Ω))),
    (:celu,         :(deriv_celu(Ω))),
    (:trelu,        :(Ω > 0)),
    (:softsign,     :(deriv_softsign(x))),
    (:softplus,     :(sigmoid_fast(x))),
    # (:softplus,     :(1 - @fastmath exp(-Ω))),  # slightly faster, check accuracy?
    # logcosh
    # mish
    (:tanhshrink,    :((x - Ω)^2)),
    (:softshrink,    :(Ω != 0)),
    ## Fast variants are the same!
    (:tanh_fast,    :(conj(1 - Ω^2))),
    (:sigmoid_fast, :(conj(Ω * (1 - Ω)))),
]

for (f, dfdx) in UNARY_ACTS
    @eval @scalar_rule($f(x), $dfdx)

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function rrule(::typeof(broadcasted),
                         ::typeof($f), x::Union{Numeric, Broadcast.Broadcasted})
        Ω = $f.(x)
        function $pullback(dΩ)
            x_thunk = InplaceableThunk(
                dx -> @.(dx += dΩ * $dfdx),
                @thunk @.(dΩ * $dfdx)
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
        $pullback(dΩ) = (NoTangent(), NoTangent(), @.(dΩ * $dfdx1), NO_ACT_GRAD)
        return Ω, $pullback
    end
end
