## Activation functions
#
# Some of activation functions have its wrapper function for GPU in NNlibCUDA.jl.
# https://github.com/JuliaGPU/CuArrays.jl/issues/614

ACTIVATIONS = [
    :σ, :hardσ, :tanh_fast, :hardtanh, :relu,
    :leakyrelu, :relu6, :rrelu, :elu, :gelu, :swish, :selu,
    :celu, :softplus, :softsign, :logσ, :logcosh,
    :mish, :tanhshrink, :softshrink, :trelu, :lisht,
    :tanh_fast, :tanh_faster, :sigmoid_fast,
    ]

for f in ACTIVATIONS
    @eval export $(f)
end

# Aliases
export sigmoid, hardsigmoid, logsigmoid, thresholdrelu

# of type float (to allow for integer inputs)
oftf(x, y) = oftype(float(x), y)

"""
    σ(x) = 1 / (1 + exp(-x))

Classic [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) activation
function.
Unicode `σ` can be entered as `\\sigma` then tab, in many editors.
The ascii name `sigmoid` is also exported.

```
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

```
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

```
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

```
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

```
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

```julia
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

julia> leakyrelu(-10f0, 1//5)
-2.0f0

julia> leakyrelu(-10f0, 1//20)
-0.5f0
```
"""
leakyrelu(x, a=oftf(x, 0.01)) = ifelse(x>0, float(x), oftf(x, a*x))  # max(a*x, x) is 3x slower

"""
    relu6(x) = min(max(0, x), 6)

[Rectified Linear Unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
activation function capped at 6.
See ["Convolutional Deep Belief Networks"](https://www.cs.toronto.edu/~kriz/conv-cifar10-aug2010.pdf) from CIFAR-10.

```
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

```julia
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
function rrelu(x::T, l=1//8, u=1//3) where T<:Number
    a = (u - l) * rand(float(T)) + l
    return leakyrelu(x, a)
end

"""
    elu(x, α=1) = x > 0 ? x : α * (exp(x) - 1)

Exponential Linear Unit activation function.
See ["Fast and Accurate Deep Network Learning by Exponential Linear Units"](https://arxiv.org/abs/1511.07289).
You can also specify the coefficient explicitly, e.g. `elu(x, 1)`.

```
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
elu(x, α=1) = ifelse(x ≥ 0, float(x), α * (exp(x) - 1))

deriv_elu(Ω, α=1) = ifelse(Ω ≥ 0, one(Ω), Ω + α)

"""
    gelu(x) = 0.5x * (1 + tanh(√(2/π) * (x + 0.044715x^3)))

Activation function from ["Gaussian Error Linear Units"](https://arxiv.org/abs/1606.08415).

```
julia> lineplot(gelu, -2, 2, height=7)
           ┌────────────────────────────────────────┐        
         2 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠔⠊│ gelu(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠊⠁⠀⠀⠀│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⣀⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀│        
   f(x)    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⣀⡠⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⣤⣤⣤⣤⣤⣤⣤⣤⡤⠤⠤⠤⠤⠤⠤⠤⣤⣤⣤⡤⡧⠶⠶⠭⠥⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠉⠉⠉⠉⠉⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
        -1 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           └────────────────────────────────────────┘        
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀        
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀        
```
"""
function gelu(x)
    α = oftf(x, 0.044715)
    λ = oftf(x, gelu_λ)
    x/2 * (1 + tanh(λ * (x + α * x^3)))
end

const gelu_λ = √(2 / π)

"""
    swish(x) = x * σ(x)

Self-gated activation function.
See ["Swish: a Self-Gated Activation Function"](https://arxiv.org/abs/1710.05941).

```
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
swish(x) = x * σ(x)

"""
    lisht(x) = x * tanh(x)

Activation function from 
["LiSHT: Non-Parametric Linearly Scaled Hyperbolic Tangent ..."](https://arxiv.org/abs/1901.05894)

```
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
```
"""
lisht(x) = x * tanh(x)

"""
    selu(x) = λ * (x ≥ 0 ? x : α * (exp(x) - 1))

    λ ≈ 1.05070...
    α ≈ 1.67326...

Scaled exponential linear units.
See ["Self-Normalizing Neural Networks"](https://arxiv.org/abs/1706.02515).

```
julia> lineplot(selu, -2, 2, height=7)
           ┌────────────────────────────────────────┐        
         3 │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│ selu(x)
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡠⠤⠔⠒│        
           │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⣀⣀⠤⠔⠒⠋⠉⠀⠀⠀⠀⠀│        
   f(x)    │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢀⣀⡤⠤⠒⠊⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           │⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⣉⡩⠭⠛⡏⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉│        
           │⠀⠀⠀⠀⠀⠀⠀⣀⣀⣀⡤⠤⠔⠒⠊⠉⠁⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
        -2 │⠒⠒⠉⠉⠉⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│        
           └────────────────────────────────────────┘        
           ⠀-2⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀2⠀        
           ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀x⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀        

julia> selu(-10f0)
-1.7580194f0
```
"""
function selu(x)
    λ = oftf(x, selu_λ)
    α = oftf(x, selu_α)
    λ * ifelse(x > 0, x, α * (exp(x) - 1))
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

```
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
celu(x, α=1) = ifelse(x ≥ 0, float(x), α * (exp(x/α) - 1))

"""
    trelu(x, theta=1) = x > theta ? x : 0

Threshold gated rectified linear activation function.
See ["Zero-bias autoencoders and the benefits of co-adapting features"](https://arxiv.org/abs/1402.3337)

```
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

```
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

"""
    softplus(x) = log(exp(x) + 1)

See ["Deep Sparse Rectifier Neural Networks"](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf), JMLR 2011.

```
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
softplus(x) = log1p(exp(-abs(x))) + max(x, 0)

"""
    logcosh(x)

Return `log(cosh(x))` which is computed in a numerically stable way.

```
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

```
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

```
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
tanhshrink(x) = x - tanh(x)

"""
    softshrink(x, λ=0.5) =
        (x ≥ λ ? x - λ : (-λ ≥ x ? x + λ : 0))

See ["Softshrink Activation Function"](https://www.gabormelli.com/RKB/Softshrink_Activation_Function).

```
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
softshrink(x, λ=oftf(x, 0.5)) = min(max(0, x - λ), x + λ)

# Provide an informative error message if activation functions are called with an array
for f in ACTIVATIONS
    @eval $(f)(x::AbstractArray, args...) =
      error("Use broadcasting (`", $(string(f)), ".(x)`) to apply activation functions to arrays.")
end

## Faster, less accurate, versions of some.
## TODO: mechanism to use automatically, and worry about GPUs.

"""
    tanh_fast(x) = exp(2x) / (1 + exp(2x))

This is faster, but slightly less accurate, version of `tanh`.

For `x::Float64`, this has errors up to about `3e-15`, sometimes `> 10 * eps(x)`.
For `x::Float32`, errors up to about `1f-6`. About 2-5 times faster.
"""
@inline function tanh_fast(x)
   exp2x = @fastmath exp(x + x)
   y = (exp2x - 1) / (exp2x + 1)
   stop = _fast_stop(tanh_fast, x)
   ifelse(x > stop, one(y), ifelse(x < -stop, -one(y), y))
end
# Badly behaved examples, without the stop:
# tanh_fast(400.0), tanh_fast(60f0), tanh_fast(Float16(6)) 
_fast_stop(::typeof(tanh_fast), x) = oftype(x, 30)
_fast_stop(::typeof(tanh_fast), ::Float16) = Float16(5)

"""
    tanh_faster(x)

This is even faster, via a rational approximation, 4/5 terms, from here:
https://math.stackexchange.com/questions/107292/rapid-approximation-of-tanhx

Errors of up to about `3e-5` or `3f-5`, on Float64 & Float32 numbers.
About 8-10 times faster. 
"""
@inline function tanh_faster(x::Real)
    T = float(typeof(x))
    x2 = 4x^2
    # y = 94.9339451088 * x * ((( x2 + 1.06315869e+03 ) * x2 + 1.88748783e+05 ) * x2 + 5.86237309e+06 ) / ((( ( x2 + 4.03183926e+03 ) * x2 + 1.64253046e+06 ) * x2 + 1.28592857e+08 ) * x2 + 1.11307745e+09 )
    num = @fastmath T(94.9339451088) * 2x * ((( x2 + T(1.06315869e+03) ) * x2 + T(1.88748783e+05) ) * x2 + T(5.86237309e+06) )
    den = @fastmath ((( ( x2 + T(4.03183926e+03) ) * x2 + T(1.64253046e+06) ) * x2 + T(1.28592857e+08) ) * x2 + T(1.11307745e+09) )
    y = num / den
    stop = _fast_stop(tanh_faster, x)
    ifelse(x > stop, one(y), ifelse(x < -stop, -one(y), y))
end
_fast_stop(::typeof(tanh_faster), x) = oftype(x, 10)

@inline function sigmoid_fast(x::Real)
   s = @fastmath exp(x)
   y = s / (s + 1)
   stop = _fast_stop(sigmoid_fast, x)
   ifelse(x > stop, one(y), ifelse(x < -stop, zero(y), y))
end
_fast_stop(::typeof(sigmoid_fast), x) = oftype(x, 60)
_fast_stop(::typeof(sigmoid_fast), ::Float16) = Float16(10)

@inline function sigmoid_faster(x::Real)
    T = float(typeof(x))
    x2 = x^2
    num = @fastmath T(47.4669725544) * x * ((( x2 + T(1.06315869e+03) ) * x2 + T(1.88748783e+05) ) * x2 + T(5.86237309e+06) )
    den = @fastmath ((( ( x2 + T(4.03183926e+03) ) * x2 + T(1.64253046e+06) ) * x2 + T(1.28592857e+08) ) * x2 + T(1.11307745e+09) )
    T(0.5) + num / den  # this is really hacked together
end


## Define rrules for some activation functions, along with the
## broadcasted rrule activation functions.
## TODO: add to the lists below all activations.

## This is a performance hack specifically for Zygote, because it doesn't handle fused
## broadcasts well; but it generally should be good (or at least harmless) for any AD, as
## it saves ADing the broadcasting machinery.
## Related Issue https://github.com/JuliaDiff/ChainRulesCore.jl/issues/271

UNARY_ACTS = [ # f, df
    (:relu,         :(x > 0)),
    (:hardtanh,     :(-1 < x < 1)),
    (:selu,         :(deriv_selu(Ω))),
    (:σ,            :(conj(Ω * (1 - Ω)))),
    (:elu,          :(deriv_elu(Ω))),
    (:softplus,     :(σ(x))),

    (:tanh_fast,    :(conj(1 - Ω^2))),
    (:tanh_faster,  :(conj(1 - Ω^2))),
    (:sigmoid_fast, :(conj(Ω * (1 - Ω)))),
    ]

for (f, df) in UNARY_ACTS
    @eval @scalar_rule($f(x), $df)

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function rrule(::typeof(broadcasted),
                         ::typeof($f), x::Numeric)
        Ω = $f.(x)
        function $pullback(Δ)
            x_thunk = InplaceableThunk(
                dx -> @.(dx += Δ * $df),
                @thunk @.(Δ * $df)
            )
            NoTangent(), NoTangent(), x_thunk
        end
        return Ω, $pullback
    end
end

BINARY_ACTS = [ # f, df1, df2
    (:elu, :(deriv_elu(Ω, x2)), :(NoTangent())), # TODO use real deriv instead of DNE
    ]

for (f, df1, df2) in BINARY_ACTS
    @eval @scalar_rule($f(x1, x2), ($df1, $df2))

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function rrule(::typeof(broadcasted),
                         ::typeof($f), 
                         x1::Numeric, x2::Numeric)
        Ω = $f.(x1, x2)
        function $pullback(Δ) 
            NoTangent(), NoTangent(), @.(Δ * $df1), @.(Δ * $df2)
        end
        return Ω, $pullback
    end
end

# For some activation functions, the gradient can be written only in terms of Ω.
# That means that `conv_bias_act` etc. can replace `Ω = σ.(x)` with `x .= σ.(x)`
# to save allocations. (Not so easy to opt-in to this mechanism, sadly.)
INPLACE_ACTS = [
    (:σ,            :(conj(Ω * (1 - Ω)))),
    (:relu,         :(Ω > 0)),
    (:leakyrelu,    :(ifelse(Ω > 0, one(Ω), oftf(Ω, 0.01)))),  # only the 1-argument method
    (:hardtanh,     :(-1 < Ω < 1)),
    (:selu,         :(deriv_selu(Ω))),
    (:elu,          :(deriv_elu(Ω))),

    (:identity,     :true),
    (:tanh,         :(conj(1 - Ω^2))),
    (:tanh_fast,    :(conj(1 - Ω^2))),
    (:tanh_faster,  :(conj(1 - Ω^2))),
    (:sigmoid_fast, :(conj(Ω * (1 - Ω)))),
    ]
