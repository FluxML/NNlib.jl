module NNlibSpecialFunctionsExt

using NNlib: NNlib, oftf
using SpecialFunctions: erf

# Full gelu (gelu_erf)
NNlib.gelu_erf(x) = x/2*(1 + erf(x/sqrt(oftf(x,2))))

function NNlib.deriv_gelu_erf(x)
    SQRT2 = sqrt(oftf(x,2))
    Φ = (1 + erf(x/SQRT2))/2
    Φ + x/SQRT2*exp(-(x^2)/2)/sqrt(oftf(x,π))
end

end