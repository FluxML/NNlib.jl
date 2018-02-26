# convert padding etc. size to an Int array of the right dimension
function psize(p, x)
    nd = ndims(x)-2
    if isa(p,Number)
        fill(Int(p),nd)
    elseif length(p)==nd
        collect(Int,p)
    else
        throw(DimensionMismatch("psize: $p $nd"))
    end
end

include("impl/pool.jl")
include("impl/conv.jl")
