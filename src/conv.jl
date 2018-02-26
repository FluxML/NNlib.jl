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

function pdims(x; window=2, padding=0, stride=window, o...)
  N = ndims(x)
  ntuple(N) do i
    if i < N-1
      wi = (if isa(window,Number); window; else window[i]; end)
      pi = (if isa(padding,Number); padding; else padding[i]; end)
      si = (if isa(stride,Number); stride; else stride[i]; end)
      1 + div(size(x,i) + 2*pi - wi, si)
    else
      size(x,i)
    end
  end
end

function cdims(w,x; padding=0, stride=1, o...)
  N = ndims(x)
  ntuple(N) do i
    if i < N-1
      pi = (if isa(padding,Number); padding; else padding[i]; end)
      si = (if isa(stride,Number); stride; else stride[i]; end)
      1 + div(size(x,i) - size(w,i) + 2*pi, si)
    elseif i == N-1
      size(w,N)
    else # i == N
      size(x,N)
    end
  end
end

include("impl/pool.jl")
include("impl/conv.jl")

maxpool2d(x, k; pad = 0) = pool2d(x; window = k, padding = pad, mode = 0)
avgpool2d(x, k; pad = 0) = pool2d(x; window = k, padding = pad, mode = 1)

maxpool3d(x, k; pad = 0) = pool3d(x; window = k, padding = pad, mode = 0)
avgpool3d(x, k; pad = 0) = pool3d(x; window = k, padding = pad, mode = 1)
