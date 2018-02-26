Dims{N} = NTuple{N,Integer}

include("impl/pool.jl")
include("impl/conv.jl")

function pdims(dims::Dims{N}, window, padding, stride) where N
  ntuple(Val{N}) do i
    if i < N-1
      1 + (dims[i] + 2*padding[i] - window[i])÷stride[i]
    else
      dims[i]
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

maxpool(x::AbstractArray{<:Real,4}, k::Dims{2}; pad = (0,0), stride = k) =
  maxpool2d!(similar(x, pdims(size(x), k, pad, stride)),
             x, window = k, padding = pad, stride = stride)

maxpool(x::AbstractArray{<:Real,5}, k::Dims{3}; pad = (0,0,0), stride = k) =
  maxpool3d!(similar(x, pdims(size(x), k, pad, stride)),
             x, window = k, padding = pad, stride = stride)

maxpool_grad(dy::AbstractArray{<:Real,4}, y::AbstractArray{<:Real,4}, x::AbstractArray{<:Real,4},
             k::Dims{2}; pad = (0,0), stride = k) =
  maxpool2d_grad!(similar(x), dy, y, x,
                  window = k, padding = pad, stride = stride)

maxpool_grad(dy::AbstractArray{<:Real,5}, y::AbstractArray{<:Real,5}, x::AbstractArray{<:Real,5},
             k::Dims{3}; pad = (0,0,0), stride = k) =
  maxpool3d_grad!(similar(x), dy, y, x,
                  window = k, padding = pad, stride = stride)

meanpool(x::AbstractArray{<:Real,4}, k::Dims{2}; pad = (0,0), stride = k) =
  meanpool2d!(similar(x, pdims(size(x), k, pad, stride)),
             x, window = k, padding = pad, stride = stride)

meanpool(x::AbstractArray{<:Real,5}, k::Dims{3}; pad = (0,0,0), stride = k) =
  meanpool3d!(similar(x, pdims(size(x), k, pad, stride)),
             x, window = k, padding = pad, stride = stride)

meanpool_grad(dy::AbstractArray{<:Real,4}, y::AbstractArray{<:Real,4}, x::AbstractArray{<:Real,4},
             k::Dims{2}; pad = (0,0), stride = k) =
  meanpool2d_grad!(similar(x), dy, y, x,
                  window = k, padding = pad, stride = stride)

meanpool_grad(dy::AbstractArray{<:Real,5}, y::AbstractArray{<:Real,5}, x::AbstractArray{<:Real,5},
             k::Dims{3}; pad = (0,0,0), stride = k) =
  meanpool3d_grad!(similar(x), dy, y, x,
                  window = k, padding = pad, stride = stride)
