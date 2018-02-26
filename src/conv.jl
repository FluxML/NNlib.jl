Dims{N} = NTuple{N,Integer}

include("impl/pool.jl")
include("impl/conv.jl")

# Convolutions

function cdims(w,x; padding=0, stride=1)
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

# Interface

conv(x::A, w::A; pad = 0, stride = 1) where A<:AbstractArray =
  conv!(similar(x, cdims(w, x, padding = pad, stride = stride)), x, w, pad = pad, stride = stride)

∇conv_data(dy::A, x::A, w::A; pad = 0, stride = 1) where A<:AbstractArray =
  ∇conv_data!(zeros(x), dy, x, w)

∇conv_filter(dy::A, x::A, w::A; pad = 0, stride = 1) where A<:AbstractArray =
  ∇conv_filter!(zeros(w), dy, x, w)

# N-D dispatch

conv!(y::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}; pad = 0, stride = 1) where T =
  conv2d!(y, x, w, padding = pad, stride = stride)

∇conv_filter!(dw::AbstractArray{T,4}, dy::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}; pad = 0, stride = 1) where T =
  conv2d_grad_w!(dw, x, w, dy, padding = pad, stride = stride)

∇conv_data!(dx::AbstractArray{T,4}, dy::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}; pad = 0, stride = 1) where T =
  conv2d_grad_w!(dx, x, w, dy, padding = pad, stride = stride)

conv!(y::AbstractArray{T,5}, x::AbstractArray{T,5}, w::AbstractArray{T,5}; pad = 0, stride = 1) where T =
  conv3d!(y, x, w, padding = pad, stride = stride)

∇conv_filter!(dw::AbstractArray{T,5}, dy::AbstractArray{T,5}, x::AbstractArray{T,5}, w::AbstractArray{T,5}; pad = 0, stride = 1) where T =
  conv3d_grad_w!(dw, x, w, dy, padding = pad, stride = stride)

∇conv_data!(dx::AbstractArray{T,5}, dy::AbstractArray{T,5}, x::AbstractArray{T,5}, w::AbstractArray{T,5}; pad = 0, stride = 1) where T =
  conv3d_grad_w!(dx, x, w, dy, padding = pad, stride = stride)

# Pooling

function pdims(dims::Dims{N}, window, padding, stride) where N
  ntuple(Val{N}) do i
    if i < N-1
      1 + (dims[i] + 2*padding[i] - window[i])÷stride[i]
    else
      dims[i]
    end
  end
end

# Interface

maxpool(x::AbstractArray, k; pad = map(_->0,k), stride = k) =
  maxpool!(similar(x, pdims(size(x), k, pad, stride)),
           x, k, pad = pad, stride = stride)

∇maxpool(dy::A, y::A, x::A, k; pad = map(_->0,k), stride = k) where A<:AbstractArray =
  ∇maxpool!(similar(x), dy, y, x, k, pad = pad, stride = stride)

# N-D dispatch

maxpool!(y::AbstractArray{<:Real,4}, x::AbstractArray{<:Real,4}, k::Dims{2}; pad = (0,0), stride = k) =
  maxpool2d!(y, x, window = k, padding = pad, stride = stride)

∇maxpool!(dx::AbstractArray{<:Real,4}, dy::AbstractArray{<:Real,4}, y::AbstractArray{<:Real,4}, x::AbstractArray{<:Real,4},
          k::Dims{2}; pad = (0,0), stride = k) =
  maxpool2d_grad!(dx, dy, y, x,
                  window = k, padding = pad, stride = stride)

maxpool!(y::AbstractArray{<:Real,5}, x::AbstractArray{<:Real,5}, k::Dims{3}; pad = (0,0), stride = k) =
  maxpool3d!(y, x, window = k, padding = pad, stride = stride)

∇maxpool!(dx::AbstractArray{<:Real,5}, dy::AbstractArray{<:Real,5}, y::AbstractArray{<:Real,5}, x::AbstractArray{<:Real,5},
          k::Dims{3}; pad = (0,0), stride = k) =
  maxpool3d_grad!(dx, dy, y, x,
                  window = k, padding = pad, stride = stride)

meanpool!(y::AbstractArray{<:Real,4}, x::AbstractArray{<:Real,4}, k::Dims{2}; pad = (0,0), stride = k) =
  meanpool2d!(y, x, window = k, padding = pad, stride = stride)

∇meanpool!(dx::AbstractArray{<:Real,4}, dy::AbstractArray{<:Real,4}, y::AbstractArray{<:Real,4}, x::AbstractArray{<:Real,4},
          k::Dims{2}; pad = (0,0), stride = k) =
  meanpool2d_grad!(dx, dy, y, x,
                   window = k, padding = pad, stride = stride)

meanpool!(y::AbstractArray{<:Real,5}, x::AbstractArray{<:Real,5}, k::Dims{3}; pad = (0,0), stride = k) =
  meanpool3d!(y, x, window = k, padding = pad, stride = stride)

∇meanpool!(dx::AbstractArray{<:Real,5}, dy::AbstractArray{<:Real,5}, y::AbstractArray{<:Real,5}, x::AbstractArray{<:Real,5},
          k::Dims{3}; pad = (0,0), stride = k) =
  meanpool3d_grad!(dx, dy, y, x,
                   window = k, padding = pad, stride = stride)
