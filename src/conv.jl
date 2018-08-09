Dims{N} = NTuple{N,Integer}

include("impl/pool.jl")
include("impl/conv.jl")

# Convolutions

function cdims(x::NTuple{N}, w::NTuple{N}, pad, stride) where N
  ntuple(Val(N)) do i
    if i < N-1
      1 + div(x[i] - w[i] + 2*pad[i], stride[i])
    elseif i == N-1
      w[N]
    else # i == N
      x[N]
    end
  end
end

# Interface

head(x) = reverse(Base.tail(reverse(x)))
padtuple(x::Tuple,p::Integer) = map(_->p, head(head(x)))
padtuple(x::Tuple,p::Tuple) = p
padtuple(x::AbstractArray,p) = padtuple(size(x),p)

function conv(x::A, w::A; pad = 0, stride = 1, dilation = 1) where A<:AbstractArray
  pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
  conv!(similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_)),
        x, w, pad = pad_, stride = stride_, dilation = dilation)
end

∇conv_data(dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1) where A<:AbstractArray =
  ∇conv_data!(zero(x), dy, x, w; pad = pad, stride = stride, dilation = dilation)

∇conv_filter(dy::A, x::A, w::A; pad = 0, stride = 1, dilation = 1) where A<:AbstractArray =
  ∇conv_filter!(zero(w), dy, x, w; pad = pad, stride = stride, dilation = dilation)

# N-D dispatch

function conv!(y::AbstractArray{T,3}, x::AbstractArray{T,3}, w::AbstractArray{T,3};
               pad = 0, stride = 1, dilation = 1) where T
    args = map(x -> reshape(x, size(x,1),1,size(x,2),size(x,3)), (y, x, w))
    conv!(args..., pad = (pad...,0), stride = (stride...,1), dilation = (dilation...,1))
    return y
end

function ∇conv_filter!(dw::AbstractArray{T,3}, dy::AbstractArray{T,3},
                       x::AbstractArray{T,3}, w::AbstractArray{T,3};
                       pad = 0, stride = 1, dilation = 1) where T
    args = map(x -> reshape(x, size(x,1),1,size(x,2),size(x,3)), (dw, dy, x, w))
    ∇conv_filter!(args..., pad = (pad...,0), stride = (stride...,1), dilation = (dilation...,1))
    return dw
end

function ∇conv_data!(dx::AbstractArray{T,3}, dy::AbstractArray{T,3},
                     x::AbstractArray{T,3}, w::AbstractArray{T,3};
                     pad = 0, stride = 1, dilation = 1) where T
    args = map(x -> reshape(x, size(x,1),1,size(x,2),size(x,3)), (dx, dy, x, w))
    ∇conv_data!(args..., pad = (pad...,0), stride = (stride...,1), dilation = (dilation..., 1))
    return dx
end

conv!(y::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4};
      pad = 0, stride = 1, dilation = 1) where T =
  conv2d!(y, x, w, padding = pad, stride = stride, dilation = dilation)

∇conv_filter!(dw::AbstractArray{T,4}, dy::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4};
              pad = 0, stride = 1, dilation = 1) where T =
  conv2d_grad_w!(dw, x, w, dy, padding = pad, stride = stride, dilation = dilation)

∇conv_data!(dx::AbstractArray{T,4}, dy::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4};
            pad = 0, stride = 1, dilation = 1) where T =
  conv2d_grad_x!(dx, x, w, dy, padding = pad, stride = stride, dilation = dilation)

conv!(y::AbstractArray{T,5}, x::AbstractArray{T,5}, w::AbstractArray{T,5};
      pad = 0, stride = 1, dilation = 1) where T =
  conv3d!(y, x, w, padding = pad, stride = stride, dilation = dilation)

∇conv_filter!(dw::AbstractArray{T,5}, dy::AbstractArray{T,5}, x::AbstractArray{T,5}, w::AbstractArray{T,5};
              pad = 0, stride = 1, dilation = 1) where T =
  conv3d_grad_w!(dw, x, w, dy, padding = pad, stride = stride, dilation = dilation)

∇conv_data!(dx::AbstractArray{T,5}, dy::AbstractArray{T,5}, x::AbstractArray{T,5}, w::AbstractArray{T,5};
            pad = 0, stride = 1, dilation = 1) where T =
  conv3d_grad_x!(dx, x, w, dy, padding = pad, stride = stride, dilation = dilation)

# Depthwise Conv

function dcdims(x::NTuple{4,Int}, w::NTuple{4,Int}, pad, stride)
  ((x[1] + 2 * pad[1] - w[1])÷stride[1] + 1,(x[2] + 2 * pad[2] - w[2])÷stride[2] + 1,w[3]*w[4],x[4])
end

function depthwiseconv(x::A, w::A; pad = 0, stride = 1) where A<:AbstractArray
  pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
  depthwiseconv!(similar(x, dcdims(size(x), size(w), pad_, stride_)), x, w, pad = pad_, stride = stride_)
end

depthwiseconv!(y::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4};
      pad = 0, stride = 1) where T =
  depthwiseconv2d!(y, x, w, padding = pad, stride = stride)

∇depthwiseconv_data(dy::A, x::A, w::A; pad = 0, stride = 1) where A<:AbstractArray =
  ∇depthwiseconv_data!(zero(x), dy, x, w; pad = pad, stride = stride)

∇depthwiseconv_filter(dy::A, x::A, w::A; pad = 0, stride = 1) where A<:AbstractArray =
  ∇depthwiseconv_filter!(zero(w), dy, x, w; pad = pad, stride = stride)

∇depthwiseconv_filter!(dw::AbstractArray{T,4}, dy::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4};
              pad = 0, stride = 1) where T =
  depthwiseconv2d_grad_w!(dw, x, w, dy, padding = pad, stride = stride)

∇depthwiseconv_data!(dx::AbstractArray{T,4}, dy::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4};
            pad = 0, stride = 1) where T =
  depthwiseconv2d_grad_x!(dx, x, w, dy, padding = pad, stride = stride)

# Pooling

function pdims(dims::Dims{N}, window, padding, stride) where N
  ntuple(Val(N)) do i
    if i < N-1
      1 + (dims[i] + 2*padding[i] - window[i])÷stride[i]
    else
      dims[i]
    end
  end
end

expand(::Type{Val{N}}, i::Integer) where N = ntuple(_ -> i, Val(N))
expand(::Type{Val{N}}, i::NTuple{N, Integer}) where N = i

# Interface

maxpool(x::AbstractArray, k; pad = map(_->0,k), stride = k) =
  maxpool!(similar(x, pdims(size(x), k, expand(Val{length(k)}, pad),
            expand(Val{length(k)}, stride))), x, k, pad = expand(Val{length(k)}, pad),
            stride = expand(Val{length(k)}, stride))

maxpool!(y::A, x::A, k; kw...) where A<:AbstractArray =
  maxpool_cpu!(y, x, k; kw...)

∇maxpool(dy::A, y::A, x::A, k; pad = map(_->0,k), stride = k) where A<:AbstractArray =
  ∇maxpool!(similar(x), dy, y, x, k, pad = expand(Val{length(k)}, pad),
            stride = expand(Val{length(k)}, stride))

∇maxpool!(dx::A, dy::A, y::A, x::A, k; kw...) where A<:AbstractArray =
  ∇maxpool_cpu!(dx, dy, y, x, k; kw...)

meanpool(x::AbstractArray, k; pad = map(_->0,k), stride = k) =
  meanpool!(similar(x, pdims(size(x), k, expand(Val{length(k)}, pad),
            expand(Val{length(k)}, stride))), x, k, pad = expand(Val{length(k)}, pad),
            stride = expand(Val{length(k)}, stride))

meanpool!(y::A, x::A, k; kw...) where A<:AbstractArray =
  meanpool_cpu!(y, x, k; kw...)

∇meanpool(dy::A, y::A, x::A, k; pad = map(_->0,k), stride = k) where A<:AbstractArray =
  ∇meanpool!(similar(x), dy, y, x, k, pad = expand(Val{length(k)}, pad),
            stride = expand(Val{length(k)}, stride))

∇meanpool!(dx::A, dy::A, y::A, x::A, k; kw...) where A<:AbstractArray =
  ∇meanpool_cpu!(dx, dy, y, x, k; kw...)

# N-D dispatch
# We use a separate function to avoid ambiguity issues
# (more specific array types vs. more specific dimensions)

maxpool_cpu!(y::A, x::A, k::Dims{2}; pad = (0,0), stride = k) where A<:AbstractArray{<:Real,4} =
  maxpool2d!(y, x, window = k, padding = pad, stride = stride)

∇maxpool_cpu!(dx::AbstractArray{<:Real,4}, dy::AbstractArray{<:Real,4}, y::AbstractArray{<:Real,4}, x::AbstractArray{<:Real,4},
              k::Dims{2}; pad = (0,0), stride = k) =
  maxpool2d_grad!(dx, dy, y, x,
                  window = k, padding = pad, stride = stride)

maxpool_cpu!(y::AbstractArray{<:Real,5}, x::AbstractArray{<:Real,5}, k::Dims{3}; pad = (0,0), stride = k) =
  maxpool3d!(y, x, window = k, padding = pad, stride = stride)

∇maxpool_cpu!(dx::AbstractArray{<:Real,5}, dy::AbstractArray{<:Real,5}, y::AbstractArray{<:Real,5}, x::AbstractArray{<:Real,5},
              k::Dims{3}; pad = (0,0), stride = k) =
  maxpool3d_grad!(dx, dy, y, x,
                  window = k, padding = pad, stride = stride)

meanpool_cpu!(y::AbstractArray{<:Real,4}, x::AbstractArray{<:Real,4}, k::Dims{2}; pad = (0,0), stride = k) =
  meanpool2d!(y, x, window = k, padding = pad, stride = stride)

∇meanpool_cpu!(dx::AbstractArray{<:Real,4}, dy::AbstractArray{<:Real,4}, y::AbstractArray{<:Real,4}, x::AbstractArray{<:Real,4},
              k::Dims{2}; pad = (0,0), stride = k) =
  meanpool2d_grad!(dx, dy, y, x,
                   window = k, padding = pad, stride = stride)

meanpool_cpu!(y::AbstractArray{<:Real,5}, x::AbstractArray{<:Real,5}, k::Dims{3}; pad = (0,0), stride = k) =
  meanpool3d!(y, x, window = k, padding = pad, stride = stride)

∇meanpool_cpu!(dx::AbstractArray{<:Real,5}, dy::AbstractArray{<:Real,5}, y::AbstractArray{<:Real,5}, x::AbstractArray{<:Real,5},
              k::Dims{3}; pad = (0,0), stride = k) =
  meanpool3d_grad!(dx, dy, y, x,
                   window = k, padding = pad, stride = stride)

# Deprecated 0.3

export conv2d, maxpool2d, avgpool2d

@deprecate conv2d(x, w; kw...) NNlib.conv(x, w; kw...)
@deprecate maxpool2d(x::AbstractArray{<:Real,4}, k::Integer) maxpool(x, (k,k))
@deprecate meanpool2d(x::AbstractArray{<:Real,4}, k::Integer) meanpool(x, (k,k))
