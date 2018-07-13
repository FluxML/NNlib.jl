module NNPACK

ccall((:nnp_initialize,"libnnpack"),Void,(),)
ptp = ccall((:pthreadpool_create, :libnnpack), Ptr{Void}, (Csize_t,), 0)

struct nnp_size
 width::Csize_t
 height::Csize_t
end

struct nnp_padding
  top::Csize_t
  right::Csize_t
  bottom::Csize_t
  left::Csize_t
end

function cdims(x::NTuple{N}, w::NTuple{N}, pad, stride) where N
  ntuple(Val{N}) do i
    if i < N-1
      1 + div(x[i] - w[i] + 2*pad[i], stride[i])
    elseif i == N-1
      w[N]
    else # i == N
      x[N]
    end
  end
end

head(x) = reverse(Base.tail(reverse(x)))
padtuple(x::Tuple,p::Integer) = map(_->p, head(head(x)))
padtuple(x::Tuple,p::Tuple) = p
padtuple(x::AbstractArray,p) = padtuple(size(x),p)

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

function dilation_dims(w, dilation = 1)
  N = ndims(w)
  dims_w = size(w)
  dil = psize(dilation, w)
  ntuple(N) do i
    if i < N - 1
      (dims_w[i] - 1) * dil[i] + 1
    else
      dims_w[i]
    end
  end
end

function softmax!(out::AbstractVecOrMat{T}, xs::AbstractVecOrMat{T}) where T<:AbstractFloat
  input = Cfloat.(xs)
  output = similar(input)
  ccall((:nnp_softmax_output,"libnnpack"),Void,(Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Void}), Csize_t(size(xs, 2)), Csize_t(size(xs, 1)), input, output, ptp)
  out .= output
  return out
end

function relu(x)
  if typeof(x) <: ForwardDiff.Dual || typeof(x) <: AbstractFloat
    return max(zero(x), x)
  end
  input = Cfloat.(x)
  out = x
  if length(size(x))>0
    out = similar(x)
  else
    input = [Cfloat(x)]
  end
  output = similar(input)
  ccall((:nnp_relu_output,"libnnpack"),Void,(Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Cfloat, Ptr{Void}), Csize_t(size(input, 2)), Csize_t(size(input, 1)), input, output, Cfloat(0), ptp)
  if length(size(x))>0
    out .= output
  else
    out = convert(typeof(x), output[1])
  end
  return out
end

function conv2d!(y::AbstractArray{Float32,4}, x::AbstractArray{Float32,4}, w::AbstractArray{Float32,4};
      pad = 0, stride = 1, dilation = 1, activation = 0, bias = zeros(Float32, size(x,3)))
  input_size = nnp_size(size(x, 1), size(x, 2))

  @show typeof(x)
  @show typeof(w)
  @show typeof(bias)
  input_padding = nnp_padding(pad[2], pad[1], pad[2], pad[1])
  kernel_size = nnp_size(size(w,1), size(w,2))
  status = ccall((:nnp_convolution_output,:libnnpack),Cint,
                 (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                  Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                  Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                 0, size(x, 4), size(x, 3), size(y, 3), input_size, input_padding, kernel_size,
                 x, w, bias, y, C_NULL, 0, activation, C_NULL, C_NULL, C_NULL)

  if status == 50
      ccall((:nnp_initialize,"libnnpack"),Void,(),)
      status = ccall((:nnp_convolution_output,:libnnpack),Cint,
                 (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                  Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                  Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                 0, size(x, 4), size(x, 3), size(y, 3), input_size, input_padding, kernel_size,
                 x, w, bias, y, C_NULL, 0, activation, C_NULL, C_NULL, C_NULL)
    end

  return y
end

function maxpool2d!(y::Array{Float32,4}, x::Array{Float32,4};
                    window::Dims{2}=(2,2), padding::Dims{2}=(0,0),
                    stride::Dims{2}=window)

  input_size = nnp_size(Csize_t(size(x,1)), Csize_t(size(x,2)))
  input_padding = nnp_padding(Csize_t(padding[2]), Csize_t(padding[1]), Csize_t(padding[2]), Csize_t(padding[1]))
  pooling_size = nnp_size(Csize_t(window[1]), Csize_t(window[2]))
  pooling_stride = nnp_size(Csize_t(stride[1]), Csize_t(stride[2]))

  status = ccall((:nnp_max_pooling_output,"libnnpack"),Cint,(Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size, nnp_size, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Void}), size(x, 4), size(x, 3), input_size, input_padding, pooling_size, pooling_stride, x, y, ptp)
  if (status == 50)
      ccall((:nnp_initialize,"libnnpack"),Void,(),)
      ccall((:nnp_max_pooling_output,"libnnpack"),Cint,(Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size, nnp_size, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Void}), size(x, 4), size(x, 3), input_size, input_padding, pooling_size, pooling_stride, x, y, ptp)
  end
  return y
end

function conv2d_grad_x!(dx::Array{Float32,4}, x::Array{Float32,4}, w::Array{Float32,4}, dy;
                   padding=0, stride=1, dilation=1, mode=1, alpha=1, activation = 0)
  println("Here")
  input_size = nnp_size(Csize_t(size(x,1)), Csize_t(size(x,2)))
  input_padding = nnp_padding(padding, padding, padding, padding)
  kernel_size = nnp_size(size(w,1), size(w,2))

  status = ccall((:nnp_convolution_input_gradient,:libnnpack),Cint,
                 (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                  Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                  Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                 0, size(x, 4), size(x, 3), size(dy, 3), input_size, input_padding, kernel_size,
                 dy, w, dx, C_NULL, 0, activation, C_NULL, C_NULL, C_NULL)
  if (status == 50)
      ccall((:nnp_initialize,"libnnpack"),Void,(),)
      ccall((:nnp_convolution_input_gradient,:libnnpack),Cint,
                       (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                        Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                       0, size(x, 4), size(x, 3), size(dy, 3), input_size, input_padding, kernel_size,
                       dy, w, dx, C_NULL, 0, activation, C_NULL, C_NULL, C_NULL)
  end
  return dx
end

function  conv2d_grad_w!(dw::Array{Float32,4}, x::Array{Float32,4}, w::Array{Float32,4}, dy;
                   padding=0, stride=1, dilation=1, mode=0, alpha=1, activation = 0)

  input_size = nnp_size(Csize_t(size(x,1)), Csize_t(size(x,2)))
  input_padding = nnp_padding(padding, padding, padding, padding)
  kernel_size = nnp_size(size(w,1), size(w,2))

  status = ccall((:nnp_convolution_kernel_gradient,:libnnpack),Cint,
                 (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                  Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                  Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                 0, size(x, 4), size(x, 3), size(dy, 3), input_size, input_padding, kernel_size,
                 x, dy, dw, C_NULL, 0, activation, C_NULL, C_NULL, C_NULL)
  if (status == 50)
      ccall((:nnp_initialize,"libnnpack"),Void,(),)
      ccall((:nnp_convolution_kernel_gradient,:libnnpack),Cint,
                       (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                        Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                       0, size(x, 4), size(x, 3), size(dy, 3), input_size, input_padding, kernel_size,
                       x, dy, dw, C_NULL, 0, activation, C_NULL, C_NULL, C_NULL)
  end
  return dw
end

function convo(x::A, w::A, bias; pad = 0, stride = 1, dilation = 1, activation = 0) where A<:AbstractArray
  pad_, stride_ = padtuple(x, pad), padtuple(x, stride)
  conv!(similar(x, cdims(size(x), dilation_dims(w, dilation), pad_, stride_)),
        x, w, pad = pad_, stride = stride_, dilation = dilation, activation = activation, bias = bias)
end

function conv!(y::AbstractArray{Float32,3}, x::AbstractArray{Float32,3}, w::AbstractArray{Float32,3};
               pad = 0, stride = 1, dilation = 1, activation = 0, bias = zeros(Float32, size(x,2)))
    args = map(x -> reshape(x, size(x,1),1,size(x,2),size(x,3)), (y, x, w))
    conv!(args..., pad = (pad...,0), stride = (stride...,1), dilation = (dilation...,1), activation = activation, bias = bias)
    return y
end

conv!(y::AbstractArray{Float32,4}, x::AbstractArray{Float32,4}, w::AbstractArray{Float32,4};
      pad = 0, stride = 1, dilation = 1, activation = 0, bias = zeros(Float32, size(x,3))) =
  conv2d!(y, x, w, pad = pad, stride = stride, dilation = dilation, activation = activation, bias = bias)


function ∇conv_data(dy, x, w; pad = 0, stride = 1, dilation = 1, activation = 0)
  println("here 5")
  ∇conv_data!(zeros(Float32, size(x)), dy, x, w; pad = pad, stride = stride, dilation = dilation, activation = activation)
end

function ∇conv_filter(dy, x, w; pad = 0, stride = 1, dilation = 1, activation = 0)
  ∇conv_filter!(zeros(Float32, size(w)), dy, x, w; pad = pad, stride = stride, dilation = dilation, activation = activation)
end

∇conv_filter!(dw::AbstractArray{T,4}, dy, x::AbstractArray{T,4}, w::AbstractArray{T,4};
              pad = 0, stride = 1, dilation = 1, activation = 0) where T<:AbstractFloat =
  conv2d_grad_w!(dw, x, w, dy, padding = pad, stride = stride, dilation = dilation, activation = activation)

∇conv_data!(dx::AbstractArray{T,4}, dy, x::AbstractArray{T,4}, w::AbstractArray{T,4};
            pad = 0, stride = 1, dilation = 1, activation = 0) where T<:AbstractFloat =
  conv2d_grad_x!(dx, x, w, dy, padding = pad, stride = stride, dilation = dilation, activation = activation)


function ∇conv_filter!(dw::AbstractArray{T,3}, dy,
                       x::AbstractArray{T,3}, w::AbstractArray{T,3};
                       pad = 0, stride = 1, dilation = 1, activation = 0) where T<:AbstractFloat
    args = map(x -> reshape(x, size(x,1),1,size(x,2),size(x,3)), (dw, dy, x, w))
    ∇conv_filter!(args..., pad = (pad...,0), stride = (stride...,1), dilation = (dilation...,1), activation = activation)
    return dw
end

function ∇conv_data!(dx::AbstractArray{T,3}, dy,
                     x::AbstractArray{T,3}, w::AbstractArray{T,3};
                     pad = 0, stride = 1, dilation = 1, activation = 0) where T<:AbstractFloat
    args = map(x -> reshape(x, size(x,1),1,size(x,2),size(x,3)), (dx, dy, x, w))
    ∇conv_data!(args..., pad = (pad...,0), stride = (stride...,1), dilation = (dilation..., 1), activation = activation)
    return dx
end

# end
end