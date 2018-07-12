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

function conv!(y::Array{Float32,4}, x::Array{Float32,4}, w::Array{Float32,4};
      pad = 0, stride = 1, dilation = 1)
  input_size = nnp_size(size(x, 1), size(x, 2))
  
  input_padding = nnp_padding(pad[2], pad[1], pad[2], pad[1])
  kernel_size = nnp_size(size(w,1), size(w,2))

  bias = zeros(Cfloat, size(x, 3))

  status = ccall((:nnp_convolution_output,:libnnpack),Cint,
                 (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                  Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                  Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                 0, size(x, 4), size(x, 3), size(y, 3), input_size, input_padding, kernel_size,
                 x, w, bias, y, C_NULL, 0, 0, C_NULL, C_NULL, C_NULL)

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

function conv2d_grad_x!(dx::Array{Float32,4}, x::Array{Float32,4}, w::Array{Float32,4}, dy::Array{Float32,4};
                   padding=0, stride=1, dilation=1, mode=1, alpha=1)

  input_size = nnp_size(Csize_t(size(x,1)), Csize_t(size(x,2)))
  input_padding = nnp_padding(padding, padding, padding, padding)
  kernel_size = nnp_size(size(w,1), size(w,2))

  status = ccall((:nnp_convolution_input_gradient,:libnnpack),Cint,
                 (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                  Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                  Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                 0, size(x, 4), size(x, 3), size(dy, 3), input_size, input_padding, kernel_size,
                 dy, w, dx, C_NULL, 0, 0, C_NULL, C_NULL, C_NULL)
  if (status == 50)
      ccall((:nnp_initialize,"libnnpack"),Void,(),)
      ccall((:nnp_convolution_input_gradient,:libnnpack),Cint,
                       (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                        Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                       0, size(x, 4), size(x, 3), size(dy, 3), input_size, input_padding, kernel_size,
                       dy, w, dx, C_NULL, 0, 0, C_NULL, C_NULL, C_NULL)
  end
  return dx
end

function  conv2d_grad_w!(dw::Array{Float32,4}, x::Array{Float32,4}, w::Array{Float32,4}, dy::Array{Float32,4};
                   padding=0, stride=1, dilation=1, mode=0, alpha=1)

  input_size = nnp_size(Csize_t(size(x,1)), Csize_t(size(x,2)))
  input_padding = nnp_padding(padding, padding, padding, padding)
  kernel_size = nnp_size(size(w,1), size(w,2))

  status = ccall((:nnp_convolution_kernel_gradient,:libnnpack),Cint,
                 (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                  Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                  Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                 0, size(x, 4), size(x, 3), size(dy, 3), input_size, input_padding, kernel_size,
                 x, dy, dw, C_NULL, 0, 0, C_NULL, C_NULL, C_NULL)
  if (status == 50)
      ccall((:nnp_initialize,"libnnpack"),Void,(),)
      ccall((:nnp_convolution_kernel_gradient,:libnnpack),Cint,
                       (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                        Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                        Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                       0, size(x, 4), size(x, 3), size(dy, 3), input_size, input_padding, kernel_size,
                       x, dy, dw, C_NULL, 0, 0, C_NULL, C_NULL, C_NULL)
  end
  return dw
end