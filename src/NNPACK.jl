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
  
  input_padding = nnp_padding(pad, pad, pad, pad)
  kernel_size = nnp_size(size(w,1), size(w,2))

  bias = zeros(Cfloat, size(x, 3))

  status = ccall((:nnp_convolution_output,:libnnpack),Cint,
                 (Cint, Csize_t, Csize_t, Csize_t, nnp_size, nnp_padding, nnp_size,
                  Ptr{Float32}, Ptr{Float32}, Ptr{Float32}, Ptr{Float32},
                  Ptr{Void}, Csize_t, Cint, Ptr{Void}, Ptr{Void}, Ptr{Void}),
                 0, size(x, 4), size(x, 3), size(y, 3), input_size, input_padding, kernel_size,
                 x, w, bias, y, C_NULL, 0, 0, C_NULL, C_NULL, C_NULL)
end