ccall((:nnp_initialize,"libnnpack"),Void,(),)
ptp = ccall((:pthreadpool_create, :libnnpack), Ptr{Void}, (Csize_t,), 0)

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