ccall((:nnp_initialize,"libnnpack"),Void,(),)
ptp = ccall((:pthreadpool_create, :libnnpack), Ptr{Void}, (Csize_t,), 0)

function softmax!(out::AbstractVecOrMat{T}, xs::AbstractVecOrMat{T}) where T<:AbstractFloat
	input = Cfloat.(xs)
	out = Cfloat.(out)
	ccall((:nnp_softmax_output,"libnnpack"),Void,(Csize_t, Csize_t, Ptr{Cfloat}, Ptr{Cfloat}, Ptr{Void}), Csize_t(size(xs, 2)), Csize_t(size(xs, 1)), input, out, ptp)
	return convert(typeof(xs), out)
end
