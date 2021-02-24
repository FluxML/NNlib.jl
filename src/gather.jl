export gather, gather!

"""
    gather!(dst, src, idx)

Reverse operation of scatter, which gather data in `src` to destination according to `idx`.
For each index `k` in `idx`, assign values to `dst` according to

    dst[:,...,k...] = src[:, ..., idx[k]...]

# Arguments
- `dst`: the destination where data would be assigned to.
- `src`: the source data to be assigned.
- `idx`: the mapping for assignment from source to destination.
"""
function gather!(dst::AbstractArray{Tdst,Ndst}, 
                 src::AbstractArray{Tsrc,Nsrc}, 
                 idx::AbstractArray{NTuple{M,Int}, Nidx}) where {Tdst,Tsrc,Ndst,Nsrc,Nidx,M}
  # TODO: use M = _length(eltype(idx)) to merge the integer method into this?
  @boundscheck _gather_checkbounds(src, idx)
  Ndst - Nidx == Nsrc - M  || throw(ArgumentError(""))
  size(dst)[1:Ndst-Nidx] ==  size(src)[1:Ndst-Nidx] || throw(ArgumentError(""))
  size(dst)[Ndst-Nidx+1:end] ==  size(idx) || throw(ArgumentError(""))
  
  coldst = ntuple(i -> Colon(), Ndst - Nidx)
  colsrc = ntuple(i -> Colon(), Nsrc - M)
  @simd for k in CartesianIndices(idx)
    @inbounds view(dst, coldst..., Tuple(k)...) .= view(src, colsrc..., idx[k]...)
  end
  return dst
end

function gather!(dst::AbstractArray{Tdst,Ndst}, 
                src::AbstractArray{Tsrc,Nsrc}, 
                idx::AbstractArray{<:Integer, Nidx}) where {Tdst,Tsrc,Ndst,Nsrc,Nidx}
    
  @boundscheck _gather_checkbounds(src, idx)
  Ndst - Nidx == Nsrc - 1  || throw(ArgumentError(""))
  size(dst)[1:Ndst-Nidx] ==  size(src)[1:Ndst-Nidx] || throw(ArgumentError(""))
  size(dst)[Ndst-Nidx+1:end] ==  size(idx) || throw(ArgumentError(""))  
  coldst = ntuple(i -> Colon(), Ndst - Nidx)
  colsrc = ntuple(i -> Colon(), Nsrc - 1)
  @simd for k in CartesianIndices(idx)
    @inbounds view(dst, coldst..., k) .= view(src, colsrc..., idx[k])
  end
  return dst
end

function _gather_checkbounds(src, idx::AbstractArray{<:Integer})
  mini, maxi = extrema(idx)
  checkbounds(src, axes(src)[1:end-1]..., mini:maxi)
end

function _gather_checkbounds(src, idx::AbstractArray{<:NTuple{M,Int}}) where M
  # TODO: use M = _length(eltype(idx)) to merge the integer method into this?
  minimaxi = ntuple(M) do d
                mini = minimum(i -> i[d], idx)    
                maxi = maximum(i -> i[d], idx)    
                mini:maxi
            end
  checkbounds(src, axes(src)[1:end-M]..., minimaxi...)
end

"""
    gather(src, idx)

Non-mutating version of [`gather!`](@ref).
"""
function gather(src::AbstractArray{Tsrc, Nsrc}, 
                idx::AbstractArray{<:IntOrTuple, Nidx}) where {Tsrc, Nsrc, Nidx}
  # Ndst - Nidx == Nsrc - M  || throw(ArgumentError(""))
  # size(dst)[1:Ndst-Nidx] ==  size(src)[1:Ndst-Nidx] || throw(ArgumentError(""))
  # size(dst)[Ndst-Nidx+1:end] ==  size(idx) || throw(ArgumentError(""))
  
  M = _length(eltype(idx)) 
  dstsize = (size(src)[1:Nsrc-M]..., size(idx)...)
  dst = similar(src, eltype(src), dstsize)
  return gather!(dst, src, idx)
end

_length(::Type{<:Integer}) = 1
_length(::Type{<:NTuple{M}}) where M = M
