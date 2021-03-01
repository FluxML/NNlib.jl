export pad_constant, pad_repeat, pad_reflect, pad_zeros

"""
    pad_zeros(x, pad::Tuple; [dims])
    pad_zeros(x, pad::Int; [dims])

Pad the array `x` with zeros.
Equivalent to [`pad_constant`](@ref) with the constant equal to 0. 
"""
pad_zeros(x::AbstractArray, pad::NTuple{M,Int}; dims = 1:M÷2) where M =
  pad_constant(x, pad, 0; dims = dims)

"""
    pad_constant(x, pad::Tuple, val = 0; [dims])
    pad_constant(x, pad::Int, val = 0; [dims])

Pad the array `x` with the constant value `val`.

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)`
of some length `2n` that specifies the left and right padding size
for each of the dimensions in `dims`. If `dims` is not given, 
it defaults to the first `n` dimensions.

For integer `pad` input instead, it is applied on both sides
on every dimension in `dims`. In this case, `dims` 
defaults to the first `ndims(x)-2` dimension 
(i.e. excludes the channel and batch dimension).

See also [`pad_reflect`](@ref) and [`pad_repeat`](@ref).

```jldoctest
julia> pad_constant(reshape(1:4, 2, 2), (1, 2, 3, 4), 8)
5×9 Matrix{Int64}:
 8  8  8  8  8  8  8  8  8
 8  8  8  1  3  8  8  8  8
 8  8  8  2  4  8  8  8  8
 8  8  8  8  8  8  8  8  8
 8  8  8  8  8  8  8  8  8
````
"""
function pad_constant(x::AbstractArray, pad::NTuple{M,Int}, val = 0; 
                    dims = 1:M÷2) where M
  length(dims) == M ÷ 2 ||
    throw(ArgumentError("The number of dims should be equal to the number of padding dimensions"))
  outsize, center = pad_outsize_and_center(size(x), pad, dims)
  y = fill!(similar(x, eltype(x), outsize), val)
  y[center...] .= x
  return y
end

function pad_outsize_and_center(sz::NTuple{N,Int}, pad, dims) where {N}
  leftpad, rightpad = pad[1:2:end], pad[2:2:end]
  
  outsize = ntuple(N) do i
     k = findfirst(==(i), dims)
     if k === nothing
       return sz[i]
     else
       return sz[i] + leftpad[k] + rightpad[k]
     end
   end::NTuple{N,Int}
  
  leftcorner = ones(Int, N)
  rightcorner = collect(outsize)
  for (i, d) in enumerate(dims)
    leftcorner[d] += leftpad[i]
    rightcorner[d] -= rightpad[i]
  end
  center = ntuple(N) do i
    leftcorner[i]:rightcorner[i]
  end 
  return outsize, center
end

function rrule(::typeof(pad_constant), x::AbstractArray, pad::NTuple{M,Int}, val=0; 
              dims=1:M÷2) where M
  szx = size(x)
  y = pad_constant(x, pad, val; dims=dims)
  
  function pad_constant_pullback(Δ)
    outsize, center = pad_outsize_and_center(szx, pad, dims)
    (NO_FIELDS, @thunk(Δ[center...]), DoesNotExist(),
     @thunk(sum(Δ) - sum(Δ[center...])),)
  end
  
  return y, pad_constant_pullback
end


"""
    pad_repeat(x, pad::Tuple; [dims])
    pad_repeat(x, pad::Int; [dims])
 
Pad the array `x` repeating the values on the border.

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)`
of some length `2n` that specifies the left and right padding size
for each of the dimensions in `dims`. If `dims` is not given, 
it defaults to the first `n` dimensions.

For integer `pad` input instead, it is applied on both sides
on every dimension in `dims`. In this case, `dims` 
defaults to the first `ndims(x)-2` dimensions 
(i.e. excludes the channel and batch dimension). 

See also [`pad_reflect`](@ref) and [`pad_constant`](@ref).

```jldoctest
julia> pad_repeat(reshape(1:9, 3, 3), (1,2,3,4))
6×10 Matrix{Int64}:
 1  1  1  1  4  7  7  7  7  7
 1  1  1  1  4  7  7  7  7  7
 2  2  2  2  5  8  8  8  8  8
 3  3  3  3  6  9  9  9  9  9
 3  3  3  3  6  9  9  9  9  9
 3  3  3  3  6  9  9  9  9  9
```
"""
function pad_repeat(x::AbstractArray, pad::NTuple{M,Int}; 
                    dims=1:M÷2) where M
  length(dims) == M ÷ 2 ||
    throw(ArgumentError("The number of dims should be equal to the number of padding dimensions"))
  for (i, d) in enumerate(dims)
    x = pad_repeat(x, (pad[2i-1], pad[2i]); dims=d)
  end  
  return x
end

function pad_repeat(x::AbstractArray{F,N}, pad::NTuple{2,Int}; 
                    dims::Int = 1) where {F,N}
  lpad, rpad = pad

  xlborder = selectdim(x, dims, 1:1)
  nrepl = ntuple(i -> i == dims ? lpad : 1, N)
  xl = repeat(xlborder, outer = nrepl)

  n = size(x, dims)
  xrborder = selectdim(x, dims, n:n)
  nrepr = ntuple(i -> i == dims ? rpad : 1, N)
  xr = repeat(xrborder, outer = nrepr)

  return cat(xl, x, xr, dims = dims)
end


"""
    pad_reflect(x, pad::Tuple; [dims])
    pad_reflect(x, pad::Int; [dims])

Pad the array `x` reflecting its values across the border.

`pad` can a tuple of integers `(l1, r1, ..., ln, rn)`
of some length `2n` that specifies the left and right padding size
for each of the dimensions in `dims`. If `dims` is not given, 
it defaults to the first `n` dimensions.

For integer `pad` input instead, it is applied on both sides
on every dimension in `dims`. In this case, `dims` 
defaults to the first `ndims(x)-2` dimensions 
(i.e. excludes the channel and batch dimension).

See also [`pad_repeat`](@ref) and [`pad_constant`](@ref).

```jldoctest
julia> pad_reflect(reshape(1:9, 3, 3), (1,2,1,2))
6×6 Matrix{Int64}:
 5  2  5  8  5  2
 4  1  4  7  4  1
 5  2  5  8  5  2
 6  3  6  9  6  3
 5  2  5  8  5  2
 4  1  4  7  4  1
```
"""
function pad_reflect(x::AbstractArray, pad::NTuple{M,Int}; 
                    dims=1:M÷2) where M
  length(dims) == M ÷ 2 ||
    throw(ArgumentError("The number of dims should be equal to the number of padding dimensions"))
  for (i, d) in enumerate(dims)
    x = pad_reflect(x, (pad[2i-1], pad[2i]); dims = d)
  end  
  return x
end

function pad_reflect(x::AbstractArray{F,N}, pad::NTuple{2,Int}; 
                    dims::Int = 1) where {F,N}
  lpad, rpad = pad
  
  n = size(x, dims)
  xl = selectdim(x, dims, lpad+1:-1:2)
  xr = selectdim(x, dims, n-1:-1:n-rpad)
  # Alternative selection, not sure which is faster...
  # xl = reverse(selectdim(x, dims, 2:lpad+1), dims)
  # xr = reverse(selectdim(x, dims, n-rpad:n-1), dims)
  return cat(xl, x, xr, dims = dims)
end

# convenience methods for symmetric and homogeneous padding
pad_repeat(x::AbstractArray{F,N}, pad::Int; dims=1:N-2) where {F,N} =
  pad_repeat(x, ntuple(_ -> pad, 2length(dims)); dims = dims)
pad_reflect(x::AbstractArray{F,N}, pad::Int; dims=1:N-2) where {F,N} =
  pad_reflect(x, ntuple(_ -> pad, 2length(dims)); dims = dims)
pad_zeros(x::AbstractArray{F,N}, pad::Int; dims=1:N-2) where {F,N} =
  pad_zeros(x, ntuple(_ -> pad, 2length(dims)); dims = dims)
pad_constant(x::AbstractArray{F,N}, pad::Int, val=0; dims=1:N-2) where {F,N} =
  pad_constant(x, ntuple(_ -> pad, 2length(dims)), val; dims = dims)
