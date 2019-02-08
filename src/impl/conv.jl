# convert padding etc. size to an Int array of the right dimension
function psize(p, x)
  nd = ndims(x)-2
  if isa(p,Number)
    ntuple(_->Int(p), nd)
  elseif length(p)==nd
    tuple(p...)
  else
    throw(DimensionMismatch("psize: $p $nd"))
  end
end

# Type system-level information about convolution dimensions. Critical for things like
# im2col_2d!() to generate efficient code.
struct ConvDims{img, kernel, channels, stride, padding, dilation, flipkernel} end
img_size(c::ConvDims{I,K,C,S,P,D,F}) where {I, K, C, S, P, D, F} = I

# Calculate the output dimensions of this convolution
function output_size(c::ConvDims{I,K,C,S,P,D,F}) where {I, K, C, S, P, D, F}
    O_w = div(I[1] + P[1] + P[2] - (K[1] - 1) * D[1] - 1, S[1]) + 1
    O_h = div(I[2] + P[3] + P[4] - (K[1] - 1) * D[1] - 1, S[1]) + 1
    return (O_w, O_h)
end
kernel_size(c::ConvDims{I,K,C,S,P,D,F}) where {I, K, C, S, P, D, F} = K
img_channels(c::ConvDims{I,K,C,S,P,D,F}) where {I, K, C, S, P, D, F} = C
stride(c::ConvDims{I,K,C,S,P,D,F}) where {I, K, C, S, P, D, F} = S
padding(c::ConvDims{I,K,C,S,P,D,F}) where {I, K, C, S, P, D, F} = P
dilation(c::ConvDims{I,K,C,S,P,D,F}) where {I, K, C, S, P, D, F} = D
flipkernel(c::ConvDims{I,K,C,S,P,D,F}) where {I, K, C, S, P, D, F} = F

function im2col_2d!(img::AbstractArray{T,3}, col::AbstractArray{T,2}, cdims::ConvDims) where T
  width, height = img_size(cdims)
  kernel_w, kernel_h = kernel_size(cdims)
  channels = img_channels(cdims)
  pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi = padding(cdims)
  dil_w, dil_h = dilation(cdims)
  stride_w, stride_h = stride(cdims)
  width_col, height_col = output_size(cdims)

  if flipkernel(cdims)
    flipk = (w, h) -> (kernel_w - w + 1, kernel_h - h + 1)
  else
    flipk = (w, h) -> (w, h)
  end

  # Reshape col for easy access.
  col_reshaped = reshape(col, (width_col, height_col, kernel_w, kernel_h, channels))

  # Let us first calculate the number of rows/columns within which we must zero out some
  # portion of the image patches we're copying over.  Note the subtractions on the `_hi`
  # variants are due to us needing to account for padding that is completely ignored due
  # to stride/dilation/kernel size combinations.
  spill_w_lo = ceil(Int, pad_w_lo/stride_w)
  spill_w_hi = width_col - div(width + pad_w_lo - (kernel_w - 1)*dil_w, stride_w)
  spill_h_lo = ceil(Int, pad_h_lo/stride_h)
  spill_h_hi = height_col - div(height + pad_h_lo - (kernel_h - 1)*dil_h, stride_h)
  spill_w_hi_abs = width_col - spill_w_hi + 1
  spill_h_hi_abs = height_col - spill_h_hi + 1

  # First, a helper function to project from output (w, h) to input (input_w, input_h)
  project(idx, stride, pad) = (idx - 1)*stride - pad + 1

  # These are the regions we're going to have to run with cognizance of padding
  padded_regions = (
    (1:width_col,                           1:spill_h_lo),
    (1:spill_w_lo,             (spill_h_lo+1):(spill_h_hi_abs-1)),
    (spill_w_hi_abs:width_col, (spill_h_lo+1):(spill_h_hi_abs-1)),
    (1:width_col,              spill_h_hi_abs:height_col),
  )

  # We begin by copying the central region of the image which requires no padding at all.
  # Eliminating the branches of the fully generalized version below gives us a nice
  # speedup on the majority of the data.
  for c in 1:channels
    for kh in 1:kernel_h
      for kw in 1:kernel_w
        for h in (spill_h_lo+1):(height_col - spill_h_hi)
          input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h

          @inbounds for w in (spill_w_lo+1):(width_col - spill_w_hi)
            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w
            col_reshaped[w, h, flipk(kw, kh)..., c] = img[input_kw, input_kh, c]
          end
        end
      end
    end
  end

  # For each "padded region", we run the fully general version
  for (w_region, h_region) in padded_regions
    for c in 1:channels
      for kh in 1:kernel_h
        for kw in 1:kernel_w
          @inbounds for h in h_region
            input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h

            # If this column is off the edge, then deal with the entire thing
            # in one fell swoop, like a ravenous flock of crows.  CAW CAW.
            if input_kh <= 0 || input_kh > height
              for w in w_region
                col_reshaped[w, h, flipk(kw, kh)..., c] = zero(eltype(col_reshaped))
              end
              continue
            end

            @inbounds for w in w_region
              input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w

              # If this pixel is off the edge of the map, clear it out.
              if input_kw <= 0 || input_kw > width
                col_reshaped[w, h, flipk(kw, kh)..., c] = zero(eltype(col_reshaped))
                continue
              end

              # Copy the data over
              col_reshaped[w, h, flipk(kw, kh)..., c] = img[input_kw, input_kh, c]
            end
          end
        end
      end
    end
  end
end

function col2im_2d!(col::AbstractArray{T,2}, img::AbstractArray{T,3}, width::Int, height::Int,
  channels::Int, kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int, stride_w::Int,
  stride_h::Int, dil_w::Int, dil_h::Int, mode::Int) where T

  height_col = div(height + 2pad_h - (kernel_h - 1) * dil_h - 1, stride_h) + 1
  width_col = div(width + 2pad_w - (kernel_w - 1) * dil_w - 1, stride_w) + 1
  channels_col = channels * kernel_h * kernel_w

  fill!(img, 0)
  #pragma omp parallel for
  for c = 1:channels_col
    w_offset = (c - 1) % kernel_w
    h_offset = div(c - 1,  kernel_w) % kernel_h
    c_im = div(c - 1, kernel_h * kernel_w)
    if mode == 0
      w_offset = kernel_w - 1 - w_offset
      h_offset = kernel_h - 1 - h_offset
    end
    for h = 1:height_col, w = 1:width_col
      h_pad = (h - 1) * stride_h - pad_h + h_offset * dil_h
      w_pad = (w - 1) * stride_w - pad_w + w_offset * dil_w
      if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
        cval::T = col[((c - 1) * height_col + h - 1) * width_col + w]
        img[(c_im * height + h_pad) * width + w_pad + 1] += cval
      end
    end
  end
end

function im2col_3d!(img::AbstractArray{T,4}, col::AbstractArray{T,2}, width::Int, height::Int, depth::Int,
  channels::Int, kernel_w::Int, kernel_h::Int, kernel_d::Int, pad_w::Int, pad_h::Int, pad_d::Int,
  stride_w::Int, stride_h::Int, stride_d::Int, dil_w::Int, dil_h::Int, dil_d::Int, mode::Int) where T

  height_col = div(height + 2pad_h - (kernel_h - 1) * dil_h - 1, stride_h) + 1
  width_col = div(width + 2pad_w - (kernel_w - 1) * dil_w - 1, stride_w) + 1
  depth_col = div(depth + 2pad_d - (kernel_d - 1) * dil_d - 1, stride_d) + 1
  channels_col = channels * kernel_h * kernel_w * kernel_d


  #pragma omp parallel for
  for c = 1:channels_col
    w_offset = (c - 1) % kernel_w
    h_offset = div(c - 1, kernel_w) % kernel_h
    d_offset = div(c - 1, kernel_w * kernel_h) % kernel_d
    c_im = div(c - 1, kernel_w * kernel_h * kernel_d)
    if mode == 0
      w_offset = kernel_w - 1 - w_offset
      h_offset = kernel_h - 1 - h_offset
      d_offset = kernel_d - 1 - d_offset
    end
    for d = 1:depth_col, h = 1:height_col, w = 1:width_col
      d_pad = (d - 1) * stride_d - pad_d + d_offset * dil_d
      h_pad = (h - 1) * stride_h - pad_h + h_offset * dil_h
      w_pad = (w - 1) * stride_w - pad_w + w_offset * dil_w
      if d_pad >= 0 && d_pad < depth && h_pad >= 0 && h_pad < height &&
        w_pad >= 0 && w_pad < width
        col[(((c - 1) * depth_col + d - 1) * height_col + h - 1) * width_col + w] =
    	    img[((c_im * depth + d_pad) * height + h_pad) * width + w_pad + 1]
    	else
    	  col[(((c - 1) * depth_col + d - 1) * height_col + h - 1) * width_col + w] = 0
      end
    end
  end
end

function col2im_3d!(col::AbstractArray{T,2}, img::AbstractArray{T,4}, width::Int, height::Int,
  depth::Int, channels::Int, kernel_w::Int, kernel_h::Int, kernel_d::Int,
  pad_w::Int, pad_h::Int, pad_d::Int, stride_w::Int, stride_h::Int, stride_d::Int,
  dil_w::Int, dil_h::Int, dil_d::Int, mode::Int) where T

  height_col = div(height + 2pad_h - (kernel_h - 1) * dil_h - 1, stride_h) + 1
  width_col = div(width + 2pad_w - (kernel_w - 1) * dil_w - 1, stride_w) + 1
  depth_col = div(depth + 2pad_d - (kernel_d - 1) * dil_d - 1, stride_d) + 1
  channels_col = channels * kernel_h * kernel_w * kernel_d

  fill!(img, 0)
  #pragma omp parallel for
  for c = 1:channels_col
    w_offset = (c - 1) % kernel_w;
    h_offset = div(c - 1, kernel_w) % kernel_h
    d_offset = div(c - 1, kernel_w * kernel_h) % kernel_d
    c_im = div(c - 1, kernel_h * kernel_w * kernel_d)

    if mode == 0
      w_offset = kernel_w - 1 - w_offset
      h_offset = kernel_h - 1 - h_offset
      d_offset = kernel_d - 1 - d_offset
    end

    for d = 1:depth_col, h = 1:height_col, w = 1:width_col
      d_pad = (d - 1) * stride_d - pad_d + d_offset * dil_d
    	h_pad = (h - 1) * stride_h - pad_h + h_offset * dil_h
    	w_pad = (w - 1) * stride_w - pad_w + w_offset * dil_w
    	if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width &&
        d_pad >= 0 && d_pad < depth
    	  cval::T = col[(((c - 1) * depth_col + d - 1) * height_col + h - 1) * width_col + w]
    	  iidx = ((c_im * depth + d_pad) * height + h_pad) * width + w_pad + 1
              #pragma omp atomic
    	  img[iidx] += cval
    	end
    end
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

function im2col_dims(w,y)
    N = ndims(y)
    r,c = 1,1
    for i=1:N-2
        r *= size(y,i)
        c *= size(w,i)
    end
    c *= size(w,N-1)
    return (r, c)
end

function im2col_dims(w::NTuple{4, Int}, y)
    N = ndims(y)
    r,c = 1,1
    for i=1:N-2
        r *= size(y,i)
        c *= w[i]
    end
    c *= w[N-1]
    return (r, c)
end

function depthwiseconv2d!(y::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4};
                  padding = 0, stride = 1, mode = 0, alpha = T(1)) where T
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,Cm,Cw = size(w) # Cm = Channel Multiplier
    @assert Cx == Cw DimensionMismatch()
    Wy,Hy,Cy,Ny = size(y) # Cy = Cw * Cm
    dims_w = (Ww,Hw,Cw,Cm*Cw)
    x2dims = im2col_dims(dims_w,y)
    x2 = similar(x, x2dims)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    M,N,K,Y = Wy*Hy,Cm,Ww*Hw,Wy*Hy*Cm
    yidx = 1
    @inbounds for i in 1:Nx
        im2col2d!(dims_w, x, x2, i, p1, p2, s1, s2, mode)
        @inbounds for j in 1:Cx
            gemm!('N','N',M,N,K,alpha,pointer(x2,(j-1)*M*K+1),pointer(w,(j-1)*K*N+1),T(0),pointer(y,yidx))
            yidx += Y
        end
    end
    return y
end

function depthwiseconv2d_grad_w!(dw::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}, dy::AbstractArray{T,4};
        padding=0, stride=1, mode=0, alpha=1) where T
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,Cm,Cw = size(w) # Cm = Channel Multiplier
    @assert Cx == Cw DimensionMismatch()
    Wy,Hy,Cy,Ny = size(dy) # Cy = Cw * Cm
    @assert Cy == Cw * Cm DimensionMismatch()
    dims_w = (Ww,Hw,Cw,Cm*Cw)
    x2dims = im2col_dims(dims_w,dy)
    x2 = similar(x, x2dims)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    M,N,K,Y,W = Ww*Hw,Cm,Wy*Hy,Wy*Hy*Cm*Cx,Ww*Hw*Cm
    alpha,beta = T(alpha),T(1)
    dyidx = 1
    @inbounds for i in 1:Nx
        im2col2d!(dims_w, x, x2, i, p1, p2, s1, s2, mode)
        dwidx = 1
        @inbounds for j in 1:Cx
            gemm!('T','N',M,N,K,alpha,pointer(x2,(j-1)*M*K+1),pointer(dy,dyidx+(j-1)*K*N),beta,pointer(dw,dwidx))
            dwidx += W
        end
        dyidx += Y
    end
    return dw
end

function depthwiseconv2d_grad_x!(dx::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}, dy::AbstractArray{T,4};
                   padding=0, stride=1, mode=0, alpha=1) where T
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,Cm,Cw = size(w) # Cm = Channel Multiplier
    @assert Cx == Cw DimensionMismatch()
    Wy,Hy,Cy,Ny = size(dy) # Cy = Cw * Cm
    @assert Cy == Cw * Cm DimensionMismatch()
    dims_w = (Ww,Hw,Cw,Cm*Cw)
    x2dims = im2col_dims(dims_w,dy)
    x2 = similar(x, x2dims)
    M,N,K,Y,W = Wy*Hy,Ww*Hw,Cm,Wy*Hy*Cm*Cx,Ww*Hw*Cm
    alpha,beta = T(alpha),T(0)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    dyidx = 1
    @inbounds for i in 1:Nx
        @inbounds for j in 1:Cx
            gemm!('N','T',M,N,K,alpha,pointer(dy,dyidx+(j-1)*K*M),pointer(w,(j-1)*K*N+1),beta,pointer(x2,(j-1)*M*N+1))
        end
        col2im2d!(dims_w,dx,x2,i,p1,p2,s1,s2,mode)
        dyidx += Y
    end
    return dx
end

function conv2d!(y::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4},
                 cdims::ConvDims; alpha=T(1)) where T
    Wx, Hx = img_size(cdims)
    Ww, Hw = kernel_size(cdims)
    Wy, Hy = output_size(cdims)
    Cx = img_channels(cdims)
    M, N, K, Y = Wy*Hy, size(y,4), Ww*Hw*Cx, Wy*Hy*size(y, 4)

    x2 = similar(x, im2col_dims(w, y))
    @inbounds for n in 1:size(x,4)
        im2col_2d!(view(x, :, :, :, n), x2, cdims)
        gemm!('N','N',M,N,K,alpha,pointer(x2),pointer(w),T(0),pointer(y,(n - 1)*Y + 1))
    end
    return y
end

function conv2d!(y::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4};
               padding=0, stride=1, dilation=1, mode=0, alpha=T(1)) where T
    if mode != 0 && mode != 1
        throw(ArgumentError("conv2d only supports mode=0 or 1."))
    end
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)

    # Check that the number of channels in `x` matches the number of channels in each
    # kernel of `w`.  IF it doesn't, throw a DimensionMismatch()
    if Cx != C1
        throw(DimensionMismatch())
    end
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    (d1,d2) = psize(dilation, x)

    cdims = ConvDims{(Wx,Hx),(Ww,Hw),Cx,(s1,s2),(p1,p1,p2,p2),(d1,d2), mode == 0}()
    return conv2d!(y, x, w, cdims; alpha=alpha)
end

function conv2d_grad_w!(dw::AbstractArray{T,4}, x::AbstractArray{T,4}, dy::AbstractArray{T,4};
                   padding=0, stride=1, dilation=1, mode=0, alpha=1) where T
    # dw = x'*dy
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(dw)
    Wy,Hy,Cy,Ny = size(dy)
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    @assert Cx==C1 && Cy==C2 && Ny==Nx
    x2dims = im2col_dims(dw,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Cy,Ww*Hw*Cx,Cy,Wy*Hy
    alpha,beta = T(alpha),T(1)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    (d1,d2) = psize(dilation,x)
    dyi = 1
    @inbounds for n in 1:Nx
        im2col2d!(dw, x, x2, n, p1, p2, s1, s2, d1, d2, mode)
        gemm!('T','N',M,N,K,alpha,pointer(x2),pointer(dy,dyi),beta,pointer(dw))
        dyi += Y
    end
    return dw
end

function conv2d_grad_x!(dx::AbstractArray{T,4}, w::AbstractArray{T,4}, dy::AbstractArray{T,4};
                   padding=0, stride=1, dilation=1, mode=0, alpha=1) where T
    # dx = dy*w'
    Wx,Hx,Cx,Nx = size(dx)
    Ww,Hw,C1,C2 = size(w)
    Wy,Hy,Cy,Ny = size(dy)
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    @assert Cx==C1 && Cy==C2 && Ny==Nx
    x2dims = im2col_dims(w,dy)
    x2 = similar(dx, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Cy,Wy*Hy,Ww*Hw*Cx,Cy
    alpha,beta = T(alpha),T(0)
    (p1,p2) = psize(padding,dx)
    (s1,s2) = psize(stride,dx)
    (d1,d2) = psize(dilation,dx)
    dyi = 1
    @inbounds for n in 1:Nx
        gemm!('N','T',M,N,K,alpha,pointer(dy,dyi),pointer(w),beta,pointer(x2))
        col2im2d!(w,dx,x2,n,p1,p2,s1,s2,d1,d2,mode)
        dyi += Y
    end
    return dx
end

function im2col2d!(w::NTuple{4,Int}, x::AbstractArray{T,4}, x2::AbstractArray{T,2},
                 n::Int, p1::Int, p2::Int, s1::Int, s2::Int, mode::Int) where T
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = w
    xn = view(x, :, :, :, n)
    cdims = ConvDims{(Wx,Hx),(Ww,Hw),Cx,(s1,s2),(p1,p1,p2,p2),(1,1), mode == 0}()
    im2col_2d!(xn,x2,cdims)
    return x2
end

function im2col2d!(w::AbstractArray{T,4}, x::AbstractArray{T,4}, x2::AbstractArray{T,2},
                 n::Int, p1::Int, p2::Int, s1::Int, s2::Int, d1::Int, d2::Int, mode::Int) where T
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    xn = view(x, :, :, :, n)
    cdims = ConvDims{(Wx,Hx),(Ww,Hw),Cx,(s1,s2),(p1,p1,p2,p2),(d1,d2), mode == 0}()
    im2col_2d!(xn,x2,cdims)
    return x2
end

function col2im2d!(w::NTuple{4,Int}, x::AbstractArray{T,4}, x2::AbstractArray{T,2},
                 n::Int, p1::Int, p2::Int, s1::Int, s2::Int, mode::Int) where T
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = w
    xn = view(x, :, :, :, n)
    col2im_2d!(x2,xn,Wx,Hx,Cx,Ww,Hw,p1,p2,s1,s2,1,1,mode)
    return x
end

function col2im2d!(w::AbstractArray{T,4}, x::AbstractArray{T,4}, x2::AbstractArray{T,2},
                 n::Int, p1::Int, p2::Int, s1::Int, s2::Int, d1::Int, d2::Int, mode::Int) where T
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    xn = view(x, :, :, :, n)
    col2im_2d!(x2,xn,Wx,Hx,Cx,Ww,Hw,p1,p2,s1,s2,d1,d2,mode)
    return x
end

function conv3d!(y::AbstractArray{T,5}, x::AbstractArray{T,5}, w::AbstractArray{T,5};
               padding=0, stride=1, dilation = 1, mode=0, alpha=T(1)) where T
    if mode != 0 && mode != 1; throw(ArgumentError("conv3d only supports mode=0 or 1.")); end
    Wx,Hx,Dx,Cx,Nx = size(x)
    Ww,Hw,Dw,C1,C2 = size(w)
    if Cx!=C1; throw(DimensionMismatch()); end
    Wy,Hy,Dy,Cy,Ny = size(y)
    # @assert Cy==C2 && Ny==Nx
    x2dims = im2col_dims(w,y)
    x2 = similar(x, x2dims)
    (p1,p2,p3) = psize(padding,x)
    (s1,s2,s3) = psize(stride,x)
    (d1,d2,d3) = psize(dilation,x)
    M,N,K,Y = Wy*Hy*Dy,Cy,Ww*Hw*Dw*Cx,Wy*Hy*Dy*Cy
    yidx = 1
    W = reshape(w, (size(w, 1),:,C1,C2))
    @inbounds for n in 1:Nx
        im2col3d!(w, x, x2, n, p1, p2, p3, s1, s2, s3, d1, d2, d3, mode)
        gemm!('N','N',M,N,K,alpha,pointer(x2),pointer(W),T(0),pointer(y,yidx))
        yidx += Y
    end
    return y
end

function conv3d_grad_w!(dw::AbstractArray{T,5}, x::AbstractArray{T,5}, dy::AbstractArray{T,5};
                   padding=0, stride=1, dilation = 1, mode=0, alpha=1) where T
    # dw = x'*dy
    Wx,Hx,Dx,Cx,Nx = size(x)
    Ww,Hw,Dw,C1,C2 = size(dw)
    Wy,Hy,Dy,Cy,Ny = size(dy)
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    @assert Cx==C1 && Cy==C2 && Ny==Nx
    x2dims = im2col_dims(dw,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Dy*Cy,Ww*Hw*Dw*Cx,Cy,Wy*Hy*Dy
    alpha,beta = T(alpha),T(1)
    (p1,p2,p3) = psize(padding,x)
    (s1,s2,s3) = psize(stride,x)
    (d1,d2,d3) = psize(dilation,x)
    dyi = 1
    @inbounds for n in 1:Nx
        im2col3d!(dw, x, x2, n, p1, p2, p3, s1, s2, s3, d1, d2, d3, mode)
        gemm!('T','N',M,N,K,alpha,pointer(x2),pointer(dy,dyi),beta,pointer(dw))
        dyi += Y
    end
    return dw
end

function conv3d_grad_x!(dx::AbstractArray{T,5}, w::AbstractArray{T,5}, dy::AbstractArray{T,5};
                   padding=0, stride=1, dilation = 1, mode=0, alpha=1) where T
    # dx = dy*w'
    Wx,Hx,Dx,Cx,Nx = size(dx)
    Ww,Hw,Dw,C1,C2 = size(w)
    Wy,Hy,Dy,Cy,Ny = size(dy)
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    @assert Cx==C1 && Cy==C2 && Ny==Nx
    x2dims = im2col_dims(w,dy)
    x2 = similar(dx, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Dy*Cy,Wy*Hy*Dy,Ww*Hw*Dw*Cx,Cy
    alpha,beta = T(alpha),T(0)
    (p1,p2,p3) = psize(padding,dx)
    (s1,s2,s3) = psize(stride,dx)
    (d1,d2,d3) = psize(dilation,dx)
    dyi = 1
    @inbounds for n in 1:Nx
        gemm!('N','T',M,N,K,alpha,pointer(dy,dyi),pointer(w),beta,pointer(x2))
        col2im3d!(w,dx,x2,n,p1,p2,p3,s1,s2,s3,d1,d2,d3,mode)
        dyi += Y
    end
    return dx
end

function im2col3d!(w::AbstractArray{T,5}, x::AbstractArray{T,5}, x2::AbstractArray{T,2},
                 n::Int, p1::Int, p2::Int, p3::Int, s1::Int, s2::Int,
                 s3::Int, d1::Int, d2::Int, d3::Int, mode::Int) where T
    Wx,Hx,Dx,Cx,Nx = size(x)
    Ww,Hw,Dw,C1,C2 = size(w)
    xn = view(x, :, :, :, :, n)
    im2col_3d!(xn,x2,Wx,Hx,Dx,Cx,Ww,Hw,Dw,p1,p2,p3,s1,s2,s3,d1,d2,d3,mode)
    return x2
end

function col2im3d!(w::AbstractArray{T,5}, x::AbstractArray{T,5}, x2::AbstractArray{T,2},
                 n::Int, p1::Int, p2::Int, p3::Int, s1::Int, s2::Int,
                 s3::Int, d1::Int, d2::Int, d3::Int, mode::Int) where T
    Wx,Hx,Dx,Cx,Nx = size(x)
    Ww,Hw,Dw,C1,C2 = size(w)
    xn = view(x, :, :, :, :, n)
    col2im_3d!(x2,xn,Wx,Hx,Dx,Cx,Ww,Hw,Dw,p1,p2,p3,s1,s2,s3,d1,d2,d3,mode)
    return x
end
