# convert padding etc. size to an Int array of the right dimension
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

function im2col_2d!{T}(img::AbstractArray{T,3}, col::AbstractArray{T,2}, width::Int, height::Int, channels::Int,
  kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int, stride_w::Int, stride_h::Int, mode::Int)

  height_col = div((height + 2pad_h - kernel_h), stride_h) + 1
  width_col = div((width + 2pad_w - kernel_w), stride_w) + 1
  channels_col = channels * kernel_h * kernel_w


  #pragma omp parallel for
  for c = 1:channels_col
    w_offset = (c - 1) % kernel_w
    h_offset = div(c - 1, kernel_w) % kernel_h
    c_im = div(c - 1, kernel_h * kernel_w)
    if mode == 0
      w_offset = kernel_w - 1 - w_offset
      h_offset = kernel_h - 1 - h_offset
    end
    for h = 1:height_col
      for w = 1:width_col
        h_pad = (h - 1) * stride_h - pad_h + h_offset
        w_pad = (w - 1) *stride_w - pad_w + w_offset
        if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
          col[((c - 1)*height_col+h-1) * width_col + w] =
           img[(c_im  * height + h_pad) * width + w_pad + 1]
        else
          col[((c - 1)*height_col+h - 1) * width_col + w] = 0
        end
      end
    end
  end
end

function col2im_2d!{T}(col::AbstractArray{T,2}, img::AbstractArray{T,3}, width::Int, height::Int,
  channels::Int, kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int, stride_w::Int,
  stride_h::Int, mode::Int)

  height_col = div(height + 2pad_h - kernel_h, stride_h) + 1
  width_col = div(width + 2pad_w - kernel_w, stride_w) + 1
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
      h_pad = (h - 1) * stride_h - pad_h + h_offset
      w_pad = (w - 1) * stride_w - pad_w + w_offset
      if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
        cval::T = col[((c - 1) * height_col + h - 1) * width_col + w]
        img[(c_im * height + h_pad) * width + w_pad + 1] += cval
      end
    end
  end
end

function im2col_3d!{T}(img::AbstractArray{T,4}, col::AbstractArray{T,2}, width::Int, height::Int, depth::Int,
  channels::Int, kernel_w::Int, kernel_h::Int, kernel_d::Int, pad_w::Int, pad_h::Int, pad_d::Int,
  stride_w::Int, stride_h::Int, stride_d::Int, mode::Int)

  height_col = div((height + 2pad_h - kernel_h), stride_h) + 1
  width_col = div((width + 2pad_w - kernel_w), stride_w) + 1
  depth_col = div((depth + 2pad_d - kernel_d), stride_d) + 1
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
      d_pad = (d - 1) * stride_d - pad_d + d_offset
      h_pad = (h - 1) * stride_h - pad_h + h_offset
      w_pad = (w - 1) *stride_w - pad_w + w_offset
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

function col2im_3d!{T}(col::AbstractArray{T,2}, img::AbstractArray{T,4}, width::Int, height::Int,
  depth::Int, channels::Int, kernel_w::Int, kernel_h::Int, kernel_d::Int,
  pad_w::Int, pad_h::Int, pad_d::Int, stride_w::Int, stride_h::Int, stride_d::Int, mode::Int)

  depth_col = div(depth + 2pad_d - kernel_d, stride_d) + 1
  height_col = div(height + 2pad_h - kernel_h, stride_h) + 1
  width_col = div(width + 2 * pad_w - kernel_w, stride_w) + 1
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
      d_pad = (d - 1) * stride_d - pad_d + d_offset
    	h_pad = (h - 1) * stride_h - pad_h + h_offset
    	w_pad = (w - 1) * stride_w - pad_w + w_offset
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

function conv2d!{T}(y::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4};
                  padding=0, stride=1, mode=0, alpha=T(1))
    if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    if Cx!=C1; throw(DimensionMismatch()); end
    Wy,Hy,Cy,Ny = size(y)
    x2dims = im2col_dims(w,y)
    x2 = similar(x, x2dims)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    M,N,K,Y = Wy*Hy,Cy,Ww*Hw*Cx,Wy*Hy*Cy
    yidx = 1
    @inbounds for n in 1:Nx
        im2col2d!(w, x, x2, n, p1, p2, s1, s2, mode)
        gemm!('N','N',M,N,K,alpha,pointer(x2),pointer(w),T(0),pointer(y,yidx))
        yidx += Y
    end
    return y
end

function conv2d_grad_w!{T}(dw::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}, dy::AbstractArray{T,4};
                   padding=0, stride=1, mode=0, alpha=1)
    # dw = x'*dy
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    Wy,Hy,Cy,Ny = size(dy)
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    # @assert Cx==C1 && Cy==C2 && Ny==Nx
    x2dims = im2col_dims(w,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Cy,Ww*Hw*Cx,Cy,Wy*Hy
    alpha,beta = T(alpha),T(1)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    dyi = 1
    @inbounds for n in 1:Nx
        im2col2d!(w, x, x2, n, p1, p2, s1, s2, mode)
        gemm!('T','N',M,N,K,alpha,pointer(x2),pointer(dy,dyi),beta,pointer(dw))
        dyi += Y
    end
    return dw
end

function conv2d_grad_x!{T}(dx::AbstractArray{T,4}, x::AbstractArray{T,4}, w::AbstractArray{T,4}, dy::AbstractArray{T,4};
                   padding=0, stride=1, mode=0, alpha=1)
    # dx = dy*w'
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    Wy,Hy,Cy,Ny = size(dy)
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    @assert Cx==C1 && Cy==C2 && Ny==Nx
    x2dims = im2col_dims(w,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Cy,Wy*Hy,Ww*Hw*Cx,Cy
    alpha,beta = T(alpha),T(0)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    dyi = 1
    @inbounds for n in 1:Nx
        gemm!('N','T',M,N,K,alpha,pointer(dy,dyi),pointer(w),beta,pointer(x2))
        col2im2d!(w,dx,x2,n,p1,p2,s1,s2,mode)
        dyi += Y
    end
    return dx
end

function im2col2d!(w::AbstractArray{T,4}, x::AbstractArray{T,4}, x2::AbstractArray{T,2},
                 n::Int, p1::Int, p2::Int, s1::Int, s2::Int, mode::Int) where T
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    xn = x[:, :, :, n]
    im2col_2d!(xn,x2,Wx,Hx,Cx,Ww,Hw,p1,p2,s1,s2,mode)
    return x2
end

function col2im2d!(w::AbstractArray{T,4}, x::AbstractArray{T,4}, x2::AbstractArray{T,2},
                 n::Int, p1::Int, p2::Int, s1::Int, s2::Int, mode::Int) where T
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    xn = x[:, :, :, n]
    col2im_2d!(x2,xn,Wx,Hx,Cx,Ww,Hw,p1,p2,s1,s2,mode)
    x[:, :, :, n] = xn
    return x
end

function conv3d!{T}(y::AbstractArray{T,5}, x::AbstractArray{T,5}, w::AbstractArray{T,5};
                  padding=0, stride=1, mode=0, alpha=T(1))
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
    M,N,K,Y = Wy*Hy*Dy,Cy,Ww*Hw*Dw*Cx,Wy*Hy*Dy*Cy
    yidx = 1
    W = reshape(w, (Ww,:,C1,C2))
    @inbounds for n in 1:Nx
        im2col3d!(w, x, x2, n, p1, p2, p3, s1, s2, s3, mode)
        gemm!('N','N',M,N,K,alpha,pointer(x2),pointer(W),T(0),pointer(y,yidx))
        yidx += Y
    end
    return y
end

function conv3d_grad_w!{T}(dw::AbstractArray{T,5}, x::AbstractArray{T,5}, w::AbstractArray{T,5}, dy::AbstractArray{T,5};
                   padding=0, stride=1, mode=0, alpha=1)
    # dw = x'*dy
    Wx,Hx,Dx,Cx,Nx = size(x)
    Ww,Hw,Dw,C1,C2 = size(w)
    Wy,Hy,Dy,Cy,Ny = size(dy)
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    # @assert Cx==C1 && Cy==C2 && Ny==Nx
    x2dims = im2col_dims(w,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Dy*Cy,Ww*Hw*Dw*Cx,Cy,Wy*Hy*Dy
    alpha,beta = T(alpha),T(1)
    (p1,p2,p3) = psize(padding,x)
    (s1,s2,s3) = psize(stride,x)
    dyi = 1
    @inbounds for n in 1:Nx
        im2col3d!(w, x, x2, n, p1, p2, p3, s1, s2, s3, mode)
        gemm!('T','N',M,N,K,alpha,pointer(x2),pointer(dy,dyi),beta,pointer(dw))
        dyi += Y
    end
    return dw
end

function conv3d_grad_x!{T}(dx::AbstractArray{T,5}, x::AbstractArray{T,5}, w::AbstractArray{T,5}, dy::AbstractArray{T,5};
                   padding=0, stride=1, mode=0, alpha=1)
    # dx = dy*w'
    Wx,Hx,Dx,Cx,Nx = size(x)
    Ww,Hw,Dw,C1,C2 = size(w)
    Wy,Hy,Dy,Cy,Ny = size(dy)
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    @assert Cx==C1 && Cy==C2 && Ny==Nx
    x2dims = im2col_dims(w,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Dy*Cy,Wy*Hy*Dy,Ww*Hw*Dw*Cx,Cy
    alpha,beta = T(alpha),T(0)
    (p1,p2,p3) = psize(padding,x)
    (s1,s2,s3) = psize(stride,x)
    dyi = 1
    @inbounds for n in 1:Nx
        gemm!('N','T',M,N,K,alpha,pointer(dy,dyi),pointer(w),beta,pointer(x2))
        col2im3d!(w,dx,x2,n,p1,p2,p3,s1,s2,s3,mode)
        dyi += Y
    end
    return dx
end

function im2col3d!(w::AbstractArray{T,5}, x::AbstractArray{T,5}, x2::AbstractArray{T,2},
                 n::Int, p1::Int, p2::Int, p3::Int, s1::Int, s2::Int,
                 s3::Int, mode::Int) where T
    Wx,Hx,Dx,Cx,Nx = size(x)
    Ww,Hw,Dw,C1,C2 = size(w)
    xn = x[:, :, :, :, n]
    im2col_3d!(xn,x2,Wx,Hx,Dx,Cx,Ww,Hw,Dw,p1,p2,p3,s1,s2,s3,mode)
    return x2
end

function col2im3d!(w::AbstractArray{T,5}, x::AbstractArray{T,5}, x2::AbstractArray{T,2},
                 n::Int, p1::Int, p2::Int, p3::Int, s1::Int, s2::Int,
                 s3::Int, mode::Int) where T
    Wx,Hx,Dx,Cx,Nx = size(x)
    Ww,Hw,Dw,C1,C2 = size(w)
    xn = x[:, :, :, :, n]
    col2im_3d!(x2,xn,Wx,Hx,Dx,Cx,Ww,Hw,Dw,p1,p2,p3,s1,s2,s3,mode)
    x[:, :, :, :, n] = xn
    return x
end
