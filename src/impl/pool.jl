function max_pooling2d_fwd!{T}(x::AbstractArray{T,4}, y::AbstractArray{T,4},
                               width::Int, height::Int, channels::Int, num::Int, pooled_width::Int,
                               pooled_height::Int, kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int,
                               stride_w::Int, stride_h::Int)
  for n = 1:num, c = 1:channels, ph = 1:pooled_height, pw = 1:pooled_width
    hstart = (ph - 1)*stride_h - pad_h
    wstart = (pw - 1)*stride_w - pad_w
    hend   = min(hstart + kernel_h, height)
    wend   = min(wstart + kernel_w, width)

    hstart = max(hstart, 0) + 1
    wstart = max(wstart, 0) + 1

    y[pw, ph, c, n] = maximum(x[wstart:wend, hstart:hend, c, n])
  end
end

function maxpool2d!(y::AbstractArray{T,4}, x::AbstractArray{T,4};
                    window::Dims{2}=(2,2), padding::Dims{2}=(0,0),
                    stride::Dims{2}=window) where T
    Wx,Hx,Cx,Nx = size(x)
    Wy,Hy,Cy,Ny = size(y)
    (w1,w2) = window
    (p1,p2) = padding
    (s1,s2) = stride
    max_pooling2d_fwd!(x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
    return y
end

function max_pooling2d_bwd!{T}(x::AbstractArray{T,4}, y::AbstractArray{T,4},
  grad_output::AbstractArray{T,4}, grad_input::AbstractArray{T,4}, width::Int, height::Int,
  channels::Int, num::Int, pooled_width::Int, pooled_height::Int, kernel_w::Int,
  kernel_h::Int, pad_w::Int, pad_h::Int, stride_w::Int, stride_h::Int)

  grad_input[:, :, :, :] = 0
  #pragma omp parallel for
  for n = 1:num, c = 1:channels, ph = 1:pooled_height, pw = 1:pooled_width
    hstart = (ph - 1) * stride_h - pad_h
    wstart = (pw - 1) * stride_w - pad_w
    hend   = min(hstart + kernel_h, height)
    wend   = min(wstart + kernel_w, width)
    hstart = max(hstart, 0) + 1
    wstart = max(wstart, 0) + 1
    maxval = y[pw, ph, c, n]
    d_maxval = grad_output[pw, ph, c, n]
    for h = hstart:hend, w = wstart:wend
      if x[w, h, c, n] == maxval
        grad_input[w, h, c, n] += d_maxval
      end
    end
  end
end

function maxpool2d_grad!(dx::AbstractArray{T,4}, dy::AbstractArray{T,4}, y::AbstractArray{T,4}, x::AbstractArray{T,4};
                         window::Dims{2}=(2,2), padding::Dims{2}=(0,0),
                         stride::Dims{2}=window) where T
    Wx,Hx,Cx,Nx = size(x)
    Wy,Hy,Cy,Ny = size(y)
    (w1,w2) = window
    (p1,p2) = padding
    (s1,s2) = stride
    max_pooling2d_bwd!(x,y,dy,dx,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
    return dx
end


function mean_pooling2d_fwd!{T}(x::AbstractArray{T,4}, y::AbstractArray{T,4},
  width::Int, height::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, kernel_w::Int, kernel_h::Int,pad_w::Int, pad_h::Int,
  stride_w::Int, stride_h::Int)
  kernel_size = kernel_w * kernel_h
  for n = 1:num, c = 1:channels, ph = 1:pooled_height, pw = 1:pooled_width
    hstart = (ph - 1) * stride_h - pad_h
    wstart = (pw - 1) * stride_w - pad_w
    hend   = min(hstart + kernel_h, height)
    wend   = min(wstart + kernel_w, width)

    hstart = max(hstart, 0) + 1
    wstart = max(wstart, 0) + 1

    y[pw, ph, c, n] = sum(x[wstart:wend, hstart:hend, c, n]) / kernel_size
  end
end

function meanpool2d!(y::AbstractArray{T,4}, x::AbstractArray{T,4};
                     window::Dims{2}=(2,2), padding::Dims{2}=(0,0),
                     stride::Dims{2}=window) where T
    Wx,Hx,Cx,Nx = size(x)
    Wy,Hy,Cy,Ny = size(y)
    (w1,w2) = window
    (p1,p2) = padding
    (s1,s2) = stride
    mean_pooling2d_fwd!(x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
    return y
end

function mean_pooling2d_bwd!{T}(x::AbstractArray{T,4}, y::AbstractArray{T,4},
  width::Int, height::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int,
  stride_w::Int, stride_h::Int)

  x[:, :, :, :] = 0
  kernel_size = kernel_w * kernel_h

  #pragma omp parallel for
  for n = 1:num, c = 1:channels, ph = 1:pooled_height, pw = 1:pooled_width
    hstart = (ph - 1) * stride_h - pad_h
    wstart = (pw - 1) * stride_w - pad_w
    hend   = min(hstart + kernel_h, height)
    wend   = min(wstart + kernel_w, width)
    hstart = max(hstart, 0) + 1
    wstart = max(wstart, 0) + 1

    oval = y[pw, ph, c, n] / kernel_size
    x[wstart:wend, hstart:hend, c, n] += oval
  end
end

function meanpool2d_grad!(dx::AbstractArray{T,4}, dy::AbstractArray{T,4}, y::AbstractArray{T,4}, x::AbstractArray{T,4};
                         window::Dims{2}=(2,2), padding::Dims{2}=(0,0),
                         stride::Dims{2}=window) where T
    Wx,Hx,Cx,Nx = size(x)
    Wy,Hy,Cy,Ny = size(y)
    (w1,w2) = window
    (p1,p2) = padding
    (s1,s2) = stride
    mean_pooling2d_bwd!(dx,dy,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
    return dx
end

function max_pooling3d_fwd!{T}(x::AbstractArray{T,5}, y::AbstractArray{T,5},
  width::Int, height::Int, depth::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, pooled_depth::Int, kernel_w::Int, kernel_h::Int, kernel_d::Int,
  pad_w::Int, pad_h::Int, pad_d::Int, stride_w::Int, stride_h::Int, stride_d::Int)
  for n = 1:num, c = 1:channels, pd = 1:pooled_depth, ph = 1:pooled_height, pw = 1:pooled_width
    dstart = (pd - 1)* stride_d - pad_d
    hstart = (ph - 1)* stride_h - pad_h
    wstart = (pw - 1)* stride_w - pad_w

    dend   = min(dstart + kernel_d, depth)
    hend   = min(hstart + kernel_h, height)
    wend   = min(wstart + kernel_w, width)

    dstart = max(dstart, 0) + 1
    hstart = max(hstart, 0) + 1
    wstart = max(wstart, 0) + 1

    y[pw, ph, pd, c, n] =
    maximum(x[wstart:wend, hstart:hend, dstart:dend, c, n])
  end
end

function maxpool3d!(y::AbstractArray{T,5}, x::AbstractArray{T,5};
                    window::Dims{3}=(2,2,2), padding::Dims{3}=(0,0,0),
                    stride::Dims{3}=window) where T
    Wx,Hx,Dx,Cx,Nx = size(x)
    Wy,Hy,Dy,Cy,Ny = size(y)
    (w1,w2,w3) = psize(window, x)
    (p1,p2,p3) = psize(padding, x)
    (s1,s2,s3) = psize(stride, x)
    max_pooling3d_fwd!(x,y,Wx,Hx,Dx,Cx,Nx,Wy,Hy,Dy,w1,w2,w3,p1,p2,p3,s1,s2,s3)
    return y
end

function max_pooling3d_bwd!{T}(x::AbstractArray{T,5}, y::AbstractArray{T,5},
  grad_output::AbstractArray{T,5}, grad_input::AbstractArray{T,5}, width::Int, height::Int, depth::Int,
  channels::Int, num::Int, pooled_width::Int, pooled_height::Int, pooled_depth::Int,
  kernel_w::Int, kernel_h::Int, kernel_d::Int, pad_w::Int, pad_h::Int, pad_d::Int,
  stride_w::Int, stride_h::Int, stride_d::Int)

  grad_input[:, :, :, :, :] = 0

  #pragma omp parallel for
  for n = 1:num, c = 1:channels, pd = 1:pooled_depth, ph = 1:pooled_height, pw = 1:pooled_width
    dstart = (pd - 1) * stride_h - pad_h
    hstart = (ph - 1) * stride_h - pad_h
    wstart = (pw - 1) * stride_w - pad_w

    dend   = min(dstart + kernel_d, depth)
    hend   = min(hstart + kernel_h, height)
    wend   = min(wstart + kernel_w, width)

    dstart = max(dstart, 0) + 1
    hstart = max(hstart, 0) + 1
    wstart = max(wstart, 0) + 1

    maxval = y[pw, ph, pd, c, n]
    d_maxval = grad_output[pw, ph, pd, c, n]
    for d = dstart:dend, h = hstart:hend, w = wstart:wend
      if x[w, h, d, c, n] == maxval
        grad_input[w, h, d, c, n] += d_maxval
      end
    end
  end
end

function maxpool3d_grad!(dx::AbstractArray{T,5}, dy::AbstractArray{T,5}, y::AbstractArray{T,5}, x::AbstractArray{T,5};
                         window::Dims{3}=(2,2,2), padding::Dims{3}=(0,0,0),
                         stride::Dims{3}=window) where T
    Wx,Hx,Dx,Cx,Nx = size(x)
    Wy,Hy,Dy,Cy,Ny = size(y)
    (w1,w2,w3) = psize(window, x)
    (p1,p2,p3) = psize(padding, x)
    (s1,s2,s3) = psize(stride, x)
    max_pooling3d_bwd!(x,y,dy,dx,Wx,Hx,Dx,Cx,Nx,Wy,Hy,Dy,w1,w2,w3,p1,p2,p3,s1,s2,s3)
    return dx
end

function mean_pooling3d_fwd!{T}(x::AbstractArray{T,5}, y::AbstractArray{T,5},
  width::Int, height::Int, depth::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, pooled_depth::Int, kernel_w::Int, kernel_h::Int, kernel_d::Int,
  pad_w::Int, pad_h::Int, pad_d::Int, stride_w::Int, stride_h::Int, stride_d::Int)

  kernel_size = kernel_w * kernel_h * kernel_d
  #pragma omp parallel for
  for n = 1:num, c = 1:channels, pd = 1:pooled_depth, ph = 1:pooled_height, pw = 1:pooled_width
    dstart = (pd - 1) * stride_d - pad_d
    hstart = (ph - 1) * stride_h - pad_h
    wstart = (pw - 1) * stride_w - pad_w

    dend   = min(dstart + kernel_d, depth)
    hend   = min(hstart + kernel_h, height)
    wend   = min(wstart + kernel_w, width)

    dstart = max(dstart, 0) + 1
    hstart = max(hstart, 0) + 1
    wstart = max(wstart, 0) + 1

    y[pw, ph, pd, c, n] =
    sum(x[wstart:wend, hstart:hend, dstart:dend, c, n]) / kernel_size
  end
end

function meanpool3d!(y::AbstractArray{T,5}, x::AbstractArray{T,5};
                     window::Dims{3}=(2,2,2), padding::Dims{3}=(0,0,0),
                     stride::Dims{3}=window) where T
    Wx,Hx,Dx,Cx,Nx = size(x)
    Wy,Hy,Dy,Cy,Ny = size(y)
    (w1,w2,w3) = psize(window, x)
    (p1,p2,p3) = psize(padding, x)
    (s1,s2,s3) = psize(stride, x)
    mean_pooling3d_fwd!(x,y,Wx,Hx,Dx,Cx,Nx,Wy,Hy,Dy,w1,w2,w3,p1,p2,p3,s1,s2,s3)
    return y
end

function mean_pooling3d_bwd!{T}(grad_input::AbstractArray{T,5}, grad_output::AbstractArray{T,5},
  width::Int, height::Int, depth::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, pooled_depth::Int, kernel_w::Int, kernel_h::Int, kernel_d::Int,
  pad_w::Int, pad_h::Int, pad_d::Int, stride_w::Int, stride_h::Int, stride_d::Int)

  kernel_size = kernel_w * kernel_h * kernel_d
  fill!(grad_input, 0.0)

  #pragma omp parallel for
  for n = 1:num, c = 1:channels, pd = 1:pooled_depth, ph = 1:pooled_height, pw = 1:pooled_width
    dstart = (pd - 1) * stride_d - pad_d
    hstart = (ph - 1) * stride_h - pad_h
    wstart = (pw - 1) * stride_w - pad_w
    dend   = min(dstart + kernel_d, depth)
    hend   = min(hstart + kernel_h, height)
    wend   = min(wstart + kernel_w, width)
    dstart = max(dstart, 0) + 1
    hstart = max(hstart, 0) + 1
    wstart = max(wstart, 0) + 1

    grad_input[wstart:wend, hstart:hend, dstart:dend, c, n] += grad_output[pw, ph, pd, c, n] / kernel_size
  end
end

function meanpool3d_grad!(dx::AbstractArray{T,5}, dy::AbstractArray{T,5}, y::AbstractArray{T,5}, x::AbstractArray{T,5};
                         window::Dims{3}=(2,2,2), padding::Dims{3}=(0,0,0),
                         stride::Dims{3}=window) where T
    Wx,Hx,Dx,Cx,Nx = size(x)
    Wy,Hy,Dy,Cy,Ny = size(y)
    (w1,w2,w3) = psize(window, x)
    (p1,p2,p3) = psize(padding, x)
    (s1,s2,s3) = psize(stride, x)
    mean_pooling3d_bwd!(dx,dy,Wx,Hx,Dx,Cx,Nx,Wy,Hy,Dy,w1,w2,w3,p1,p2,p3,s1,s2,s3)
    return dx
end

function max_pooling_nd_fwd!{T}(x::AbstractArray{T}, y::AbstractArray{T},
                               kernel_dim::AbstractArray{Int}, padding::AbstractArray{Int},
                               stride::AbstractArray{Int})

  inp_dim = collect(size(x))
  out_dim = collect(size(y))
  n_dim = ndims(y)
  for i = 1:prod(out_dim)
    dim_iter = zeros(n_dim)
    dim_iter[1] = (i - 1) % out_dim[1]
    for j = 2:n_dim
      dim_iter[j] = div(i - 1, prod(out_dim[1:j-1])) % out_dim[j]
    end

    dim_iter = Int.(dim_iter)

    dim_start = dim_iter[1:n_dim-2] .* stride - padding
    dim_end = min.(dim_start + kernel_dim, inp_dim[1:n_dim-2])

    dim_start = max.(dim_start, 0) + 1

    out_ind = CartesianIndex(Int.(tuple(dim_iter + 1...)))

    rngs = []
    for i=1:n_dim-2
      push!(rngs, dim_start[i]:dim_end[i])
    end
    push!(rngs, dim_iter[n_dim-1] + 1:dim_iter[n_dim-1] + 1)
    push!(rngs, dim_iter[n_dim] + 1:dim_iter[n_dim] + 1)

    cart_rng = CartesianRange(tuple(rngs...))

    max_ = x[CartesianIndex(ntuple(i->i<=n_dim-2?dim_start[i]:dim_iter[i]+1, n_dim))]
    for i in cart_rng
      max_ = max(max_, x[i])
    end
    y[out_ind] = max_
  end
end

function maxpool_nd!{T, N}(y::AbstractArray{T}, x::AbstractArray{T},
                    window::Dims{N}; padding::Dims{N}=tuple(zeros(Int, N)), stride::Dims{N}=window)

    window  = collect(window)
    padding = collect(padding)
    stride  = collect(stride)
    max_pooling_nd_fwd!(x, y, window, padding, stride)
    return y
end

function max_pooling_nd_bwd!{T}(x::AbstractArray{T}, y::AbstractArray{T},
  grad_output::AbstractArray{T}, grad_input::AbstractArray{T},
  kernel_dim::AbstractArray{Int}, padding::AbstractArray{Int}, stride::AbstractArray{Int})

  fill!(grad_input, 0)

  inp_dim = collect(size(x))
  out_dim = collect(size(y))
  n_dim = ndims(y)

  for i = 1:prod(out_dim)
    dim_iter = zeros(n_dim)
    dim_iter[1] = (i - 1) % out_dim[1]
    for j = 2:n_dim
      dim_iter[j] = div(i - 1, prod(out_dim[1:j-1])) % out_dim[j]
    end

    dim_iter = Int.(dim_iter)

    dim_start = dim_iter[1:n_dim-2] .* stride - padding
    dim_end = min.(dim_start + kernel_dim, inp_dim[1:n_dim-2])

    dim_start = max.(dim_start, 0) + 1

    out_ind = CartesianIndex(Int.(tuple(dim_iter + 1...)))

    rngs = []
    for i=1:n_dim-2
      push!(rngs, dim_start[i]:dim_end[i])
    end
    push!(rngs, dim_iter[n_dim-1] + 1:dim_iter[n_dim-1] + 1)
    push!(rngs, dim_iter[n_dim] + 1:dim_iter[n_dim] + 1)

    cart_rng = CartesianRange(tuple(rngs...))

    maxval = y[i]
    d_maxval = grad_output[i]

    for i in cart_rng
      if x[i] == maxval
        grad_input[i] += d_maxval
      end
    end
  end
end

function maxpool_nd_grad!{T, N}(dx::AbstractArray{T}, dy::AbstractArray{T},
                          y::AbstractArray{T}, x::AbstractArray{T}, window::Dims{N};
                          padding::Dims{N}=tuple(zeros(Int, N)), stride::Dims{N}=window)
    window = collect(window)
    padding = collect(padding)
    stride = collect(stride)
    max_pooling_nd_bwd!(x,y,dy,dx,window,padding,stride)
    return dx
end

function mean_pooling_nd_fwd!{T}(x::AbstractArray{T}, y::AbstractArray{T},
                               kernel_dim::AbstractArray{Int}, padding::AbstractArray{Int},
                               stride::AbstractArray{Int})

  inp_dim = collect(size(x))
  out_dim = collect(size(y))
  n_dim = ndims(y)
  kernel_size = prod(kernel_dim[1:n_dim-2])
  for i = 1:prod(out_dim)
    dim_iter = zeros(n_dim)
    dim_iter[1] = (i - 1) % out_dim[1]
    for j = 2:n_dim
      dim_iter[j] = div(i - 1, prod(out_dim[1:j-1])) % out_dim[j]
    end

    dim_iter = Int.(dim_iter)

    dim_start = dim_iter[1:n_dim-2] .* stride - padding
    dim_end = min.(dim_start + kernel_dim, inp_dim[1:n_dim-2])

    dim_start = max.(dim_start, 0) + 1

    out_ind = CartesianIndex(Int.(tuple(dim_iter + 1...)))

    rngs = []
    for i=1:n_dim-2
      push!(rngs, dim_start[i]:dim_end[i])
    end
    push!(rngs, dim_iter[n_dim-1] + 1:dim_iter[n_dim-1] + 1)
    push!(rngs, dim_iter[n_dim] + 1:dim_iter[n_dim] + 1)

    cart_rng = CartesianRange(tuple(rngs...))

    sum_ = 0
    for i in cart_rng
      sum_ += x[i] / kernel_size
    end
    y[out_ind] = sum_
  end
end

function meanpool_nd!{T, N}(y::AbstractArray{T}, x::AbstractArray{T},
                    window::Dims{N}; padding::Dims{N}=tuple(zeros(Int, N)), stride::Dims{N}=window)

    window  = collect(window)
    padding = collect(padding)
    stride  = collect(stride)
    mean_pooling_nd_fwd!(x, y, window, padding, stride)
    return y
end

function mean_pooling_nd_bwd!{T}(x::AbstractArray{T}, y::AbstractArray{T},
  kernel_dim::AbstractArray{Int}, padding::AbstractArray{Int}, stride::AbstractArray{Int})

  fill!(x, 0)

  inp_dim = collect(size(x))
  out_dim = collect(size(y))
  n_dim = ndims(y)
  kernel_size = prod(kernel_dim[1:n_dim-2])

  for i = 1:prod(out_dim)
    dim_iter = zeros(n_dim)
    dim_iter[1] = (i - 1) % out_dim[1]
    for j = 2:n_dim
      dim_iter[j] = div(i - 1, prod(out_dim[1:j-1])) % out_dim[j]
    end

    dim_iter = Int.(dim_iter)

    dim_start = dim_iter[1:n_dim-2] .* stride - padding
    dim_end = min.(dim_start + kernel_dim, inp_dim[1:n_dim-2])

    dim_start = max.(dim_start, 0) + 1

    out_ind = CartesianIndex(Int.(tuple(dim_iter + 1...)))

    rngs = []
    for i=1:n_dim-2
      push!(rngs, dim_start[i]:dim_end[i])
    end
    push!(rngs, dim_iter[n_dim-1] + 1:dim_iter[n_dim-1] + 1)
    push!(rngs, dim_iter[n_dim] + 1:dim_iter[n_dim] + 1)

    cart_rng = CartesianRange(tuple(rngs...))

    for i in cart_rng
      x[i] += y[out_ind] / kernel_size
    end
  end
end

function meanpool_nd_grad!{T, N}(dx::AbstractArray{T}, dy::AbstractArray{T},
                          y::AbstractArray{T}, x::AbstractArray{T}, window::Dims{N};
                          padding::Dims{N}=tuple(zeros(Int, N)), stride::Dims{N}=window)
    window = collect(window)
    padding = collect(padding)
    stride = collect(stride)
    mean_pooling_nd_bwd!(dy,dx,window,padding,stride)
    return dx
end
