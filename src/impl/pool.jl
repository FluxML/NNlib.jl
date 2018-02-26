function max_pooling2d_fwd!{T}(global_input::AbstractArray{T,4}, global_output::AbstractArray{T,4},
  width::Int, height::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int,
  stride_w::Int, stride_h::Int)
  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = (ph - 1)* stride_h - pad_h
          wstart = (pw - 1)* stride_w - pad_w
          hend   = min(hstart + kernel_h, height)
          wend   = min(wstart + kernel_w, width)

          hstart = max(hstart, 0) + 1
          wstart = max(wstart, 0) + 1

          global_output[pw, ph, c, n] = maximum(global_input[wstart:wend, hstart:hend, c, n])
        end
      end
    end
  end
end

function max_pooling2d_bwd!{T}(global_input::AbstractArray{T,4}, global_output::AbstractArray{T,4},
  grad_output::AbstractArray{T,4}, grad_input::AbstractArray{T,4}, width::Int, height::Int,
  channels::Int, num::Int, pooled_width::Int, pooled_height::Int, kernel_w::Int,
  kernel_h::Int, pad_w::Int, pad_h::Int, stride_w::Int, stride_h::Int)

  grad_input[:, :, :, :] = 0
  #pragma omp parallel for
  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = (ph - 1) * stride_h - pad_h
          wstart = (pw - 1) * stride_w - pad_w
          hend   = min(hstart + kernel_h, height)
          wend   = min(wstart + kernel_w, width)
          hstart = max(hstart, 0) + 1
          wstart = max(wstart, 0) + 1
          maxval = global_output[pw, ph, c, n]
          d_maxval = grad_output[pw, ph, c, n]
          for h = hstart:hend
            for w = wstart:wend
              if global_input[w, h, c, n] == maxval
                grad_input[w, h, c, n] += d_maxval
              end
            end
          end
        end
      end
    end
  end
end


function mean_pooling2d_fwd!{T}(global_input::AbstractArray{T,4}, global_output::AbstractArray{T,4},
  width::Int, height::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, kernel_w::Int, kernel_h::Int,pad_w::Int, pad_h::Int,
  stride_w::Int, stride_h::Int)
  kernel_size = kernel_w * kernel_h
  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = (ph - 1) * stride_h - pad_h
          wstart = (pw - 1) * stride_w - pad_w
          hend   = min(hstart + kernel_h, height)
          wend   = min(wstart + kernel_w, width)

          hstart = max(hstart, 0) + 1
          wstart = max(wstart, 0) + 1

          global_output[pw, ph, c, n] = sum(global_input[wstart:wend, hstart:hend, c, n]) / kernel_size
        end
      end
    end
  end
end

function mean_pooling2d_bwd!{T}(global_input::AbstractArray{T,4}, global_output::AbstractArray{T,4},
  width::Int, height::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, kernel_w::Int, kernel_h::Int, pad_w::Int, pad_h::Int,
  stride_w::Int, stride_h::Int)

  global_input[:, :, :, :] = 0
  kernel_size = kernel_w * kernel_h

  #pragma omp parallel for
  for n = 1:num
    for c = 1:channels
      for ph = 1:pooled_height
        for pw = 1:pooled_width
          hstart = (ph - 1) * stride_h - pad_h
          wstart = (pw - 1) * stride_w - pad_w
          hend   = min(hstart + kernel_h, height)
          wend   = min(wstart + kernel_w, width)
          hstart = max(hstart, 0) + 1
          wstart = max(wstart, 0) + 1

          oval = global_output[pw, ph, c, n] / kernel_size
          global_input[wstart:wend, hstart:hend, c, n] += oval
        end
      end
    end
  end
end

function pool2d(x::AbstractArray{T,4}; window=2, padding=0, stride=window, mode=0) where T
    Wx,Hx,Cx,Nx = size(x)
    Wy,Hy,Cy,Ny = pdims(x;window=window,padding=padding,stride=stride)
    y = similar(x, (Wy,Hy,Cy,Ny))
    (w1,w2) = psize(window, x)
    (p1,p2) = psize(padding, x)
    (s1,s2) = psize(stride, x)
    if mode == 0
        max_pooling2d_fwd!(x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
    elseif mode == 1 || (mode == 2 && p1==p2==0)
        mean_pooling2d_fwd!(x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
    else
        throw(ArgumentError("mode $mode not supported by cpu pool"))
    end
    return y
end

function pool2d_grad(x::AbstractArray{T,4}, y::AbstractArray{T,4}, dy::AbstractArray{T,4};
                     window=2, padding=0, stride=window, mode=0) where T
    Wx,Hx,Cx,Nx = size(x)
    Wy,Hy,Cy,Ny = size(y)
    dx = similar(x)
    (w1,w2) = psize(window, x)
    (p1,p2) = psize(padding, x)
    (s1,s2) = psize(stride, x)
    if mode == 0
        max_pooling2d_bwd!(x,y,dy,dx,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
    elseif mode == 1 || (mode == 2 && p1==p2==0)
        mean_pooling2d_bwd!(dx,dy,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
    else
        throw(ArgumentError("mode $mode not supported by cpu pool"))
    end
    return dx
end

function max_pooling3d_fwd!{T}(global_input::AbstractArray{T,5}, global_output::AbstractArray{T,5},
  width::Int, height::Int, depth::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, pooled_depth::Int, kernel_w::Int, kernel_h::Int, kernel_d::Int,
  pad_w::Int, pad_h::Int, pad_d::Int, stride_w::Int, stride_h::Int, stride_d::Int)
  for n = 1:num
    for c = 1:channels
      for pd = 1:pooled_depth
        for ph = 1:pooled_height
          for pw = 1:pooled_width
            dstart = (pd - 1)* stride_d - pad_d
            hstart = (ph - 1)* stride_h - pad_h
            wstart = (pw - 1)* stride_w - pad_w

            dend   = min(dstart + kernel_d, depth)
            hend   = min(hstart + kernel_h, height)
            wend   = min(wstart + kernel_w, width)

            dstart = max(dstart, 0) + 1
            hstart = max(hstart, 0) + 1
            wstart = max(wstart, 0) + 1

            global_output[pw, ph, pd, c, n] =
            maximum(global_input[wstart:wend, hstart:hend, dstart:dend, c, n])
          end
        end
      end
    end
  end
end

function max_pooling3d_bwd!{T}(global_input::AbstractArray{T,5}, global_output::AbstractArray{T,5},
  grad_output::AbstractArray{T,5}, grad_input::AbstractArray{T,5}, width::Int, height::Int, depth::Int,
  channels::Int, num::Int, pooled_width::Int, pooled_height::Int, pooled_depth::Int,
  kernel_w::Int, kernel_h::Int, kernel_d::Int, pad_w::Int, pad_h::Int, pad_d::Int,
  stride_w::Int, stride_h::Int, stride_d::Int)

  grad_input[:, :, :, :, :] = 0

  #pragma omp parallel for
  for n = 1:num
    for c = 1:channels
      for pd = 1:pooled_depth
        for ph = 1:pooled_height
          for pw = 1:pooled_width
            dstart = (pd - 1) * stride_h - pad_h
            hstart = (ph - 1) * stride_h - pad_h
            wstart = (pw - 1) * stride_w - pad_w

            dend   = min(dstart + kernel_d, depth)
            hend   = min(hstart + kernel_h, height)
            wend   = min(wstart + kernel_w, width)

            dstart = max(dstart, 0) + 1
            hstart = max(hstart, 0) + 1
            wstart = max(wstart, 0) + 1

            maxval = global_output[pw, ph, pd, c, n]
            d_maxval = grad_output[pw, ph, pd, c, n]
            for d = dstart:dend
              for h = hstart:hend
                for w = wstart:wend
                  if global_input[w, h, d, c, n] == maxval
                    grad_input[w, h, d, c, n] += d_maxval
                  end
                end
              end
            end
          end
        end
      end
    end
  end
end


function mean_pooling3d_fwd!{T}(global_input::AbstractArray{T,5}, global_output::AbstractArray{T,5},
  width::Int, height::Int, depth::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, pooled_depth::Int, kernel_w::Int, kernel_h::Int, kernel_d::Int,
  pad_w::Int, pad_h::Int, pad_d::Int, stride_w::Int, stride_h::Int, stride_d::Int)

  kernel_size = kernel_w * kernel_h * kernel_d
  #pragma omp parallel for
  for n = 1:num
    for c = 1:channels
      for pd = 1:pooled_depth
        for ph = 1:pooled_height
          for pw = 1:pooled_width
            dstart = (pd - 1) * stride_d - pad_d
            hstart = (ph - 1) * stride_h - pad_h
            wstart = (pw - 1) * stride_w - pad_w

            dend   = min(dstart + kernel_d, depth)
            hend   = min(hstart + kernel_h, height)
            wend   = min(wstart + kernel_w, width)

            dstart = max(dstart, 0) + 1
            hstart = max(hstart, 0) + 1
            wstart = max(wstart, 0) + 1

            global_output[pw, ph, pd, c, n] =
            sum(global_input[wstart:wend, hstart:hend, dstart:dend, c, n]) / kernel_size
          end
        end
      end
    end
  end
end

function mean_pooling3d_bwd!{T}(grad_input::AbstractArray{T,5}, grad_output::AbstractArray{T,5},
  width::Int, height::Int, depth::Int, channels::Int, num::Int, pooled_width::Int,
  pooled_height::Int, pooled_depth::Int, kernel_w::Int, kernel_h::Int, kernel_d::Int,
  pad_w::Int, pad_h::Int, pad_d::Int, stride_w::Int, stride_h::Int, stride_d::Int)

  kernel_size = kernel_w * kernel_h * kernel_d
  fill!(grad_input, 0.0)

  #pragma omp parallel for
  for n = 1:num
    for c = 1:channels
      for pd = 1:pooled_depth
        for ph = 1:pooled_height
          for pw = 1:pooled_width
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
      end
    end
  end
end

function pool3d(x::AbstractArray{T,5}; window=2, padding=0, stride=window, mode=0) where T
    Wx,Hx,Dx,Cx,Nx = size(x)
    Wy,Hy,Dy,Cy,Ny = pdims(x;window=window,padding=padding,stride=stride)
    y = similar(x, (Wy,Hy,Dy,Cy,Ny))
    (w1,w2,w3) = psize(window, x)
    (p1,p2,p3) = psize(padding, x)
    (s1,s2,s3) = psize(stride, x)
    if mode == 0
        max_pooling3d_fwd!(x,y,Wx,Hx,Dx,Cx,Nx,Wy,Hy,Dy,w1,w2,w3,p1,p2,p3,s1,s2,s3)
    elseif mode == 1 || (mode == 2 && p1==p2==0)
        mean_pooling3d_fwd!(x,y,Wx,Hx,Dx,Cx,Nx,Wy,Hy,Dy,w1,w2,w3,p1,p2,p3,s1,s2,s3)
    else
        throw(ArgumentError("mode $mode not supported by cpu pool"))
    end
    return y
end

function pool3d_grad(x::AbstractArray{T,5}, y::AbstractArray{T,5}, dy::AbstractArray{T,5};
                     window=2, padding=0, stride=window, mode=0) where T
    Wx,Hx,Dx,Cx,Nx = size(x)
    Wy,Hy,Dy,Cy,Ny = size(y)
    dx = similar(x)
    (w1,w2,w3) = psize(window, x)
    (p1,p2,p3) = psize(padding, x)
    (s1,s2,s3) = psize(stride, x)
    if mode == 0
        max_pooling3d_bwd!(x,y,dy,dx,Wx,Hx,Dx,Cx,Nx,Wy,Hy,Dy,w1,w2,w3,p1,p2,p3,s1,s2,s3)
    elseif mode == 1 || (mode == 2 && p1==p2==0)
        mean_pooling3d_bwd!(dx,dy,Wx,Hx,Dx,Cx,Nx,Wy,Hy,Dy,w1,w2,w3,p1,p2,p3,s1,s2,s3)
    else
        throw(ArgumentError("mode $mode not supported by cpu pool"))
    end
    return dx
end
