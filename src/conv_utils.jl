function max_pooling2d_fwd!{T}(global_input::Array{T, 4}, global_output::Array{T, 4},
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

function max_pooling2d_bwd!{T}(global_input::Array{T, 4}, global_output::Array{T, 4},
  grad_output::Array{T, 4}, grad_input::Array{T, 4}, width::Int, height::Int,
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


function mean_pooling2d_fwd!{T}(global_input::Array{T, 4}, global_output::Array{T, 4},
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

function mean_pooling2d_bwd!{T}(global_input::Array{T, 4}, global_output::Array{T, 4},
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


function max_pooling3d_fwd!{T}(global_input::Array{T, 5}, global_output::Array{T, 5},
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

function max_pooling3d_bwd!{T}(global_input::Array{T, 5}, global_output::Array{T, 5},
  grad_output::Array{T, 5}, grad_input::Array{T, 5}, width::Int, height::Int, depth::Int,
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


function mean_pooling3d_fwd!{T}(global_input::Array{T, 5}, global_output::Array{T, 5},
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

function mean_pooling3d_bwd!{T}(grad_input::Array{T, 5}, grad_output::Array{T, 5},
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


function im2col_2d!{T}(img::Array{T, 3}, col::Array{T, 2}, width::Int, height::Int, channels::Int,
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


function col2im_2d!{T}(col::Array{T, 2}, img::Array{T, 3}, width::Int, height::Int,
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
    for h = 1:height_col
      for w = 1:width_col
        h_pad = (h - 1) * stride_h - pad_h + h_offset
        w_pad = (w - 1) * stride_w - pad_w + w_offset
        if h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width
          cval::T = col[((c - 1) * height_col + h - 1) * width_col + w]
          img[(c_im * height + h_pad) * width + w_pad + 1] += cval
        end
      end
    end
  end
end


function im2col_3d!{T}(img::Array{T, 4}, col::Array{T, 2}, width::Int, height::Int, depth::Int,
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
    for d = 1:depth_col
      for h = 1:height_col
        for w = 1:width_col
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
  end
end

function col2im_3d!{T}(col::Array{T, 2}, img::Array{T, 4}, width::Int, height::Int,
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

    for d = 1:depth_col
      for h = 1:height_col
        for w = 1:width_col
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
  end
end
