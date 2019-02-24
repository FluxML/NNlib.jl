export upsample, ∇upsample

function upsample!(y, x, height, width, channels, batch, stride, scale = 1)
    w_mul = height
    ch_mul = w_mul * width
    b_mul = ch_mul * channels
    
    @inbounds begin
        for b in 1:batch,
            ch in 1:channels,
            w in 1:(width * stride[2]),
            h in 1:(height * stride[1])

            x_idx = (b - 1) * b_mul + (ch - 1) * ch_mul + ((w - 1) ÷ stride[2]) * w_mul + (h - 1) ÷ stride[1] + 1
            y_idx = (b - 1) * b_mul * stride[2] * stride[1] + (ch - 1) * ch_mul * stride[2] * stride[1] + (w - 1) * w_mul * stride[1] + h
            y[y_idx] = scale * x[x_idx]      
        end
    end
    y
end

function upsample(x, stride, scale = 1)
    (height, width, channels, batch) = size(x)
    y = similar(x, (height * stride[1], width * stride[2], channels, batch))
    upsample!(y, x, height, width, channels, batch, stride, scale)
end

function ∇upsample!(dx, dy, height, width, channels, batch, stride, scale = 1)
    w_mul = height
    ch_mul = w_mul * width
    b_mul = ch_mul * channels
    
    @inbounds begin
        for b in 1:batch,
            ch in 1:channels,
            w in 1:(width * stride[2]),
            h in 1:(height * stride[1])

            dx_idx = (b - 1) * b_mul + (ch - 1) * ch_mul + ((w - 1) ÷ stride[2]) * w_mul + (h - 1) ÷ stride[1] + 1
            dy_idx = (b - 1) * b_mul * stride[2] * stride[1] + (ch - 1) * ch_mul * stride[2] * stride[1] + (w - 1) * w_mul * stride[1] + h
            dx[dx_idx] += dy[dy_idx] / scale      
        end
    end
    dx
end

function ∇upsample(dy, stride, scale = 1)
    (height, width, channels, batch) = size(dy)
    @assert height % stride[1] == 0
    @assert width % stride[2] == 0
    dx = similar(dy, (height ÷ stride[1], width ÷ stride[2], channels, batch))
    ∇upsample!(dx, dy, size(dx)..., stride, scale)
end
