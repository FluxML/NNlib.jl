export pixel_shuffle

"""
    pixel_shuffle(x, r)
    
Pixel shuffling operation. `r` is the scale factor for shuffling.
The operation converts an input of size [W,H,r²C,N] to size [rW,rH,C,N]
Used extensively in super-resolution networks to upsample 
towards high resolution features.

Reference : https://arxiv.org/pdf/1609.05158.pdf
"""
function pixel_shuffle(x::AbstractArray{T, 4}, r::Integer) where T <:Number
    w, h, c, n = size(x)
    @assert c % r^2 == 0
    c_out = c ÷ r^2
    w_out = w * r
    h_out = h * r
    x = reshape(x, (w, h, r, r, c_out, n))
    x = permutedims(x, (3, 1, 4, 2, 5, 6))
    return reshape(x, (w_out, h_out, c_out, n))
end