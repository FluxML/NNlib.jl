export unfold, fold

"""
    unfold(X, W; stride=1, padding=0, dilation=1)
Extracts sliding local blocks from a batched input tensor. X is the input 5d vector of size 
`(image_w, image_h, image_d, channels, batch_size)`. Only 3D images are supported. Can be 
used for 2D or 1D images, by adding an extra singleton dimention. W is the size of kernel, 
in format `(kernel_w, kernel_h, kernel_d, channels_in, channels_out)`. Note that 
`channels_in = channels_out`. Output has the size of `(L, channels*kernel_w*kernel_h*kernel_d, batch_size)`, 
where L is the total number of blocks.

"""
function unfold(X::AbstractArray{T,5} where T, w_dim::NTuple{5}; stride=1, padding=0, dilation=1)
    x_dim = size(X)
    if w_dim[end] != w_dim[end-1]
        throw(DimensionMismatch("input channels must be equal to output channels in the kernel"))
    end

    # Make DenseConvDims object
    cdims = DenseConvDims(x_dim, w_dim; stride=stride, padding=padding, dilation=dilation)

    # Calculate the total number of sliding blocks
    col_dim = (im2col_dims(cdims))[1:2]
    col = undef

    # Iterate through all batchs
    for i = 1:x_dim[end]
        temp = fill(0., col_dim)
        im2col!(temp, X[:,:,:,:,i], cdims)
        if i == 1
            col = reshape(temp, col_dim[1], col_dim[2], 1)
        else
            col = cat(dims=3, col, temp)
        end
    end
    return col;
end

"""
    fold(col, out_dim, W; stride=1, padding=0, dilation=1)
Does the opposite of `unfold()`, Combines an array of sliding local blocks into a large containing
tensor. `col` is a 3d array of shape `(L, channels*kernel_w*kernel_h*kernel_d, batch_size)`, where,
L is the total number of blocks. out_dim is the spatial dimention of the required image. Note that 
this is a 3d tuple. W is the spatial dimentions of the kernel(3d).

"""
function fold(col::AbstractArray{T,3} where T, out_dim::NTuple{3}, w_dim::NTuple{3}; stride=1, padding=0, dilation=1)
    # Create DenseConvDims object
    col_dim = size(col)
    channels = col_dim[2]÷prod(w_dim)
    x_dim = (out_dim[1],out_dim[2],out_dim[3],channels,col_dim[3])
    w_dim = (w_dim[1],w_dim[2],w_dim[3],channels,channels)
    cdims = DenseConvDims(x_dim,w_dim; stride=stride, padding=padding, dilation=dilation)

    img = undef

    # Iterate through all batchs
    for i = 1:x_dim[end]
        temp = fill(0., x_dim[1:4])
        col2im!(temp, col[:,:,i], cdims)
        if i == 1
            img = reshape(temp, x_dim[1], x_dim[2], x_dim[3], x_dim[4], 1)
        else
            img = cat(dims=5, img, temp)
        end
    end
    return img;
end