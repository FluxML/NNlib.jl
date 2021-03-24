export unfold, fold

"""
    unfold(X, W; stride=1, padding=0, dilation=1)
Extracts sliding local blocks from a batched input tensor. X is the input 5d vector of size 
`(spatial_dims... , channels, batch_size)`. W is the size of kernel, in format 
`(spatial_dims... , channels)`. Output has the size of `(L, channels*kernel_w*kernel_h*kernel_d, batch_size)`, 
where L is the total number of blocks.

"""
function unfold(X::AbstractArray{T,M} where T, w_dim::NTuple{K}; stride=1, padding=0, dilation=1) where M where K
    x_dim = size(X)
    if ndims(X) > 5 || ndims(X) < 3
        throw(DimensionMismatch("X and W must be 3d/4d/5d for 1d/2d/3d image. got $(ndims(X))d input"))
    end

    if ndims(X)-2 != length(w_dim)-1
        throw(DimensionMismatch("spatial dimentions of image and kernel must be equal, got $(ndims(X)-2),$(length(w_dim)-1)"))
    end

    # reassign x_dim after converting it to a 3d image type input
    x_dim = ( x_dim[1:end-2]... , fill(1,5-ndims(X))... , x_dim[end-1:end]... )
    # w_dim must be in following format: (spatial_dims..., channels_in, channels_out)
    w_dim = ( w_dim[1:end-1]... , fill(1,4-length(w_dim))... , w_dim[end], w_dim[end] ) 
    X = reshape(X, x_dim)

    # Make DenseConvDims object
    cdims = DenseConvDims(x_dim, w_dim; stride=stride, padding=padding, dilation=dilation)

    # Calculate the total number of sliding blocks
    col_dim = (im2col_dims(cdims))[1:2] # im2col_dims() returns (col_dim_x, col_dim_y, thread_num)
    col = fill(0., col_dim[1],col_dim[2],x_dim[end]) # x_dim[end] is number of batches

    # Iterate through all batchs
    @views for i = 1:x_dim[end]
        im2col!(col[:,:,i], X[:,:,:,:,i], cdims)
    end
    return col
end

"""
    fold(col, out_dim, W; stride=1, padding=0, dilation=1)
Does the opposite of `unfold()`, Combines an array of sliding local blocks into a large containing
tensor. `col` is a 3d array of shape `(L, channels*kernel_w*kernel_h*kernel_d, batch_size)`, where,
L is the total number of blocks. out_dim is the spatial dimention of the required image. W is the 
spatial dimentions of the kernel.

"""
function fold(col::AbstractArray{T,3} where T, out_dim::NTuple{M}, w_dim::NTuple{M}; stride=1, padding=0, dilation=1) where M
    # Validate input
    if length(out_dim) > 3
        throw(DimensionMismatch("output dimentions cannot be greater than 3, got $(ndims(out_dim))"))
    end

    # Create DenseConvDims object
    col_dim = size(col)
    channels = col_dim[2]÷prod(w_dim)
    x_dim = (out_dim... , fill(3-length(out_dim))... , channels,col_dim[3])
    w_dim = (w_dim... , fill(3-length(w_dim))... , channels,channels)
    cdims = DenseConvDims(x_dim,w_dim; stride=stride, padding=padding, dilation=dilation)

    img = fill(0., x_dim)

    # Iterate through all batchs
    @views for i = 1:x_dim[end]
        col2im!(img[:,:,:,:,i], col[:,:,i], cdims)
    end

    return reshape(img, (out_dim... , channels,col_dim[3]))
end