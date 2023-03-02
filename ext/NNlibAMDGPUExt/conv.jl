function NNlib.conv!(
    y::ROCArray{T, N}, x::ROCArray{T, N}, w::ROCArray{T, N}, cdims::DenseConvDims,
) where {T <: MIOPENFloat, N}
    if !NNlib.flipkernel(cdims)
        @warn """
        MIOpen supports only cross-correlation (flipkernel=true).
        Therefore for every regular convolution (flipkernel=false)
        kernel is flipped before calculation.
        For better performance, use cross-correlation (flipkernel=true)
        and manually flip the kernel before `NNlib.conv` call.
        """ maxlog=1
        flip_dims = ntuple(
            i -> (i ≤ ndims(w) - 2) ? (size(w, i):-1:1) : Colon(),
            ndims(w))
        w = w[flip_dims...]
    end

    nd = max(0, 4 - N)
    ncdims = NNlib.insert_singleton_spatial_dimension(cdims, nd)
    MIOpen.convolution!(
        NNlib.insert_singleton_spatial_dimension(y, nd),
        NNlib.insert_singleton_spatial_dimension(x, nd),
        NNlib.insert_singleton_spatial_dimension(w, nd);
        padding=nnlib_padding(ncdims), stride=NNlib.stride(ncdims),
        dilation=NNlib.dilation(ncdims), groups=NNlib.groupcount(ncdims))
    return y
end

function NNlib.∇conv_data!(
    dx::ROCArray{T, N}, dy::ROCArray{T, N}, w::ROCArray{T, N}, cdims::DenseConvDims,
) where {T <: MIOPENFloat, N}
    if !NNlib.flipkernel(cdims)
        @warn """
        MIOpen supports only cross-correlation (flipkernel=true).
        Therefore for every regular convolution (flipkernel=false)
        kernel is flipped before calculation.
        For better performance, use cross-correlation (flipkernel=true)
        and manually flip the kernel before `NNlib.conv` call.
        """ maxlog=1
        flip_dims = ntuple(
            i -> (i ≤ ndims(w) - 2) ? (size(w, i):-1:1) : Colon(),
            ndims(w))
        w = w[flip_dims...]
    end

    nd = max(0, 4 - N)
    ncdims = NNlib.insert_singleton_spatial_dimension(cdims, nd)
    MIOpen.∇convolution_data!(
        NNlib.insert_singleton_spatial_dimension(dx, nd),
        NNlib.insert_singleton_spatial_dimension(dy, nd),
        NNlib.insert_singleton_spatial_dimension(w, nd);
        padding=nnlib_padding(ncdims), stride=NNlib.stride(ncdims),
        dilation=NNlib.dilation(ncdims), groups=NNlib.groupcount(ncdims))
    return dx
end

function NNlib.∇conv_filter!(
    dw::ROCArray{T, N}, x::ROCArray{T, N}, dy::ROCArray{T, N}, cdims::DenseConvDims,
) where {T <: MIOPENFloat, N}
    nd = max(0, 4 - N)
    ncdims = NNlib.insert_singleton_spatial_dimension(cdims, nd)
    MIOpen.∇convolution_weight!(
        NNlib.insert_singleton_spatial_dimension(dw, nd),
        NNlib.insert_singleton_spatial_dimension(dy, nd),
        NNlib.insert_singleton_spatial_dimension(x, nd);
        padding=nnlib_padding(ncdims), stride=NNlib.stride(ncdims),
        dilation=NNlib.dilation(ncdims), groups=NNlib.groupcount(ncdims))

    if !NNlib.flipkernel(cdims)
        @warn """
        MIOpen supports only cross-correlation (flipkernel=true).
        Therefore for every regular convolution (flipkernel=false)
        kernel is flipped before calculation.
        For better performance, use cross-correlation (flipkernel=true)
        and manually flip the kernel before `NNlib.conv` call.
        """ maxlog=1
        flip_dims = ntuple(
            i -> (i ≤ ndims(dw) - 2) ? (size(dw, i):-1:1) : Colon(),
            ndims(dw))
        dw = dw[flip_dims...]
    end
    return dw
end
