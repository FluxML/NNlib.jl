import NNlib: maxpool!, ∇maxpool!, meanpool!, ∇meanpool!

pool_kwargs(pdims::PoolDims) =
    (window=NNlib.kernel_size(pdims),
     padding=NNlib.padding(pdims),
     stride=NNlib.stride(pdims))

function maxpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims) where T<:CUDNNFloat
    cuDNN.maxpool!(y, x; pool_kwargs(pdims)...)
end

function ∇maxpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims;
                   alpha=1, beta=0, kwargs...) where T<:CUDNNFloat
    cuDNN.∇maxpool!(dx, dy, y, x; pool_kwargs(pdims)..., alpha, beta)
end

function meanpool!(y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims;
                   count_include_pad::Bool=true) where T<:CUDNNFloat
    cuDNN.meanpool!(y, x; pool_kwargs(pdims)..., count_include_pad)
end

function ∇meanpool!(dx::DenseCuArray{T}, dy::DenseCuArray{T}, y::DenseCuArray{T}, x::DenseCuArray{T}, pdims::PoolDims;
                    count_include_pad::Bool=true, alpha=1, beta=0, kwargs...) where T<:CUDNNFloat
    cuDNN.∇meanpool!(dx, dy, y, x; pool_kwargs(pdims)..., count_include_pad, alpha, beta)
end

### Preserve NNlib's 1D pooling promotion.

add1d(x) = reshape(x, 1, size(x)...)

function fix_pooldims_1d(pdims::PoolDims{1,K,S,P,D}) where {K,S,P,D}
    PoolDims{2, K + 1, S + 1, P + 2, D + 1}((1, NNlib.input_size(pdims)...),
                                            (1, NNlib.kernel_size(pdims)...),
                                            NNlib.channels_in(pdims),
                                            (1, NNlib.stride(pdims)...),
                                            (0, 0, NNlib.padding(pdims)...),
                                            (1, NNlib.dilation(pdims)...))
end

function maxpool!(y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims) where T<:CUDNNFloat
    maxpool!(add1d(y), add1d(x), fix_pooldims_1d(pdims))
    return y
end

function meanpool!(y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims;
                   count_include_pad::Bool=true) where T<:CUDNNFloat
    meanpool!(add1d(y), add1d(x), fix_pooldims_1d(pdims); count_include_pad)
    return y
end

function ∇maxpool!(dx::DenseCuArray{T,3}, dy::DenseCuArray{T,3}, y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims; kwargs...) where T<:CUDNNFloat
    ∇maxpool!(add1d(dx), add1d(dy), add1d(y), add1d(x), fix_pooldims_1d(pdims); kwargs...)
    return dx
end

function ∇meanpool!(dx::DenseCuArray{T,3}, dy::DenseCuArray{T,3}, y::DenseCuArray{T,3}, x::DenseCuArray{T,3}, pdims::PoolDims;
                    count_include_pad::Bool=true, kwargs...) where T<:CUDNNFloat
    ∇meanpool!(add1d(dx), add1d(dy), add1d(y), add1d(x), fix_pooldims_1d(pdims); count_include_pad, kwargs...)
    return dx
end
