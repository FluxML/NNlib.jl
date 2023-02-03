for poolname in (:maxpool, :meanpool)
    @eval function NNlib.$(poolname)(
        x::ROCArray{T, N}, pdims::PoolDims,
    ) where {T <: MIOPENFloat, N}
        y = similar(x, NNlib.output_size(pdims)..., NNlib.channels_out(pdims), size(x, N))
        nd = max(0, 4 - N)
        npdims = NNlib.insert_singleton_spatial_dimension(pdims, nd)
        MIOpen.$(Symbol("$(poolname)!"))(
            NNlib.insert_singleton_spatial_dimension(y, nd),
            NNlib.insert_singleton_spatial_dimension(x, nd);
            dims=NNlib.kernel_size(npdims), padding=nnlib_padding(npdims),
            stride=NNlib.stride(npdims), do_backward=false)
        return y
    end

    @eval function ChainRulesCore.rrule(
        ::typeof(NNlib.$(poolname)), x::ROCArray{T, N}, pdims::PoolDims,
    ) where {T <: MIOPENFloat, N}
        y = similar(x, NNlib.output_size(pdims)..., NNlib.channels_out(pdims), size(x, N))
        nd = max(0, 4 - N)
        npdims = NNlib.insert_singleton_spatial_dimension(pdims, nd)

        # `workspace` is used in the pullback.
        _, workspace = MIOpen.$(Symbol("$(poolname)!"))(
            NNlib.insert_singleton_spatial_dimension(y, nd),
            NNlib.insert_singleton_spatial_dimension(x, nd);
            dims=NNlib.kernel_size(npdims), padding=nnlib_padding(npdims),
            stride=NNlib.stride(npdims))

        function _pooling_pullback(Δ)
            dx = similar(x)
            MIOpen.$(Symbol("∇$(poolname)!"))(
                NNlib.insert_singleton_spatial_dimension(dx, nd),
                NNlib.insert_singleton_spatial_dimension(unthunk(Δ), nd),
                NNlib.insert_singleton_spatial_dimension(y, nd),
                NNlib.insert_singleton_spatial_dimension(x, nd);
                dims=NNlib.kernel_size(npdims), padding=nnlib_padding(npdims),
                stride=NNlib.stride(npdims), workspace)
            return NoTangent(), dx, NoTangent()
        end
        y, _pooling_pullback
    end
end
