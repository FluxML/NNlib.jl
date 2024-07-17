for (f, op) in [
        NNlib.relu => MIOpen.relu,
        NNlib.relu6 => x -> MIOpen.clippedrelu(x, 6),
        NNlib.softplus => MIOpen.softrelu,
        NNlib.σ => MIOpen.sigmoid,
        Base.tanh => MIOpen.tanh,
        # TODO define for leakyrelu, elu, etc.?
    ], N in 1:5
    @eval function Base.materialize(
        bc::Broadcast.Broadcasted{<:Any,<:Any,typeof($f),<:Tuple{ROCArray{<:MIOPENFloat,$N}}}
    )
        return $op(bc.args[1])
    end
end

Base.broadcasted(::typeof(identity), x::ROCArray{T}) where {T<:MIOPENFloat} = x
