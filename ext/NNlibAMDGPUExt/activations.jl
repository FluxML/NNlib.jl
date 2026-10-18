for (f, op) in [
        NNlib.relu => MIOpen.relu,
        NNlib.relu6 => x -> MIOpen.clippedrelu(x, 6),
        # Not softplus: MIOpen's softrelu adds 1 + exp(x) in the input type, which
        # for Float16 is off by up to 77% at x < 0 and returns 0 below x = -7.6.
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
