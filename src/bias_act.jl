
export dense_bias_act, bias_act!

"""
    dense_bias_act(σ, w, x, b)
    dense_bias_act(σ, w, x, w′, x′, b)

This is equivalent to `σ.((w * x) .+ b)`, but should be more efficient.
Or to `σ.((w * x) .+ (w′ * x′) .+ b)` for the 5-argument form.

Calls [`bias_act!`](@ref), which mutates only the intermediate 
result `w * x` allocated within this function.

See also [`conv_bias_act`](@ref).
"""
dense_bias_act(σ, w, x, b=false) = bias_act!(σ, w * x, b)
dense_bias_act(σ, w, x, ww, xx, b=false) = bias_act!(σ, muladd!(w, x, ww * xx), b)

"""
    muladd!(w, x, z) == muladd(w, x, z)
                     == (w * x) + z
                     == mul!(z, w, x, true, true)

This variant of `muladd` overwrites its *last* argument.
Expects `size(w*x) == size(z)`
Unlike `mul!`, it has a gradient rule.
"""
muladd!(A, B, C) = mul!(C, A, B, true, true)

function ChainRulesCore.rrule(::typeof(muladd!), A, B, C)
    proj_C = ProjectTo(C)
    function muladd!_back(dZ0)
        dZ = unthunk(dZ0)
        (NoTangent(), ProjectTo(A)(@thunk dZ * B'), ProjectTo(B)(@thunk A' * dZ), proj_C(dZ))
    end
    return muladd!(A, B, C), muladd!_back
end

"""
    bias_act!(σ, x, b)

This is equivalent to `σ.(x .+ b)`, but faster because it will:
1. overwrite `x` to save memory, when safe,
2. fuse the computation of the the gradient,
3. replace `sigmoid` & `tanh` with `sigmoid_fast` & `tanh_fast`.

The greatest re-use will be when `x isa StridedArray{<:AbstractFloat}`,
and the activation has a method of `derivatives_given_output` which does
not need the input at all.
`x::Array{Int}` and `b::Vector{Dual}` and won't work in-place.
"""
bias_act!(σ::Function, x::AbstractArray, b=false) = fast_act(σ, x).(x .+ b)
bias_act!(σ::Function, x::StridedArray{<:AbstractFloat}, b::AbstractArray{<:Union{Bool, AbstractFloat}}) =
    x .= fast_act(σ, x).(x .+ b)
bias_act!(σ::Function, x::StridedArray{<:AbstractFloat}, b::Bool=false) =
    b ? (x .= fast_act(σ, x).(x .+ b)) : (x .= fast_act(σ, x).(x))
bias_act!(::typeof(identity), x::StridedArray{<:AbstractFloat}, b::Bool=false) =
    b ? (x .+= 1) : x

"""
    NNlib.fast_act(σ, x::AbstractArray)

This replaces `σ == tanh` with `fast_tanh`, etc.
Takes a 2nd argument so that this replacement could be disabled for CUDA.
"""
fast_act(σ, ::AbstractArray) = σ
# fast_act(::typeof(tanh), ::AbstractArray) = fast_tanh
# fast_act(::typeof(sigmoid), ::AbstractArray) = fast_sigmoid

# This has no methods, used for testing whether `derivatives_given_output(Ω, f, x)`
# is independent of `x`:
struct NotaNumber <: Real end

function rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(bias_act!), σ::F, x::T, b::B) where {F, T, B}
    if eltype(B) != Bool
        # This allows for conv layers whose bias vector has been reshaped to feature dim:
        b_dims = ntuple(d -> size(b, d)==1 ? d : ndims(x)+1, ndims(x))
        # For b::Vector, proj_b will drop trivial dimensions for us, i.e. trivial reshape:
        proj_b = ProjectTo(b)
    end
    proj_x = ProjectTo(x)
    if isconcretetype(Core.Compiler._return_type(
            derivatives_given_output, Tuple{eltype(T), F, NotaNumber}))
        # Fast path: it is now safe to overwrite x, since this is not needed for gradient of σ
        Ω = bias_act!(σ, x, b)  # now x === Ω, most likely
        function bias_act!_fastback(Δ)
            # Tempting to overwrite x again, but only safe if you call pullback at most once:
            dx = first.(first.(derivatives_given_output.(Ω, σ, NotaNumber()))) .* unthunk(Δ)
            db = eltype(B) == Bool ? NoTangent() : proj_b(sum(dx; dims = b_dims))
            (NoTangent(), NoTangent(), proj_x(dx), db)
        end
        return Ω, bias_act!_fastback
    elseif isconcretetype(Core.Compiler._return_type(
            derivatives_given_output, Tuple{eltype(T), F, eltype(T)}))
        # Slower path: can't overwrite x, but can use derivatives_given_output
        Ω = σ.(x) .+ b
        function bias_act!_back(Δ)
            dx = first.(first.(derivatives_given_output.(Ω, σ, x))) .* unthunk(Δ)
            db = eltype(B) == Bool ? NoTangent() : proj_b(sum(dx; dims = b_dims))
            (NoTangent(), NoTangent(), proj_x(dx), db)
        end
        return Ω, bias_act!_back
    else
        # Fallback path: let AD handle the broadcast
        Ω, back = rrule_via_ad(cfg, broadcast, σ, bias_act!(identity, x, b))
        function bias_act!_slowback(Δ)
            _, _, dx = back(Δ)
            db = eltype(B) == Bool ? NoTangent() : proj_b(sum(dx; dims = b_dims))
            (NoTangent(), NoTangent(), proj_x(dx), db)
        end
        return Ω, bias_act!_slowback
    end
end

function rrule(::typeof(bias_act!), σ::typeof(identity), x::T, b::B) where {T, B}
    if eltype(B) != Bool
        b_dims = ntuple(d -> size(b, d)==1 ? d : ndims(x)+1, ndims(x))
        proj_b = ProjectTo(b)
    end
    proj_x = ProjectTo(x)
    function bias_act!_idback(Δ)
        if eltype(B) == Bool
            (NoTangent(), NoTangent(), proj_x(unthunk(Δ)), NoTangent())
        else
            dx = unthunk(Δ)
            db = proj_b(sum(dx; dims = b_dims))
            (NoTangent(), NoTangent(), proj_x(dx), db)
        end
    end
    return bias_act!(σ, x, b), bias_act!_idback
end

#=

# Some benchmarks, Julia 1.7 + M1 rosetta.


julia> using NNlib, BenchmarkTools

julia> w, b = rand(100, 100), rand(100);

julia> @btime bias_act!(relu, $w, $b);
  4.327 μs (2 allocations: 64 bytes)

julia> @btime relu.($w .+ $b);  # faster, that's odd?
  3.323 μs (2 allocations: 78.17 KiB)

julia> @btime bias_act!(tanh, $w, $b);
  96.084 μs (2 allocations: 64 bytes)

julia> @btime tanh.($w .+ $b);
  98.125 μs (2 allocations: 78.17 KiB)


# gradients:

julia> using Zygote

julia> @btime gradient((w,b) -> sum(bias_act!(relu, w, b)), $w, $b);
  22.458 μs (59 allocations: 81.05 KiB)

julia> @btime gradient((w,b) -> sum(relu.(w .+ b)), $w, $b);
  20.166 μs (30 allocations: 236.22 KiB)

julia> @btime gradient((w,b) -> sum(bias_act!(tanh, w, b)), $w, $b);
  115.041 μs (59 allocations: 81.05 KiB)

julia> @btime gradient((w,b) -> sum(tanh.(w .+ b)), $w, $b);
  117.250 μs (30 allocations: 236.22 KiB)


# with matmul too:

julia> w, b = rand(100, 100), rand(100); x = rand(size(w)...);

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(relu, w, x, b)), $w, $x, $b);
  195.209 μs (71 allocations: 315.95 KiB)

julia> @btime gradient((w,x,b) -> sum(relu.((w * x) .+ b)), $w, $x, $b);
  190.000 μs (37 allocations: 470.91 KiB)

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(tanh, w, x, b)), $w, $x, $b);
  226.000 μs (71 allocations: 315.95 KiB)

julia> @btime gradient((w,x,b) -> sum(tanh.((w * x) .+ b)), $w, $x, $b);
  223.917 μs (37 allocations: 470.91 KiB)

# ... two of them:

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(relu, w, x, w, x, b)), $w, $x, $b);
  367.875 μs (85 allocations: 629.23 KiB)

julia> @btime gradient((w,x,b) -> sum(relu.((w * x) .+ (w * x) .+ b)), $w, $x, $b);
  379.000 μs (49 allocations: 940.22 KiB)


# memory -- not half anymore

julia> 314.59 / 470.14
0.6691411069043264

julia> @btime copy($w);
  2.635 μs (2 allocations: 78.17 KiB)

julia> 314.59 / 78.17
4.02443392605859

=#
