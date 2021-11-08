
export dense_bias_act, bias_act!

"""
    dense_bias_act(σ, w, x, b)
    dense_bias_act(σ, w, x, w′, x′, b)

This is equivalent to `σ.((w * x) .+ b)`, but should be more efficient.
Or to `σ.((w * x) .+ (w′ * x′) .+ b)` for the 5-argument form.

Calls [`bias_act!`](@ref), which replaces `tanh` with `tanh_fast`,
and fuses the broadcast. (It mutates only the intermediate 
result `w * x` allocated within this function).

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
1. overwrite `x` to save memory, when possible,
2. fuse the computation of the the gradient,
3. replace `sigmoid` & `tanh` with `sigmoid_fast` & `tanh_fast`.

The greatest re-use will be when `x isa StridedArray{<:AbstractFloat}`,
since `x::Array{Int}` and `b::Vector{Dual}` and can't work in-place.

And when the activation has a method of `derivatives_given_output` which does
not need the input at all. This is defined by e.g. `@scalar_rule relu(x) (Ω > 0)`,
where `(x > 0)` would give the same results but need to keep `x` around.
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

This replaces `σ == tanh` with `tanh_fast`, etc.
Takes a 2nd argument so that this replacement could be disabled for CUDA.
"""
@inline fast_act(σ, ::AbstractArray) = σ
@inline fast_act(::typeof(tanh), ::AbstractArray) = tanh_fast
@inline fast_act(::typeof(sigmoid), ::AbstractArray) = fast_sigmoid

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

julia> w, b = rand(Float32, 100, 100), rand(Float32, 100);

julia> @btime bias_act!(relu, $w, $b);
  4.173 μs (3 allocations: 96 bytes)

julia> @btime relu.($w .+ $b);  # faster, that's odd?
  1.825 μs (2 allocations: 39.11 KiB)

julia> @btime bias_act!(tanh, $w, $b);  # now with tanh_fast
  33.208 μs (3 allocations: 96 bytes)

julia> @btime tanh.($w .+ $b);
  66.041 μs (2 allocations: 39.11 KiB)

julia> @btime tanh_fast.($w .+ $b);  # saves 57 μs
  8.486 μs (2 allocations: 39.11 KiB)

# gradients:

julia> using Zygote

julia> @btime gradient((w,b) -> sum(bias_act!(relu, w, b)), $w, $b);  # slower!
  20.291 μs (58 allocations: 41.55 KiB)

julia> @btime gradient((w,b) -> sum(relu.(w .+ b)), $w, $b);
  16.166 μs (30 allocations: 118.64 KiB)

julia> @btime gradient((w,b) -> sum(bias_act!(tanh, w, b)), $w, $b);  # now with tanh_fast
  47.250 μs (58 allocations: 41.55 KiB)

julia> @btime gradient((w,b) -> sum(tanh.(w .+ b)), $w, $b);
  75.583 μs (30 allocations: 118.64 KiB)

julia> @btime gradient((w,b) -> sum(bias_act!(tanh, wr[], b)), wr, $b)  setup=(wr=Ref(randn(Float32,100,100))) evals=1;
  48.458 μs (76 allocations: 42.66 KiB)

julia> @btime gradient((w,b) -> sum(tanh.(w .+ b)), wr[], $b)  setup=(wr=Ref(randn(Float32,100,100))) evals=1;
  96.833 μs (30 allocations: 118.64 KiB)


# with matmul too:

julia> w, b = rand(Float32, 100, 100), rand(Float32, 100); x = rand(Float32, size(w)...);

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(relu, w, x, b)), $w, $x, $b);
  152.542 μs (70 allocations: 159.26 KiB)

julia> @btime gradient((w,x,b) -> sum(relu.((w * x) .+ b)), $w, $x, $b);
  151.083 μs (37 allocations: 236.14 KiB)

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(tanh, w, x, b)), $w, $x, $b);
  185.958 μs (70 allocations: 159.26 KiB)

julia> @btime gradient((w,x,b) -> sum(tanh.((w * x) .+ b)), $w, $x, $b);  # faster, WTF?
  153.167 μs (37 allocations: 236.14 KiB)

julia> @btime gradient((w,x,b) -> sum(tanh_fast.((w * x) .+ b)), $w, $x, $b);  # why doesn't this save 57 μs
  149.791 μs (37 allocations: 236.14 KiB)

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(identity, w, x, b)), $w, $x, $b);
  604.000 μs (53 allocations: 189.16 KiB)
julia> @btime gradient((w,x,b) -> sum(identity.((w * x) .+ b)), $w, $x, $b);  # Oh, this hits generic matmul
  602.583 μs (31 allocations: 227.23 KiB)

# ... two of them:

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(relu, w, x, w, x, b)), $w, $x, $b);
  290.084 μs (84 allocations: 316.29 KiB)

julia> @btime gradient((w,x,b) -> sum(relu.((w * x) .+ (w * x) .+ b)), $w, $x, $b);
  287.125 μs (49 allocations: 471.08 KiB)


# memory -- not half anymore

julia> 314.59 / 470.14
0.6691411069043264

julia> @btime copy($w);
  2.635 μs (2 allocations: 78.17 KiB)

julia> 314.59 / 78.17
4.02443392605859

=#
