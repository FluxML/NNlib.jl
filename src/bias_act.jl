
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

The greatest re-use requires, first, that `x isa StridedArray{<:AbstractFloat}`,
since `x::Array{Int}` and `b::Vector{Dual}` and can't work in-place.

And, second, that the activation has a method of `derivatives_given_output` which does
not need the input at all. This is defined by e.g. `@scalar_rule relu(x) (Ω > 0)`,
where `(x > 0)` would give the same results, but need to keep `x` around.
"""
bias_act!(σ::F, x::AbstractArray, b=false) where {F<:Function} = fast_act(σ, x).(x .+ b)
bias_act!(σ::F, x::StridedArray{<:AbstractFloat}, b::AbstractArray{<:AbstractFloat}) where {F<:Function} =
    # x .= fast_act(σ, x).(x .+ b)  # this is what we want, but is slow because of JuliaLang/julia/issues/43153
    # fast_act(σ, x).(x .+ b)  # for testing, faster but allocates
    fast_broadcast!(σ, x, b)  # hand-written version below.
bias_act!(σ::F, x::StridedArray{<:AbstractFloat}, b::Bool=false) where {F<:Function} =
    # b ? (x .= fast_act(σ, x).(x .+ b)) : (x .= fast_act(σ, x).(x))
    # b ? (fast_act(σ, x).(x .+ b)) : (fast_act(σ, x).(x))
    fast_broadcast!(σ, x, b)
bias_act!(::typeof(identity), x::StridedArray{<:AbstractFloat}, b::Bool=false) =
    b ? (x .+= 1) : x

# This has no methods, used for testing whether `derivatives_given_output(Ω, f, x)`
# is independent of `x`:
struct NotaNumber <: Real end

@inline function rrule(cfg::RuleConfig{>:HasReverseMode}, ::typeof(bias_act!), σ::F, x::T, b::B) where {F, T, B}
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
        @inline function bias_act!_fastback(Δ)
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
        @inline function bias_act!_back(Δ)
            dx = first.(first.(derivatives_given_output.(Ω, σ, x))) .* unthunk(Δ)
            db = eltype(B) == Bool ? NoTangent() : proj_b(sum(dx; dims = b_dims))
            (NoTangent(), NoTangent(), proj_x(dx), db)
        end
        return Ω, bias_act!_back
    else
        # Fallback path: let AD handle the broadcast
        Ω, back = rrule_via_ad(cfg, broadcast, σ, bias_act!(identity, x, b))
        @inline function bias_act!_slowback(Δ)
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



"""
    NNlib.fast_broadcast!(σ, x, b)

This is equivalent to `x .= fast_act(σ, x).(x .+ b)`, but works around
an issue with broadcasting that prevents SIMD in such cases.

Can be removed once https://github.com/JuliaLang/julia/issues/43153 is fixed.
"""
function fast_broadcast!(σ::F, x::Array{<:AbstractFloat}, b) where {F<:Function}
    f = fast_act(σ, x)
    if b === false
        @simd ivdep for I in eachindex(x)
            @inbounds x[I] = f(x[I])
        end
    else
        xplus = Broadcast.instantiate(Broadcast.broadcasted(+, x, b))
        @simd ivdep for I in eachindex(xplus)
            @inbounds x[I] = f(xplus[I])
        end
    end
    return x
end
function fast_broadcast!(σ::F, x::StridedArray{<:AbstractFloat}, b) where {F<:Function}
    # CuArray has its own broadcasting.
    x .= fast_act(σ, x).(x .+ b)
    return x
end

#=

# Some benchmarks, Julia 1.8 + M1
# Note the mean times, which include GC


julia> using NNlib, BenchmarkTools

julia> w, b = rand(Float32, 100, 100), rand(Float32, 100);

julia> @btime bias_act!(relu, $w, $b);
  min 1.587 μs, mean 1.614 μs (0 allocations)

julia> @btime relu.($w .+ $b);
  min 1.833 μs, mean 4.953 μs (2 allocations, 39.11 KiB)

julia> @btime bias_act!(tanh, $w, $b);  # using tanh_fast
  min 6.300 μs, mean 6.359 μs (0 allocations)

julia> @btime tanh.($w .+ $b);
  min 60.500 μs, mean 64.928 μs (2 allocations, 39.11 KiB)

julia> @btime tanh_fast.($w .+ $b);  # saves 57 μs
  min 6.467 μs, mean 9.421 μs (2 allocations, 39.11 KiB)



########## gradients:

julia> using Zygote

julia> @btime gradient((w,b) -> sum(bias_act!(relu, w, b)), $w, $b);  # slower!
  min 19.583 μs, mean 27.610 μs (55 allocations, 41.46 KiB)

julia> @btime gradient((w,b) -> sum(relu.(w .+ b)), $w, $b);
  min 18.750 μs, mean 32.133 μs (30 allocations, 118.64 KiB)

julia> @btime gradient((w,b) -> sum(bias_act!(tanh, w, b)), $w, $b);  # now with tanh_fast
  min 24.875 μs, mean 29.964 μs (55 allocations, 41.46 KiB)

julia> @btime gradient((w,b) -> sum(tanh.(w .+ b)), $w, $b);
  min 73.583 μs, mean 85.504 μs (30 allocations, 118.64 KiB)

# repeat those with 1 eval:

julia> @btime gradient((w,b) -> sum(bias_act!(tanh, wr[], b)), wr, $b)  setup=(wr=Ref(randn(Float32,100,100))) evals=1;
  min 25.000 μs, mean 32.429 μs (73 allocations, 42.57 KiB)

julia> @btime gradient((w,b) -> sum(tanh.(w .+ b)), wr[], $b)  setup=(wr=Ref(randn(Float32,100,100))) evals=1;
  min 127.333 μs, mean 142.678 μs (30 allocations, 118.64 KiB)



########## with matmul too:
# The reason to put sum(abs2, x) is to ensure you never get a FillArray into matmul.

julia> w, b = rand(Float32, 100, 100), rand(Float32, 100); x = rand(Float32, size(w)...);

julia> @btime gradient((w,x,b) -> sum(abs2, x), $w, $x, $b);  # baseline
  min 3.135 μs, mean 7.540 μs (2 allocations, 39.11 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, dense_bias_act(relu, w, x, b)), $w, $x, $b);
  min 38.584 μs, mean 61.920 μs (68 allocations, 198.21 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, relu.((w * x) .+ b)), $w, $x, $b);
  min 33.084 μs, mean 60.657 μs (39 allocations, 275.25 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, dense_bias_act(tanh, w, x, b)), $w, $x, $b);
  min 42.166 μs, mean 64.340 μs (68 allocations, 198.21 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, tanh.((w * x) .+ b)), $w, $x, $b);  # faster, WTF? Tooke 127.333 μs without matmul?
  min 40.958 μs, mean 67.304 μs (39 allocations, 275.25 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, tanh_fast.((w * x) .+ b)), $w, $x, $b);  # why doesn't this save 57 μs
  min 37.500 μs, mean 74.563 μs (39 allocations, 275.25 KiB)

# repeat those with 1 eval:

julia> @btime gradient((w,x,b) -> sum(abs2, dense_bias_act(tanh, w, x, b)), wr[], $x, $b)  setup=(wr=Ref(randn(Float32,100,100))) evals=1;
  min 44.417 μs, mean 82.670 μs (68 allocations, 198.21 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, tanh.((w * x) .+ b)), wr[], $x, $b)  setup=(wr=Ref(randn(Float32,100,100))) evals=1;
  min 113.250 μs, mean 157.216 μs (39 allocations, 275.25 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, tanh_fast.((w * x) .+ b)), wr[], $x, $b)  setup=(wr=Ref(randn(Float32,100,100))) evals=1;
  min 39.625 μs, mean 78.753 μs (39 allocations, 275.25 KiB)


# ... two of them:

julia> @btime gradient((w,x,b) -> sum(abs2, dense_bias_act(relu, w, x, w, x, b)), $w, $x, $b);
  min 67.333 μs, mean 113.888 μs (82 allocations, 355.24 KiB)

julia> @btime gradient((w,x,b) -> sum(abs2, relu.((w * x) .+ (w * x) .+ b)), $w, $x, $b);
  min 61.583 μs, mean 120.099 μs (51 allocations, 510.19 KiB)



# memory -- not half anymore, but still a saving

julia> (198.21 - 39.11) / (275.25 - 39.11)
0.6737528584737869

julia> @btime copy($w);
  min 807.000 ns, mean 4.038 μs (2 allocations, 39.11 KiB)

julia> (275.25 - 39.11) / 78.17
3.020851989254189

julia> (198.21 - 39.11) / 78.17
2.035307662786235

=#
