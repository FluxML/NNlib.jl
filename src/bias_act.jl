
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
# bias_act!(σ1::F, x::AbstractArray, b=false) where {F} = σ1.(x .+ b)
# bias_act!(σ1::F, x::StridedArray{<:AbstractFloat}, b::AbstractArray{<:Union{Bool, AbstractFloat}}) where {F} =
#     x .= σ1.(x .+ b)
# bias_act!(σ::F, x::StridedArray{<:AbstractFloat}, b::Bool=false) where {F<:Function} =
#     b ? (x .= σ.(x .+ b)) : (x .= fast_act(σ, x).(x))
# bias_act!(::typeof(identity), x::StridedArray{<:AbstractFloat}, b::Bool=false) =
#     b ? (x .+= 1) : x

bias_act!(σ::F, x::AbstractArray, b=false) where {F<:Function} = fast_act(σ, x).(x .+ b)
bias_act!(σ::F, x::StridedArray{<:AbstractFloat}, b::AbstractArray{<:Union{Bool, AbstractFloat}}) where {F<:Function} =
    x .= fast_act(σ, x).(x .+ b)
    # fast_act(σ, x).(x .+ b)
bias_act!(σ::F, x::StridedArray{<:AbstractFloat}, b::Bool=false) where {F<:Function} =
    b ? (x .= fast_act(σ, x).(x .+ b)) : (x .= fast_act(σ, x).(x))
    # b ? (fast_act(σ, x).(x .+ b)) : (fast_act(σ, x).(x))
bias_act!(::typeof(identity), x::StridedArray{<:AbstractFloat}, b::Bool=false) =
    b ? (x .+= 1) : x

"""
    NNlib.fast_act(σ, x::AbstractArray)

This replaces `σ == tanh` with `tanh_fast`, etc.
Takes a 2nd argument so that this replacement could be disabled for CUDA.
"""
@inline fast_act(σ::F, ::AbstractArray) where {F<:Function} = σ
@inline fast_act(::typeof(tanh), ::AbstractArray) = tanh_fast
@inline fast_act(::typeof(sigmoid), ::AbstractArray) = fast_sigmoid

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


# Fixing https://github.com/JuliaLang/julia/issues/43153
# obviously this is piracy and should not be merged!

@eval Base.Broadcast @inline function copyto!(dest::AbstractArray, bc::Broadcasted{Nothing})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && bc.args isa Tuple{AbstractArray} # only a single input argument to broadcast!
        A = bc.args[1]
        if axes(dest) == axes(A)
            return copyto!(dest, A)
        end
    end
    bc′ = preprocess(dest, bc)
    # Performance may vary depending on whether `@inbounds` is placed outside the
    # for loop or not. (cf. https://github.com/JuliaLang/julia/issues/38086)
    # @simd for I in eachindex(bc′)
    #     @inbounds dest[I] = bc′[I]
    # end
    if dest isa StridedArray{<:Base.HWNumber}
        @simd ivdep for I in eachindex(dest)
           @inbounds dest[I] = bc′[I]
        end
    else
       @simd for I in eachindex(dest)
           @inbounds dest[I] = bc′[I]
       end
    end
    return dest
end

#=

# Some benchmarks, Julia 1.7 + M1 rosetta.


julia> using NNlib, BenchmarkTools

julia> w, b = rand(Float32, 100, 100), rand(Float32, 100);

julia> @btime bias_act!(relu, $w, $b);
  4.173 μs (3 allocations: 96 bytes)
  1.042 μs (0 allocations: 0 bytes)  # after fixing broadcast!

julia> @btime relu.($w .+ $b);  # faster, that's odd?
  1.825 μs (2 allocations: 39.11 KiB)
  1.488 μs (2 allocations: 39.11 KiB)

julia> @btime bias_act!(tanh, $w, $b);  # now with tanh_fast
  33.208 μs (3 allocations: 96 bytes)
  7.239 μs (0 allocations: 0 bytes)

julia> @btime tanh.($w .+ $b);
  66.041 μs (2 allocations: 39.11 KiB)
  19.416 μs (2 allocations: 39.11 KiB)

julia> @btime tanh_fast.($w .+ $b);  # saves 57 μs
  8.486 μs (2 allocations: 39.11 KiB)
  7.136 μs (2 allocations: 39.11 KiB)

# These times miss the cost of allocations. E.g. mean 9 μs here:

julia> @benchmark bias_act!(tanh, $w, $b)
┌ Trial:
│  min 7.250 μs, median 7.281 μs, mean 7.343 μs, 99ᵗʰ 8.979 μs
│  0 allocations
│         ◑◕ *
│         █
│  ▁▁▁▁▁▁▅█▆▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▁▁▁▁▂▁▁▂▂▁▂▂▂▂▂▁▂ ▂
└  7.1 μs                  10_000 samples, each 4 evaluations                    9 μs +

julia> @benchmark tanh_fast.($w .+ $b)
┌ Trial:
│  min 7.177 μs, median 7.375 μs, mean 9.077 μs, 99ᵗʰ 20.542 μs
│  2 allocations, total 39.11 KiB
│  GC time: mean 1.261 μs (13.89%), max 600.552 μs (97.72%)
│   ◑         *
│   █
│  ▂█▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▁▂▂▁▂▂▂▂▂▁▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▁▂▂▂▂▂▂▂▁ ▂
└  7.1 μs                  10_000 samples, each 4 evaluations                   21 μs +


# gradients:

julia> using Zygote

julia> @btime gradient((w,b) -> sum(bias_act!(relu, w, b)), $w, $b);  # slower!
  20.291 μs (58 allocations: 41.55 KiB)
  15.042 μs (55 allocations: 41.46 KiB)

julia> @btime gradient((w,b) -> sum(relu.(w .+ b)), $w, $b);
  16.166 μs (30 allocations: 118.64 KiB)
  13.666 μs (30 allocations: 118.64 KiB)

julia> @btime gradient((w,b) -> sum(bias_act!(tanh, w, b)), $w, $b);  # now with tanh_fast
  47.250 μs (58 allocations: 41.55 KiB)
  19.500 μs (55 allocations: 41.46 KiB)

julia> @btime gradient((w,b) -> sum(tanh.(w .+ b)), $w, $b);
  75.583 μs (30 allocations: 118.64 KiB)
  43.708 μs (30 allocations: 118.64 KiB)

julia> @btime gradient((w,b) -> sum(bias_act!(tanh, wr[], b)), wr, $b)  setup=(wr=Ref(randn(Float32,100,100))) evals=1;
  48.458 μs (76 allocations: 42.66 KiB)
  20.584 μs (73 allocations: 42.57 KiB)

julia> @btime gradient((w,b) -> sum(tanh.(w .+ b)), wr[], $b)  setup=(wr=Ref(randn(Float32,100,100))) evals=1;
  96.833 μs (30 allocations: 118.64 KiB)
  79.167 μs (30 allocations: 118.64 KiB)

julia> @benchmark gradient((w,b) -> sum(bias_act!(relu, w, b)), $w, $b)
┌ Trial:
│  min 15.209 μs, median 15.708 μs, mean 19.257 μs, 99ᵗʰ 35.251 μs
│  55 allocations, total 41.46 KiB
│  GC time: mean 1.541 μs (8.00%), max 1.786 ms (96.01%)
│    ◑◕            *
│   ▃█▁
│  ▂███▄▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▂▂▁▂▂▁▂▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂ ▂
└  15 μs                    10_000 samples, each 1 evaluation                   36 μs +

julia> @benchmark gradient((w,b) -> sum(relu.(w .+ b)), $w, $b)
┌ Trial:
│  min 14.000 μs, median 15.542 μs, mean 19.660 μs, 99ᵗʰ 21.459 μs
│  30 allocations, total 118.64 KiB
│  GC time: mean 3.851 μs (19.59%), max 1.554 ms (97.34%)
│               ◔ ◑  ◕                                       *
│             ▂▁▃█▄▃▆
│  ▂▁▂▂▂▂▂▂▃▄▅████████▇█▅▄▄▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▁▁▁▁▁▁▁▂▂▂▁▂▁▂▂▁▁▂▁▂▁▁▁▂▂▁▂▂▁ ▃
└  14 μs                    10_000 samples, each 1 evaluation                   22 μs +

# with matmul too:

julia> w, b = rand(Float32, 100, 100), rand(Float32, 100); x = rand(Float32, size(w)...);

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(relu, w, x, b)), $w, $x, $b);
  152.542 μs (70 allocations: 159.26 KiB)
  141.792 μs (67 allocations: 159.16 KiB)

julia> @btime gradient((w,x,b) -> sum(relu.((w * x) .+ b)), $w, $x, $b);
  151.083 μs (37 allocations: 236.14 KiB)
  136.750 μs (37 allocations: 236.14 KiB)

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(tanh, w, x, b)), $w, $x, $b);
  185.958 μs (70 allocations: 159.26 KiB)
  145.917 μs (67 allocations: 159.16 KiB)

julia> @btime gradient((w,x,b) -> sum(tanh.((w * x) .+ b)), $w, $x, $b);  # faster, WTF?
  153.167 μs (37 allocations: 236.14 KiB)
  147.375 μs (37 allocations: 236.14 KiB)

julia> @btime gradient((w,x,b) -> sum(tanh_fast.((w * x) .+ b)), $w, $x, $b);  # why doesn't this save 57 μs
  149.791 μs (37 allocations: 236.14 KiB)
  141.042 μs (37 allocations: 236.14 KiB)

julia> @btime gradient((w,x,b) -> sum(dense_bias_act(identity, w, x, b)), $w, $x, $b);
  604.000 μs (53 allocations: 189.16 KiB)
julia> @btime gradient((w,x,b) -> sum(identity.((w * x) .+ b)), $w, $x, $b);  # Oh, this hits generic matmul
  602.583 μs (31 allocations: 227.23 KiB)

# Maybe min is misleading?

julia> @benchmark gradient((w,x,b) -> sum(dense_bias_act(tanh, w, x, b)), $w, $x, $b)
┌ Trial:
│  min 144.542 μs, median 292.833 μs, mean 270.655 μs, 99ᵗʰ 762.302 μs
│  67 allocations, total 159.16 KiB
│  GC time: mean 5.306 μs (1.96%), max 1.818 ms (80.30%)
│   ◔               * ◑ ◕
│   █                  ▄
│  ▂█▆▃▂▂▂▂▂▂▁▂▂▂▂▂▂▂▂▆█▄▃▃▃▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▁▂▂▁▂ ▂
└  140 μs                   7_329 samples, each 1 evaluation                   770 μs +

julia> @benchmark gradient((w,x,b) -> sum(tanh_fast.((w * x) .+ b)), $w, $x, $b)
┌ Trial:
│  min 140.375 μs, median 152.792 μs, mean 174.254 μs, 99ᵗʰ 427.959 μs
│  37 allocations, total 236.14 KiB
│  GC time: mean 7.073 μs (4.06%), max 1.400 ms (88.64%)
│     ◑◕    *
│     █
│  ▂▂▄█▇▃▂▂▂▂▂▂▂▁▁▂▁▂▁▂▁▁▁▁▁▁▂▂▂▁▂▁▁▁▁▁▁▁▁▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▂▁▂▁▂▁▂▂▂▂▂▂▂▁▂ ▂
└  140 μs                   10_000 samples, each 1 evaluation                  430 μs +



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






# CUDA 

julia> using CUDA, NNlib, BenchmarkTools

julia> w, b = cu(rand(Float32, 100, 100)), cu(rand(Float32, 100));

julia> @btime CUDA.@sync bias_act!(relu, $w, $b);
  min 20.508 μs, mean 67.102 μs (25 allocations, 1.73 KiB)

julia> @btime CUDA.@sync relu.($w .+ $b); 
  min 23.991 μs, mean 48.431 μs (28 allocations, 1.83 KiB)

julia> @btime CUDA.@sync bias_act!(tanh, $w, $b);
  min 20.828 μs, mean 29.528 μs (25 allocations, 1.73 KiB)

julia> @btime CUDA.@sync tanh.($w .+ $b);
  min 24.365 μs, mean 40.737 μs (28 allocations, 1.83 KiB)

julia> @btime CUDA.@sync tanh_fast.($w .+ $b);
  min 23.943 μs, mean 118.688 μs (28 allocations, 1.83 KiB)

julia> using Zygote

julia> @btime CUDA.@sync gradient((w,b) -> sum(bias_act!(relu, w, b)), $w, $b);
  min 178.961 μs, mean 330.394 μs (237 allocations, 13.10 KiB. GC mean 1.97%)

julia> @btime CUDA.@sync gradient((w,b) -> sum(relu.(w .+ b)), $w, $b);
  min 181.137 μs, mean 388.223 μs (199 allocations, 11.20 KiB. GC mean 1.89%)

julia> @btime CUDA.@sync gradient((w,b) -> sum(bias_act!(tanh, w, b)), $w, $b);
  min 184.964 μs, mean 337.266 μs (237 allocations, 13.10 KiB. GC mean 1.50%)

julia> @btime CUDA.@sync gradient((w,b) -> sum(tanh.(w .+ b)), $w, $b);
  min 187.349 μs, mean 366.614 μs (201 allocations, 11.33 KiB. GC mean 2.59%)


julia> ww, xx, bb = cu(rand(Float32, 1000, 1000)), cu(rand(Float32, 1000, 1000)), cu(rand(Float32, 1000));


julia> CUDA.@time gradient((w,x,b) -> sum(dense_bias_act(relu, w, x, b)), ww, xx, bb);
  0.103057 seconds (39.01 k CPU allocations: 1.972 MiB) (8 GPU allocations: 19.078 MiB, 0.11% memmgmt time)

julia> CUDA.@time gradient((w,x,b) -> sum(relu.((w * x) .+ b)), ww, xx, bb);
  0.105209 seconds (55.74 k CPU allocations: 2.755 MiB) (10 GPU allocations: 26.707 MiB, 0.53% memmgmt time)

julia> CUDA.@time gradient((w,x,b) -> sum(dense_bias_act(tanh, w, x, b)), ww, xx, bb);
  0.120706 seconds (39.01 k CPU allocations: 1.972 MiB) (8 GPU allocations: 19.078 MiB, 0.35% memmgmt time)

julia> CUDA.@time gradient((w,x,b) -> sum(tanh.((w * x) .+ b)), ww, xx, bb);
  0.209251 seconds (63.83 k CPU allocations: 3.131 MiB) (10 GPU allocations: 26.707 MiB, 41.74% memmgmt time)




# CPU, on cyclops

julia> w, b = rand(Float32, 100, 100), rand(Float32, 100);

julia> @btime bias_act!(relu, $w, $b);
  min 14.375 μs, mean 14.510 μs (2 allocations, 64 bytes)

julia> @btime relu.($w .+ $b);  # faster, that's odd?
  min 6.326 μs, mean 10.979 μs (2 allocations, 39.11 KiB. GC mean 14.39%)

julia> @btime bias_act!(tanh, $w, $b); 
  min 182.245 μs, mean 184.365 μs (2 allocations, 64 bytes)

julia> @btime tanh.($w .+ $b);
  min 185.692 μs, mean 188.829 μs (2 allocations, 39.11 KiB. GC mean 0.85%)

julia> @btime bias_act!(tanh_fast, $w, $b);
  min 81.591 μs, mean 82.282 μs (2 allocations, 64 bytes)

julia> @btime tanh_fast.($w .+ $b);  # saves 57 μs
  min 16.892 μs, mean 19.127 μs (2 allocations, 39.11 KiB. GC mean 8.45%)

# With broadcast 

julia> w, b = rand(Float32, 100, 100), rand(Float32, 100);

julia> @btime bias_act!(relu, $w, $b);
  min 3.247 μs, mean 3.285 μs (2 allocations, 64 bytes)

julia> @btime relu.($w .+ $b);  # faster, that's odd?
  min 5.019 μs, mean 8.953 μs (2 allocations, 39.11 KiB. GC mean 16.00%)

julia> @btime bias_act!(tanh, $w, $b);
  min 57.139 μs, mean 65.288 μs (2 allocations, 64 bytes)

julia> @btime tanh.($w .+ $b);
  min 58.902 μs, mean 61.109 μs (2 allocations, 39.11 KiB. GC mean 2.48%)

julia> @btime bias_act!(tanh_fast, $w, $b);
  min 11.861 μs, mean 11.989 μs (2 allocations, 64 bytes)

julia> @btime tanh_fast.($w .+ $b);  # saves 57 μs
  min 11.911 μs, mean 13.903 μs (2 allocations, 39.11 KiB. GC mean 10.68%)

# CUDA again

julia> using CUDA, NNlib, BenchmarkTools

julia> w, b = cu(rand(Float32, 100, 100)), cu(rand(Float32, 100));

julia> @btime CUDA.@sync bias_act!(relu, $w, $b);
  min 23.631 μs, mean 45.085 μs (25 allocations, 1.73 KiB)

julia> @btime CUDA.@sync relu.($w .+ $b);
  min 27.816 μs, mean 33.317 μs (28 allocations, 1.83 KiB)

julia> @btime CUDA.@sync bias_act!(tanh, $w, $b);
  min 24.280 μs, mean 49.152 μs (25 allocations, 1.73 KiB)

julia> @btime CUDA.@sync tanh.($w .+ $b);
  min 28.355 μs, mean 30.918 μs (28 allocations, 1.83 KiB)

julia> @btime CUDA.@sync tanh_fast.($w .+ $b);
  min 23.805 μs, mean 90.291 μs (28 allocations, 1.83 KiB)

julia> using CUDA, NNlib, BenchmarkTools

julia> w, b = cu(rand(Float32, 100, 100)), cu(rand(Float32, 100));

julia> @btime CUDA.@sync bias_act!(relu, $w, $b);
  min 20.989 μs, mean 23.078 μs (25 allocations, 1.73 KiB)

julia> @btime CUDA.@sync relu.($w .+ $b);
  min 23.684 μs, mean 28.132 μs (28 allocations, 1.83 KiB)

julia> @btime CUDA.@sync bias_act!(tanh, $w, $b);
  min 21.212 μs, mean 79.937 μs (25 allocations, 1.73 KiB)

julia> @btime CUDA.@sync tanh.($w .+ $b);
  min 23.929 μs, mean 26.815 μs (28 allocations, 1.83 KiB)

julia> @btime CUDA.@sync tanh_fast.($w .+ $b);
  min 23.713 μs, mean 53.370 μs (28 allocations, 1.83 KiB)


julia> CUDA.@time gradient((w,x,b) -> sum(dense_bias_act(relu, w, x, b)), ww, xx, bb);
  0.126533 seconds (39.02 k CPU allocations: 1.971 MiB, 7.23% gc time) (8 GPU allocations: 19.078 MiB, 1.58% memmgmt time)

julia> CUDA.@time gradient((w,x,b) -> sum(relu.((w * x) .+ b)), ww, xx, bb);
  0.153388 seconds (55.78 k CPU allocations: 2.756 MiB, 30.03% gc time) (10 GPU allocations: 26.707 MiB, 1.27% memmgmt time)

julia> CUDA.@time gradient((w,x,b) -> sum(dense_bias_act(tanh, w, x, b)), ww, xx, bb);
  0.159702 seconds (39.01 k CPU allocations: 1.971 MiB, 16.39% gc time) (8 GPU allocations: 19.078 MiB, 1.22% memmgmt time)

julia> CUDA.@time gradient((w,x,b) -> sum(tanh.((w * x) .+ b)), ww, xx, bb);
  0.135776 seconds (63.86 k CPU allocations: 3.132 MiB, 10.36% gc time) (10 GPU allocations: 26.707 MiB, 5.97% memmgmt time)

julia> w=ww; x=xx; b=bb;

julia> @btime CUDA.@sync bias_act!(relu, $w, $b);
  min 51.828 μs, mean 323.729 μs (25 allocations, 1.73 KiB)

julia> @btime CUDA.@sync relu.($w .+ $b);
  min 59.475 μs, mean 974.905 μs (28 allocations, 1.83 KiB. GC mean 13.67%)

julia> @btime CUDA.@sync bias_act!(tanh, $w, $b);
  min 58.309 μs, mean 295.973 μs (25 allocations, 1.73 KiB)

julia> @btime CUDA.@sync tanh.($w .+ $b);
  min 66.315 μs, mean 1.096 ms (28 allocations, 1.83 KiB. GC mean 10.03%)

julia> @btime CUDA.@sync tanh_fast.($w .+ $b);
  min 61.692 μs, mean 1.172 ms (28 allocations, 1.83 KiB. GC mean 9.96%)

julia> @btime CUDA.@sync bias_act!(relu, $w, $b);
  min 48.374 μs, mean 51.494 μs (25 allocations, 1.73 KiB)

julia> @btime CUDA.@sync relu.($w .+ $b);
  min 59.586 μs, mean 306.838 μs (28 allocations, 1.83 KiB. GC mean 29.80%)

julia> @btime CUDA.@sync bias_act!(tanh, $w, $b);
  min 55.015 μs, mean 56.979 μs (25 allocations, 1.73 KiB)

julia> @btime CUDA.@sync tanh.($w .+ $b);
  min 61.429 μs, mean 480.402 μs (28 allocations, 1.83 KiB. GC mean 20.96%)

julia> @btime CUDA.@sync tanh_fast.($w .+ $b);
  min 63.250 μs, mean 219.819 μs (28 allocations, 1.83 KiB. GC mean 0.68%)

# Very noisy! But some beneefit, maybe?

=#
