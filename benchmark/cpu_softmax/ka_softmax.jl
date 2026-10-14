# `softmax!` of CPU arrays as KernelAbstractions kernels: `softmax_ka!`, the one meant to be
# fast on the CPU backend, and `softmax_ka_slice!`, the kernel one would write first.
#
# The loops are those of `NNlib.softmax!` (`src/softmax.jl`), with what the CPU backend of
# KernelAbstractions 0.10 asks for. That backend compiles a kernel to SPIR-V and hands it
# to PoCL, and the annotations Julia's own compiler goes by are lost on the way:
#
#   * The maximum is taken on the bit patterns, as integers. PoCL does not see the
#     `@fastmath` of a floating-point `max`, and leaves that reduction scalar.
#   * `exp` rounds by adding and subtracting a constant. `round` becomes OpenCL's `rint`,
#     which PoCL does not vectorize on every CPU.
#   * In place there is a kernel taking the array once. `@simd ivdep` is lost too, and
#     PoCL does not vectorize a loop reading one argument and writing another when the
#     two are the same array.
#   * A work-item takes a chunk of slices and the workgroup size is 1, see `softmax_ka!`.
#
# With KernelAbstractions 0.9 the kernels are compiled by Julia, and none of this matters.

using NNlib, KernelAbstractions
using NNlib: _mul_pow2, softmax_shift, softmax_shift_inf, softmax_shape, softmax_tile

const KA = KernelAbstractions

# `NNlib._exp` with `round(x)` written as `(x + c) - c`. The results are the same, bit for bit.
@inline ka_round(x::Float32) = (x + 1.2582912f7) - 1.2582912f7
@inline ka_round(x::Float64) = (x + 6.755399441055744e15) - 6.755399441055744e15

@inline function ka_exp(x::Float32)
    x = clamp(x, -104f0, 89f0)
    N = ka_round(x * 1.442695f0)
    r = muladd(N, -0.69314575f0, x)
    r = muladd(N, -1.4286068f-6, r)
    p = evalpoly(r, (1f0, 1f0, 0.5f0, 0.16666667f0, 0.041666217f0,
                     0.008333249f0, 0.001394858f0, 0.00019924171f0))
    _mul_pow2(p, unsafe_trunc(Int32, N))
end
@inline function ka_exp(x::Float64)
    x = clamp(x, -746.0, 710.0)
    N = ka_round(x * 1.4426950408889634)
    hi = muladd(N, -0.6931471803691238, x)
    lo = N * 1.9082149292705877e-10
    r = hi - lo
    r2 = r * r
    c = r - r2 * evalpoly(r2, (0.16666666666666602, -0.0027777777777015593,
                               6.613756321437934e-5, -1.6533902205465252e-6,
                               4.1381367970572385e-8))
    _mul_pow2(1 + (hi - (lo - r * c / (2 - c))), unsafe_trunc(Int64, N))
end

# Floats as integers in the same order, and back.
@inline ka_ordint(x::Float32) = (i = reinterpret(Int32, x); i ⊻ ((i >> 31) & typemax(Int32)))
@inline ka_ordint(x::Float64) = (i = reinterpret(Int64, x); i ⊻ ((i >> 63) & typemax(Int64)))
@inline ka_ordfloat(i::Int32) = reinterpret(Float32, i ⊻ ((i >> 31) & typemax(Int32)))
@inline ka_ordfloat(i::Int64) = reinterpret(Float64, i ⊻ ((i >> 63) & typemax(Int64)))

# softmax of the slice `x[base+lo+1:base+lo+n]`, contiguous in memory. The loops run over
# `lo+1:lo+n`: with `base = 0` their bounds are those of the slice, with `lo = 0` they are
# the same for every slice, which matters to `softmax_ka_slice!`.
@inline function ka_slice!(out::AbstractArray{T}, x::AbstractArray{T}, base::Int, lo::Int, n::Int) where {T}
    mi = ka_ordint(T(-Inf))
    @inbounds @simd for i in lo+1:lo+n
        mi = max(mi, ka_ordint(x[base+i]))
    end
    m = ka_ordfloat(mi)
    s = zero(T)
    if isfinite(m)
        @inbounds @simd ivdep for i in lo+1:lo+n
            e = ka_exp(x[base+i] - m)
            out[base+i] = e
            s += e
        end
    else
        @inbounds @simd ivdep for i in lo+1:lo+n
            e = ka_exp(softmax_shift_inf(x[base+i], m))
            out[base+i] = e
            s += e
        end
    end
    r = inv(s)
    @inbounds @simd ivdep for i in lo+1:lo+n
        out[base+i] *= r
    end
    return nothing
end

# softmax along the second dimension of `nt` rows of a slice, the first of which starts at
# `x[off+1]`. `ws[wo+1:wo+nt]` receives the maxima of the rows, `ws[wo+nt+1:wo+2nt]` their sums.
@inline function ka_block!(out::AbstractArray{T}, x::AbstractArray{T}, ws::AbstractArray{T}, wo::Int,
                           off::Int, nt::Int, pre::Int, n::Int) where {T}
    finite = true
    @inbounds begin
        @simd ivdep for t in 1:nt
            ws[wo+t] = x[off+t]
        end
        for j in 1:n-1
            o = off + j * pre
            @simd ivdep for t in 1:nt
                ws[wo+t] = ifelse(x[o+t] > ws[wo+t], x[o+t], ws[wo+t])
            end
        end
        @simd for t in 1:nt
            finite &= isfinite(ws[wo+t])
        end
    end
    if finite
        ka_block!(softmax_shift, out, x, ws, wo, off, nt, pre, n)
    else
        ka_block!(softmax_shift_inf, out, x, ws, wo, off, nt, pre, n)
    end
    return nothing
end

@inline function ka_block!(shift::S, out::AbstractArray{T}, x::AbstractArray{T}, ws::AbstractArray{T},
                           wo::Int, off::Int, nt::Int, pre::Int, n::Int) where {S,T}
    so = wo + nt
    @inbounds begin
        @simd ivdep for t in 1:nt
            ws[so+t] = zero(T)
        end
        for j in 0:n-1
            o = off + j * pre
            @simd ivdep for t in 1:nt
                e = ka_exp(shift(x[o+t], ws[wo+t]))
                out[o+t] = e
                ws[so+t] += e
            end
        end
        @simd ivdep for t in 1:nt
            ws[so+t] = inv(ws[so+t])
        end
        for j in 0:n-1
            o = off + j * pre
            @simd ivdep for t in 1:nt
                out[o+t] *= ws[so+t]
            end
        end
    end
    return nothing
end

# The items `items`: slices if they are contiguous (`pre == 1`), tiles of rows if not.
@inline function ka_items!(out, x, ws, wo::Int, pre::Int, n::Int, tile::Int, items)
    if pre == 1
        for k in items
            ka_slice!(out, x, 0, (k - 1) * n, n)
        end
    else
        ntiles = cld(pre, tile)
        for item in items
            k, it = divrem(item - 1, ntiles)
            ka_block!(out, x, ws, wo, k * pre * n + it * tile, min(tile, pre - it * tile), pre, n)
        end
    end
    return nothing
end

#######################
# chunks
#######################

# A work-item takes a chunk of `chunk` items, and the workgroup size is 1. PoCL runs a
# workgroup as a loop over its work-items, and may move that loop inside the loops of the
# kernel (KernelAbstractions.jl#890): with one work-item there is nothing to move. A chunk
# is also what a workspace is for: the work-items are told apart by their index alone.
@kernel inbounds=true function softmax_chunk_kernel!(out, x, ws, pre, n, tile, nitems, chunk)
    c = @index(Global, Linear)
    i0 = (c - 1) * chunk + 1
    ka_items!(out, x, ws, (c - 1) * 2tile, pre, n, tile, i0:min(i0 + chunk - 1, nitems))
end

# In place, see above.
@kernel inbounds=true function softmax_chunk_inplace_kernel!(x, ws, pre, n, tile, nitems, chunk)
    c = @index(Global, Linear)
    i0 = (c - 1) * chunk + 1
    ka_items!(x, x, ws, (c - 1) * 2tile, pre, n, tile, i0:min(i0 + chunk - 1, nitems))
end

"""
    softmax_ka!(out, x; dims = 1, oversub = 4)

`softmax!` on the CPU backend of KernelAbstractions: `oversub` chunks of work for each
thread, each of them one work-item. Small arrays, and all of them on a single thread, are
left to `ka_items!` without a kernel, as `NNlib.softmax!` leaves them to its loops
without a task.
"""
function softmax_ka!(out::Array{T}, x::Array{T}; dims = 1, oversub::Int = 4) where {T}
    pre, n, post = softmax_shape(out, x, dims)
    nthreads = Threads.nthreads(:default)
    maxtasks = NNlib.softmax_ntasks(length(x))
    tile = pre == 1 ? 1 : softmax_tile(pre, post, maxtasks)
    nitems = cld(pre, tile) * post
    ntasks = min(maxtasks, nitems)
    if ntasks == 1
        ka_items!(out, x, Vector{T}(undef, pre == 1 ? 0 : 2tile), 0, pre, n, tile, 1:nitems)
        return out
    end
    # as many chunks as tasks when not every thread is to run: KernelAbstractions 0.9
    # gives every thread a chunk if there are enough of them
    chunk = cld(nitems, ntasks < nthreads ? ntasks : min(nitems, oversub * ntasks))
    nchunks = cld(nitems, chunk)
    ws = Matrix{T}(undef, pre == 1 ? 0 : 2tile, nchunks)
    backend = KA.CPU()
    if out === x
        softmax_chunk_inplace_kernel!(backend, 1)(x, ws, pre, n, tile, nitems, chunk; ndrange = nchunks)
    else
        softmax_chunk_kernel!(backend, 1)(out, x, ws, pre, n, tile, nitems, chunk; ndrange = nchunks)
    end
    KA.synchronize(backend)
    return out
end

#######################
# a work-item for each slice
#######################

@kernel inbounds=true function softmax_slice_kernel!(out, x, n)
    k = @index(Global, Linear)
    ka_slice!(out, x, 0, (k - 1) * n, n)
end

# Without the mask on the indices the work-items past the last slice run as well: the
# number of slices has to be a multiple of the workgroup size.
@kernel inbounds=true unsafe_indices=true function softmax_slice_unsafe_kernel!(out, x, n)
    k = @index(Global, Linear)
    ka_slice!(out, x, 0, (k - 1) * n, n)
end

# The same with loops over `1:n`, whatever the slice. These are the loops that PoCL moves
# outside the one over the work-items of a workgroup (KernelAbstractions.jl#890).
@kernel inbounds=true unsafe_indices=true function softmax_slice_unsafe_shared_kernel!(out, x, n)
    k = @index(Global, Linear)
    ka_slice!(out, x, (k - 1) * n, 0, n)
end

"""
    softmax_ka_slice!(out, x; dims = 1, workgroupsize = nothing, unsafe_indices = false,
                      shared_bounds = false)

`softmax!` as a kernel with one work-item for each slice, for contiguous slices only.
`workgroupsize = nothing` leaves the workgroup size to the backend. With `shared_bounds`
the loops of every work-item have the same bounds, `1:n`, instead of those of its slice.
"""
function softmax_ka_slice!(out::Array{T}, x::Array{T}; dims = 1, workgroupsize = nothing,
                           unsafe_indices::Bool = false, shared_bounds::Bool = false) where {T}
    pre, n, post = softmax_shape(out, x, dims)
    pre == 1 || throw(ArgumentError("the slices along dims = $dims are not contiguous"))
    shared_bounds && !unsafe_indices && throw(ArgumentError("`shared_bounds` is for `unsafe_indices` only"))
    backend = KA.CPU()
    kernel = shared_bounds ? softmax_slice_unsafe_shared_kernel! :
             unsafe_indices ? softmax_slice_unsafe_kernel! : softmax_slice_kernel!
    kernel = workgroupsize === nothing ? kernel(backend) : kernel(backend, workgroupsize)
    kernel(out, x, n; ndrange = post)
    KA.synchronize(backend)
    return out
end
