#
# Upsampling
#

# GPU based bilinear upsampling including its gradient
#
# Based on the Caffe2 implementation at:
# The code is a translation from the following files:
# - https://github.com/pytorch/pytorch/blob/v1.8.0-rc1/caffe2/operators/upsample_op.cu
# - https://github.com/pytorch/pytorch/blob/v1.8.0-rc1/caffe2/core/common_gpu.h
#
# Copyright (c) 2016-2021 Facebook Inc.
# Copyright (c) 2015 Google Inc.
# Copyright (c) 2015 Yangqing Jia
# Copyright 2019-2020 Kakao Brain
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of
#    conditions and the following disclaimer in the documentation and/or other materials
#    provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America and
#    IDIAP Research Institute nor the names of its contributors may be used to endorse or
#    promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Forward and backward pass have been tested to produce the same output
# as pytorch with align_corners=True - it works modulo bit noise.

@inline function compute_source_index(ratio::T, dst_index, align_corners) where T
    if align_corners
        return ratio*dst_index
    else
        src_idx = ratio * (dst_index + T(0.5)) - T(0.5)
        return max(zero(T), src_idx)
    end
end

function NNlib.upsample_linear_kernel!(y::CuArray{T,N}, x::CuArray{T,N}; align_corners=true) where {T,N}
    out_size = prod(size(y)[1:N-2])

    if align_corners
        ratios = ntuple(i -> T((size(x,i)-1) / (size(y,i)-1)), N-2)
    else
        ratios = ntuple(i -> T(size(x,i) / size(y,i)), N-2)
    end

    kernel = @cuda launch=false upsample_linear_cuda_kernel!(out_size, ratios..., x, y, align_corners)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = Base.min(out_size, config.threads)
    blocks = cld(out_size, threads)
    kernel(out_size, ratios..., x, y, align_corners; threads=threads, blocks=blocks)
    return y
end

function NNlib.∇upsample_linear_kernel!(dx::CuArray{T,N}, Δ::CuArray{T,N}; align_corners=true) where {T,N}
    in_size = prod(size(Δ)[1:N-2])

    if align_corners
        ratios = ntuple(i -> T((size(dx,i)-1) / (size(Δ,i)-1)), N-2)  # reversed compared to forward pass
    else
        ratios = ntuple(i -> T(size(dx,i) / size(Δ,i)), N-2)
    end

    kernel = @cuda launch=false ∇upsample_linear_cuda_kernel!(in_size, ratios..., Δ, dx, align_corners)
    config = launch_configuration(kernel.fun; max_threads=256)
    threads = Base.min(in_size, config.threads)
    blocks = cld(in_size, threads)
    kernel(in_size, ratios..., Δ, dx, align_corners; threads=threads, blocks=blocks)
    return dx
end


###########
# linear
###########
function upsample_linear_cuda_kernel!(n_elem, rwidth, x::CuDeviceArray{<:Any, 3}, y::CuDeviceArray{<:Any, 3}, align_corners)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x

    if index < n_elem
        in_w, channels, batchsize = size(x)
        out_w, _, _ = size(y)

        ow = index % out_w

        # real_index = rwidth*ow
        real_index = compute_source_index(rwidth, ow, align_corners)
        iw0 = Base.floor(Int, real_index)
        offset = (iw0 < in_w-1) ? 1 : 0
        iw1 = iw0 + offset + 1
        w1lambda = real_index - iw0
        w0lambda = 1 - w1lambda
        iw0 += 1

        @inbounds for n in 1:batchsize
            for c in 1:channels
                val = (w0lambda * x[iw0, c, n]  + # w0 * i00
                       w1lambda * x[iw1, c, n])   # w1 * i01
                y[ow+1, c, n] = val
            end
        end
    end
    return nothing
end

# Δ is the gradient backpropagated from downstream layers
function ∇upsample_linear_cuda_kernel!(n_elem, rwidth, Δ::CuDeviceArray{<:Any, 3}, dx::CuDeviceArray{<:Any, 3}, align_corners)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x

    if index < n_elem
        in_width, channels, batchsize = size(Δ)
        out_width, _, _ = size(dx)

        iw = index % in_width

        # real_index_w = rwidth * iw
        real_index_w = compute_source_index(rwidth, iw, align_corners)
        ow0 = Base.floor(Int, real_index_w)
        offset = (ow0 < out_width - 1) ? 1 : 0
        ow1 = ow0 + offset + 1
        w1lambda = real_index_w - ow0
        w0lambda = 1 - w1lambda
        ow0 += 1

        @inbounds for n in 1:batchsize
            for c in 1:channels
                val = Δ[iw+1, c, n]
                CUDA.@atomic dx[ow0, c, n] += w0lambda * val
                CUDA.@atomic dx[ow1, c, n] += w1lambda * val
            end
        end
    end # if
    return nothing
end


###########
# bilinear
###########
function upsample_linear_cuda_kernel!(n_elem, rwidth, rheight, x::CuDeviceArray{<:Any, 4}, y::CuDeviceArray{<:Any, 4}, align_corners)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x

    if index < n_elem
        in_w, in_h, channels, batchsize = size(x)
        out_w, out_h, _, _ = size(y)

        ow = index % out_w
        oh = index ÷ out_w

        # real_index = rheight*oh
        real_index = compute_source_index(rheight, oh, align_corners)
        ih0 = Base.floor(Int, real_index)
        offset = (ih0 < in_h-1) ? 1 : 0
        ih1 = ih0 + offset + 1
        h1lambda = real_index - ih0
        h0lambda = 1 - h1lambda
        ih0 += 1

        # real_index = rwidth*ow
        real_index = compute_source_index(rwidth, ow, align_corners)
        iw0 = Base.floor(Int, real_index)
        offset = (iw0 < in_w-1) ? 1 : 0
        iw1 = iw0 + offset + 1
        w1lambda = real_index - iw0
        w0lambda = 1 - w1lambda
        iw0 += 1

        @inbounds for n in 1:batchsize
            for c in 1:channels
                val = h0lambda * (w0lambda * x[iw0, ih0, c, n]  + # h0 * w0 * i00
                                  w1lambda * x[iw1, ih0, c, n]) + # h0 * w1 * i01
                      h1lambda * (w0lambda * x[iw0, ih1, c, n]  + # h1 * w0 * i10
                                  w1lambda * x[iw1, ih1, c, n])   # h1 * w1 * i11
                y[ow+1, oh+1, c, n] = val
            end
        end
    end
    return nothing
end

# Δ is the gradient backpropagated from downstream layers
function ∇upsample_linear_cuda_kernel!(n_elem, rwidth, rheight, Δ::CuDeviceArray{<:Any, 4}, dx::CuDeviceArray{<:Any, 4}, align_corners)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x

    if index < n_elem
        in_width, in_height, channels, batchsize = size(Δ)
        out_width, out_height, _, _ = size(dx)

        iw = index % in_width
        ih = index ÷ in_width

        # Compute Y axis lambdas
        # real_index_h = rheight*ih
        real_index_h = compute_source_index(rheight, ih, align_corners)
        oh0 = Base.floor(Int, real_index_h)
        offset = (oh0 < out_height-1) ? 1 : 0
        oh1 = oh0 + offset + 1
        h1lambda = real_index_h - oh0
        h0lambda = 1 - h1lambda
        oh0 += 1

        # # Compute X axis lambdas
        # real_index_w = rwidth * iw
        real_index_w = compute_source_index(rwidth, iw, align_corners)
        ow0 = Base.floor(Int, real_index_w)
        offset = (ow0 < out_width - 1) ? 1 : 0
        ow1 = ow0 + offset + 1
        w1lambda = real_index_w - ow0
        w0lambda = 1 - w1lambda
        ow0 += 1

        @inbounds for n in 1:batchsize
            for c in 1:channels
                val = Δ[iw+1, ih+1, c, n]
                CUDA.@atomic dx[ow0, oh0, c, n] += h0lambda * w0lambda * val
                CUDA.@atomic dx[ow1, oh0, c, n] += h0lambda * w1lambda * val
                CUDA.@atomic dx[ow0, oh1, c, n] += h1lambda * w0lambda * val
                CUDA.@atomic dx[ow1, oh1, c, n] += h1lambda * w1lambda * val
            end
        end
    end # if
    return nothing
end


###########
# trilinear
###########
function upsample_linear_cuda_kernel!(n_elem, rwidth, rheight, rdepth, x::CuDeviceArray{<:Any, 5}, y::CuDeviceArray{<:Any, 5}, align_corners)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x

    if index < n_elem
        in_w, in_h, in_d, channels, batchsize = size(x)
        out_w, out_h, out_d, _, _ = size(y)

        ow = (index % (out_w * out_h)) % out_w
        oh = (index % (out_w * out_h)) ÷ out_w
        od = index ÷ (out_w * out_h)

        # real_index = rwidth*ow
        real_index = compute_source_index(rwidth, ow, align_corners)
        iw0 = Base.floor(Int, real_index)
        offset = (iw0 < in_w-1) ? 1 : 0
        iw1 = iw0 + offset + 1
        w1lambda = real_index - iw0
        w0lambda = 1 - w1lambda
        iw0 += 1

        # real_index = rheight*oh
        real_index = compute_source_index(rheight, oh, align_corners)
        ih0 = Base.floor(Int, real_index)
        offset = (ih0 < in_h-1) ? 1 : 0
        ih1 = ih0 + offset + 1
        h1lambda = real_index - ih0
        h0lambda = 1 - h1lambda
        ih0 += 1

        # real_index = rdepth*od
        real_index = compute_source_index(rdepth, od, align_corners)
        id0 = Base.floor(Int, real_index)
        offset = (id0 < in_d-1) ? 1 : 0
        id1 = id0 + offset + 1
        d1lambda = real_index - id0
        d0lambda = 1 - d1lambda
        id0 += 1

        @inbounds for n in 1:batchsize
            for c in 1:channels
                val = d0lambda *
                         (h0lambda *
                           (w0lambda * x[iw0, ih0, id0, c, n]  +
                            w1lambda * x[iw1, ih0, id0, c, n]) +
                          h1lambda *
                           (w0lambda * x[iw0, ih1, id0, c, n]   +
                            w1lambda * x[iw1, ih1, id0, c, n])) +
                      d1lambda *
                         (h0lambda *
                           (w0lambda * x[iw0, ih0, id1, c, n]  +
                            w1lambda * x[iw1, ih0, id1, c, n]) +
                          h1lambda *
                           (w0lambda * x[iw0, ih1, id1, c, n] +
                            w1lambda * x[iw1, ih1, id1, c, n]))

                y[ow+1, oh+1, od+1, c, n] = val
            end
        end
    end
    return nothing
end

# Δ is the gradient backpropagated from downstream layers
function ∇upsample_linear_cuda_kernel!(n_elem, rwidth, rheight, rdepth, Δ::CuDeviceArray{<:Any, 5}, dx::CuDeviceArray{<:Any, 5}, align_corners)
    index = (threadIdx().x - 1) + (blockIdx().x - 1) * blockDim().x

    if index < n_elem
        in_width, in_height, in_depth, channels, batchsize = size(Δ)
        out_width, out_height, out_depth, _, _ = size(dx)

        iw = (index % (in_height * in_width)) % in_width
        ih = (index % (in_height * in_width)) ÷ in_width
        id = index ÷  (in_height * in_width)

        real_index_w = compute_source_index(rwidth, iw, align_corners)
        ow0 = Base.floor(Int, real_index_w)
        offset = (ow0 < out_width - 1) ? 1 : 0
        ow1 = ow0 + offset + 1
        w1lambda = real_index_w - ow0
        w0lambda = 1 - w1lambda
        ow0 += 1

        real_index_h = compute_source_index(rheight, ih, align_corners)
        oh0 = Base.floor(Int, real_index_h)
        offset = (oh0 < out_height-1) ? 1 : 0
        oh1 = oh0 + offset + 1
        h1lambda = real_index_h - oh0
        h0lambda = 1 - h1lambda
        oh0 += 1

        real_index_d = compute_source_index(rdepth, id, align_corners)
        od0 = Base.floor(Int, real_index_d)
        offset = (od0 < out_depth-1) ? 1 : 0
        od1 = od0 + offset + 1
        d1lambda = real_index_d - od0
        d0lambda = 1 - d1lambda
        od0 += 1

        @inbounds for n in 1:batchsize
            for c in 1:channels
                val = Δ[iw+1, ih+1, id+1, c, n]
                CUDA.@atomic dx[ow0, oh0, od0, c, n] += w0lambda * h0lambda * d0lambda * val
                CUDA.@atomic dx[ow1, oh0, od0, c, n] += w1lambda * h0lambda * d0lambda * val
                CUDA.@atomic dx[ow0, oh1, od0, c, n] += w0lambda * h1lambda * d0lambda * val
                CUDA.@atomic dx[ow1, oh1, od0, c, n] += w1lambda * h1lambda * d0lambda * val

                CUDA.@atomic dx[ow0, oh0, od1, c, n] += w0lambda * h0lambda * d1lambda * val
                CUDA.@atomic dx[ow1, oh0, od1, c, n] += w1lambda * h0lambda * d1lambda * val
                CUDA.@atomic dx[ow0, oh1, od1, c, n] += w0lambda * h1lambda * d1lambda * val
                CUDA.@atomic dx[ow1, oh1, od1, c, n] += w1lambda * h1lambda * d1lambda * val
            end
        end
    end # if
    return nothing
end
