# Pooling is so similar, we abstract over meanpooling and maxpooling, simply replacing
# the inner loop operation and a few initialization parameters.
for name in (:max, :mean, :lpnorm)
    @eval function $((Symbol("$(name)pool_direct!")))(
                    y::AbstractArray{<:Any, 5}, x::AbstractArray{<:Any, 5},
                    pdims::PoolDims; alpha=1, beta=0, kwargs...) 
        $((Symbol("$(name)pool_direct!")))(
            y, x, pdims,
            Val(kernel_size(pdims)), Val(channels_out(pdims)),
            Val(padding(pdims)), Val(dilation(pdims)), Val(stride(pdims));
            alpha, beta, kwargs...)
        return y
    end

    @eval function $((Symbol("$(name)pool_direct!")))(
        y::AbstractArray{T,5}, x::AbstractArray{<:Any,5},
        pdims::PoolDims,
        # kernel size, channels out, padding, dilation, stride
        ::Val{K}, ::Val{C}, ::Val{P}, ::Val{D}, ::Val{S};
        alpha=1, beta=0, count_include_pad::Bool=true, kwargs...
    ) where {T, K, C, P, D, S}
        @assert iszero(beta) "beta not supported yet"
        check_dims(size(x), size(y), pdims)

        width, height, depth = input_size(pdims)
        kernel_w, kernel_h, kernel_d = K
        pad_w_lo, _, pad_h_lo, _, pad_d_lo, _ = P
        dil_w, dil_h, dil_d = D
        stride_w, stride_h, stride_d = S

        # We use calc_padding_regions to split outselves up into separate regions that may or
        # may not need to worry about padding:
        padded_regions, central_region = calc_padding_regions(pdims)

        # A helper function to project from output (w, h) to input (input_w, input_h)
        @inline project(idx, stride, pad) = (idx - 1) * stride - pad + 1

        # If we're doing mean pooling, we represent division by kernel size by rolling it
        # into the `alpha` multiplier. 
        # The type might change here, that's why we prepend the underscore 
        # (does it make a difference, though?)
        _alpha = if $(name == :mean)
            T(alpha / prod(K))
        else
            T(alpha)
        end
        # _beta = T(beta)

        # A quick note on the array element types `T` and `R`:
        # Ideally, `T == R`, but in some edge-cases, this might not be the case 
        # (e.g. with `ReverseDiff.TrackedArray`, see issue #484).
        # If the types differ, we will initialize variables (like `_alpha` above) with the 
        # target eltype `T`.

        p = if $(name != :lpnorm) 0 else
            !haskey(kwargs, :p) && error("lpnormpool needs keyword argument `p`")
            kwargs[:p]
        end

        # Each loop, we initialize `m` to something, set that here.
        m_init = if $(name == :max)
            T <: AbstractFloat ? nextfloat(typemin(T)) : typemin(T)
        elseif $(name == :mean) || $(name == :lpnorm)
            T(0)
        else
            error("Unimplemented codegen path")
        end

        # Start with the central region
        w_region, h_region, d_region = central_region

        @inbounds for batch_idx in 1:size(x, 5), c in 1:C
            for d in d_region
            pd = project(d, stride_d, pad_d_lo)
            for h in h_region
            ph = project(h, stride_h, pad_h_lo)
            for w in w_region
            pw = project(w, stride_w, pad_w_lo)
            m = m_init

            for kd in 1:kernel_d,
                kh in 1:kernel_h,
                kw in 1:kernel_w

                input_kd = pd + (kd - 1) * dil_d
                input_kh = ph + (kh - 1) * dil_h
                input_kw = pw + (kw - 1) * dil_w

                # This conditional will be optimized away at compile time
                if $(name == :max)
                    xv = x[input_kw, input_kh, input_kd, c, batch_idx]
                    if xv > m
                        m = xv
                    end
                elseif $(name == :mean)
                    m += x[input_kw, input_kh, input_kd, c, batch_idx]
                elseif $(name == :lpnorm)
                    # y = (∑ᵢ |xᵢ|^p)^(1 / p), here to calculate ∑ᵢ |xᵢ|^p
                    m += abs(x[input_kw, input_kh, input_kd, c, batch_idx])^p
                else
                    error("Unimplemented codegen path")
                end
            end

            # for lpnormpool, y = (∑ᵢ |xᵢ|^p)^(1 / p)
            m = $(name == :lpnorm) ? m^(T(1) / p) : m

            y[w, h, d, c, batch_idx] = _alpha * m # + _beta * y[w, h, d, c, batch_idx]
            end
            end
            end
        end

        # Next, the padded regions
        @inbounds for (w_region, h_region, d_region) in padded_regions
            for batch_idx in 1:size(x, 5), c in 1:C
                for d in d_region
                pd = project(d, stride_d, pad_d_lo)
                for h in h_region
                ph = project(h, stride_h, pad_h_lo)
                for w in w_region
                pw = project(w, stride_w, pad_w_lo)
                m = m_init
                # For mean pooling with `count_include_pad=false` we divide by the number
                # of in-bounds (non-padding) elements actually visited in this window.
                kernel_count = 0

                for kd in 1:kernel_d
                    input_kd = pd + (kd - 1) * dil_d
                    if input_kd <= 0 || input_kd > depth
                        continue
                    end

                    for kh in 1:kernel_h
                        input_kh = ph + (kh - 1) * dil_h
                        if input_kh <= 0 || input_kh > height
                            continue
                        end

                        for kw in 1:kernel_w
                            input_kw = pw + (kw - 1) * dil_w
                            if input_kw <= 0 || input_kw > width
                                continue
                            end

                            if $(name == :max)
                                xv = x[input_kw, input_kh, input_kd, c, batch_idx]
                                if xv > m
                                    m = xv
                                end
                            elseif $(name == :mean)
                                m += x[input_kw, input_kh, input_kd, c, batch_idx]
                                kernel_count += 1
                            elseif $(name == :lpnorm)
                                m += abs(x[input_kw, input_kh, input_kd, c, batch_idx])^p
                            else
                                error("Unimplemented codegen path")
                            end
                        end
                    end
                end
                if $(name == :mean) && !count_include_pad
                    # Divide by the number of non-padding elements in the window.
                    win_alpha = kernel_count == 0 ? T(0) : T(alpha) / T(kernel_count)
                    y[w, h, d, c, batch_idx] = win_alpha * m
                else
                    $(name == :lpnorm) && (m = m^(T(1) / p))
                    y[w, h, d, c, batch_idx] = _alpha * m # + _beta * y[w, h, d, c, batch_idx]
                end
                end
                end
                end
            end
        end

        return y
    end

    @eval function $((Symbol("∇$(name)pool_direct!")))(
                    dx::AbstractArray{<:Any,5}, dy::AbstractArray{<:Any,5},
                    y::AbstractArray{<:Any,5}, x::AbstractArray{<:Any,5},
                    pdims::PoolDims; kwargs...)
        $((Symbol("∇$(name)pool_direct!")))(
            dx, dy, y, x, pdims, Val(kernel_size(pdims)); kwargs...)
        return dx
    end

    # Same story for gradients, and although this is very similar to the forward pass,
    # it's unfortunately different enough that I think we need a separate function.  :(
    @eval function $((Symbol("∇$(name)pool_direct!")))(
                    dx::AbstractArray{T,5}, dy::AbstractArray{<:Any,5},
                    y::AbstractArray{<:Any,5}, x::AbstractArray{<:Any,5},
                    pdims::PoolDims, ::Val{K}; # == kernel_size(pdims)
                    alpha=1, beta=0, count_include_pad::Bool=true, kwargs...) where {T, K}
        check_dims(size(x), size(dy), pdims)

        width, height, depth = input_size(pdims)
        kernel_w, kernel_h, kernel_d = K
        out_c = channels_out(pdims)
        pad_w_lo, _, pad_h_lo, _, pad_d_lo, _ = padding(pdims)
        dil_w, dil_h, dil_d = dilation(pdims)
        stride_w, stride_h, stride_d = stride(pdims)

        # Concerning array eltypes `DX, DY, X, Y`, we want handle them like above, i.e.,
        # initialize everything with the left-hand-side type (target type).
        # Of course, ideally the types are all the same anyways.

        # We use calc_padding_regions to split outselves up into separate regions that
        # may or may not need to worry about padding:
        padded_regions, central_region = calc_padding_regions(pdims)

        # A helper function to project from output (w, h) to input (input_w, input_h)
        @inline project(idx, stride, pad) = (idx - 1) * stride - pad + 1

        # If we're doing mean pooling, we represent division by kernel size by rolling
        # it into the `_alpha` multiplier.
        _alpha = if $(name == :mean)
            T(alpha / prod(K))
        else
            T(alpha)
        end

        p = if $(name != :lpnorm) 0 else
            !haskey(kwargs, :p) && error("lpnormpool must pass p")
            kwargs[:p]
        end

        # For max pooling we find the element that gets a window's gradient by redoing
        # the forward pass' scan of that window, which starts from this value (`y` has
        # the eltype the forward pass was targeting). We don't look for the element
        # matching `y` instead: `y` may not be bit-identical to it (e.g. if it has a
        # different eltype than `x`), and with a tolerance an element that merely comes
        # close to the maximum can take the gradient (issue #771).
        m_init = if $(name == :max)
            eltype(y) <: AbstractFloat ? nextfloat(typemin(eltype(y))) : typemin(eltype(y))
        else
            nothing
        end

        # Start with the central region
        w_region, h_region, d_region = central_region
        @inbounds for batch_idx in 1:size(x, 5), c in 1:out_c
            for d in d_region
            pd = project(d, stride_d, pad_d_lo)
            for h in h_region
            ph = project(h, stride_h, pad_h_lo)
            for w in w_region
            pw = project(w, stride_w, pad_w_lo)

            # Grab the output at this index for future use
            y_idx = y[w, h, d, c, batch_idx]
            dy_idx = dy[w, h, d, c, batch_idx]
            # For max pooling: the largest element of the window so far, and which one
            # it is, numbering them in scan order (zero as long as there is none).
            m = m_init
            kmax = 0

            for kd in 1:kernel_d,
                kh in 1:kernel_h,
                kw in 1:kernel_w

                input_kd = pd + (kd - 1) * dil_d
                input_kh = ph + (kh - 1) * dil_h
                input_kw = pw + (kw - 1) * dil_w

                # This conditional will be optimized away at compile time,
                # or my name isn't shengdan jingyu
                # x_idxs = (input_kw, input_kh, input_kd, c, batch_idx)
                if $(name == :max)
                    # Same comparison as in the forward pass, so that among equal
                    # elements we pick the first one too. Which element is the largest
                    # is hard to predict, so we avoid branching on it: with `if`, most
                    # of the time spent here goes into mispredicted branches. For the
                    # same reason we keep track of a single number, not of three indices.
                    xv = x[input_kw, input_kh, input_kd, c, batch_idx]
                    is_max = xv > m
                    m = ifelse(is_max, xv, m)
                    kmax = ifelse(is_max, kw + kernel_w * ((kh - 1) + kernel_h * (kd - 1)), kmax)
                elseif $(name == :mean)
                    # Either does meanpool :(
                    dx[input_kw, input_kh, input_kd, c, batch_idx] += dy_idx * _alpha
                elseif $(name == :lpnorm)
                    # y = (∑ᵢ |xᵢ|^p)^(1 / p), ∂y/∂xᵢ = |xᵢ|^(p-1) × y^(1-p) × sign(xᵢ)
                    xv = x[input_kw, input_kh, input_kd, c, batch_idx]
                    grad = abs(xv)^(p-1) * y_idx^(1-p) * sign(xv)
                    dx[input_kw, input_kh, input_kd, c, batch_idx] += dy_idx * grad
                else
                    error("Unimplemented codegen path")
                end
            end

            # We only choose one element per kernel window, all other elements of dx
            # must be zero. Maxpooling does not support `beta` right now.  :(
            if $(name == :max) && kmax > 0
                # Back from the element's number to its position in the kernel
                max_kd, max_khw = divrem(kmax - 1, kernel_w * kernel_h)
                max_kh, max_kw = divrem(max_khw, kernel_w)
                dx[pw + max_kw * dil_w, ph + max_kh * dil_h, pd + max_kd * dil_d, c, batch_idx] += dy_idx * _alpha #+ _beta * dx[x_idxs...]
            end
            end
            end
            end
        end

        # Next, the padded regions
        @inbounds for (w_region, h_region, d_region) in padded_regions
            for batch_idx in 1:size(x, 5), c in 1:out_c
                for d in d_region
                pd = project(d, stride_d, pad_d_lo)
                for h in h_region
                ph = project(h, stride_h, pad_h_lo)
                for w in w_region
                pw = project(w, stride_w, pad_w_lo)

                # Grab the incoming gradient at this index for future use
                y_idx = y[w, h, d, c, batch_idx]
                dy_idx = dy[w, h, d, c, batch_idx]
                m = m_init
                kmax = 0

                # For mean pooling with `count_include_pad=false` the divisor is the
                # number of non-padding elements in this window, so we count them up front
                # to get the per-window scaling factor.
                win_alpha = _alpha
                if $(name == :mean) && !count_include_pad
                    kernel_count = 0
                    for kd in 1:kernel_d
                        ikd = pd + (kd - 1) * dil_d
                        (ikd <= 0 || ikd > depth) && continue
                        for kh in 1:kernel_h
                            ikh = ph + (kh - 1) * dil_h
                            (ikh <= 0 || ikh > height) && continue
                            for kw in 1:kernel_w
                                ikw = pw + (kw - 1) * dil_w
                                (ikw <= 0 || ikw > width) && continue
                                kernel_count += 1
                            end
                        end
                    end
                    win_alpha = kernel_count == 0 ? T(0) : T(alpha) / T(kernel_count)
                end

                # In these loops, we have to check that we're not reaching off the edge,
                # we do so by putting in a bunch of conditionals.  :/
                for kd in 1:kernel_d
                    input_kd = pd + (kd - 1) * dil_d
                    if input_kd <= 0 || input_kd > depth
                        continue
                    end

                    for kh in 1:kernel_h
                        input_kh = ph + (kh - 1) * dil_h
                        if input_kh <= 0 || input_kh > height
                            continue
                        end

                        for kw in 1:kernel_w
                            input_kw = pw + (kw - 1) * dil_w
                            if input_kw <= 0 || input_kw > width
                                continue
                            end

                            # Same as above
                            # x_idxs = (input_kw, input_kh, input_kd, c, batch_idx)
                            if $(name == :max)
                                xv = x[input_kw, input_kh, input_kd, c, batch_idx]
                                is_max = xv > m
                                m = ifelse(is_max, xv, m)
                                kmax = ifelse(is_max, kw + kernel_w * ((kh - 1) + kernel_h * (kd - 1)), kmax)
                            elseif $(name == :mean)
                                dx[input_kw, input_kh, input_kd, c, batch_idx] += dy_idx * win_alpha #+ _beta * dx[x_idxs...]
                            elseif $(name == :lpnorm)
                                xv = x[input_kw, input_kh, input_kd, c, batch_idx]
                                grad = abs(xv)^(p-1) * y_idx^(1-p) * sign(xv)
                                dx[input_kw, input_kh, input_kd, c, batch_idx] += dy_idx * grad
                            else
                                error("Unimplemented codegen path")
                            end
                        end
                    end
                end

                if $(name == :max) && kmax > 0
                    max_kd, max_khw = divrem(kmax - 1, kernel_w * kernel_h)
                    max_kh, max_kw = divrem(max_khw, kernel_w)
                    dx[pw + max_kw * dil_w, ph + max_kh * dil_h, pd + max_kd * dil_d, c, batch_idx] += dy_idx * _alpha #+ _beta * dx[x_idxs...]
                end
            end
            end
            end
            end
        end

        return dx
    end
end
