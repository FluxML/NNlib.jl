using Statistics

# Pooling is so similar, we abstract over meanpooling and maxpooling, simply replacing
# the inner loop operation and a few initialization parameters.
for name in (:max, :mean)
    @eval function $((Symbol("$(name)pool_direct!")))(
                    y::AbstractArray{T,5}, x::AbstractArray{T,5},
                    pdims::PoolDims; alpha::T = T(1), beta::T = T(0)) where {T}
        check_dims(size(x), size(y), pdims)

        width, height, depth = input_size(pdims)
        kernel_w, kernel_h, kernel_d = kernel_size(pdims)
        out_c = channels_out(pdims)
        pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(pdims)
        dil_w, dil_h, dil_d = dilation(pdims)
        stride_w, stride_h, stride_d = stride(pdims)
        out_width, out_height, out_depth = output_size(pdims)

        # We use calc_padding_regions to split outselves up into separate regions that may or
        # may not need to worry about padding:
        padded_regions, central_region = calc_padding_regions(pdims)

        # A helper function to project from output (w, h) to input (input_w, input_h)
        @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1

        # If we're doing mean pooling, we represent division by kernel size by rolling it
        # into the `alpha` multiplier.
        if $(name == :mean)
            alpha = alpha/prod(kernel_size(pdims))
        end

        # Each loop, we initialize `m` to something, set that here.
        m_init = if $(name == :max)
            typemin(T)
        elseif $(name == :mean)
            T(0)
        else
            error("Unimplemented codegen path")
        end

        # Start with the central region
        w_region, h_region, d_region = central_region
        @inbounds for batch_idx in 1:size(x)[end],
            c in 1:out_c,
            d in d_region,
            h in h_region,
            w in w_region
          
            # Initialize `m` to `0.0`, or `typemin(T)` or whatever.
            m = m_init

            for kd in 1:kernel_d,
                kh in 1:kernel_h,
                kw in 1:kernel_w

                input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
                input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
                input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w

                # This conditional will be optimized away at compile time
                if $(name == :max)
                    m = max(m, x[input_kw, input_kh, input_kd, c, batch_idx])
                elseif $(name == :mean)
                    m += x[input_kw, input_kh, input_kd, c, batch_idx]
                else
                    error("Unimplemented codegen path")
                end
            end
            y[w, h, d, c, batch_idx] = alpha*m + beta*y[w, h, d, c, batch_idx]
        end

        # Next, the padded regions
        @inbounds for (w_region, h_region, d_region) in padded_regions
            for batch_idx in 1:size(x)[end],
                c in 1:out_c,
                d in d_region,
                h in h_region,
                w in w_region

                # In these loops, we have to check that we're not reaching off the edge, we
                # do so by putting in a bunch of conditionals.  :/
                m = m_init
                for kd in 1:kernel_d
                    input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
                    if input_kd <= 0 || input_kd > depth
                        m = max(m, 0.0)
                        continue
                    end

                    for kh in 1:kernel_h
                        input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
                        if input_kh <= 0 || input_kh > height
                            m = max(m, 0.0)
                            continue
                        end

                        for kw in 1:kernel_w
                            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w
                            if input_kw <= 0 || input_kw > width
                                m = max(m, 0.0)
                                continue
                            end

                            if $(name == :max)
                                m = max(m, x[input_kw, input_kh, input_kd, c, batch_idx])
                            elseif $(name == :mean)
                                m += x[input_kw, input_kh, input_kd, c, batch_idx]
                            else
                                error("Unimplemented codegen path")
                            end
                        end
                    end
                end
                y[w, h, d, c, batch_idx] = alpha*m + beta*y[w, h, d, c, batch_idx]
            end
        end

        # Return `y`
        return y
    end

    # Same story for gradients, and although this is very similar to the forward pass,
    # it's unfortunately different enough that I think we need a separate function.  :(
    @eval function $((Symbol("∇$(name)pool_direct!")))(
                    dx::AbstractArray{T,5}, dy::AbstractArray{T,5},
                    y::AbstractArray{T,5}, x::AbstractArray{T,5},
                    pdims::PoolDims; alpha::T = T(1), beta::T = T(0)) where {T}
        check_dims(size(x), size(dy), pdims)

        width, height, depth = input_size(pdims)
        kernel_w, kernel_h, kernel_d = kernel_size(pdims)
        out_c = channels_out(pdims)
        pad_w_lo, pad_w_hi, pad_h_lo, pad_h_hi, pad_d_lo, pad_d_hi = padding(pdims)
        dil_w, dil_h, dil_d = dilation(pdims)
        stride_w, stride_h, stride_d = stride(pdims)
        out_width, out_height, out_depth = output_size(pdims)

        # We use calc_padding_regions to split outselves up into separate regions that
        # may or may not need to worry about padding:
        padded_regions, central_region = calc_padding_regions(pdims)

        # A helper function to project from output (w, h) to input (input_w, input_h)
        @inline project(idx, stride, pad) = (idx - 1)*stride - pad + 1

        # If we're doing mean pooling, we represent division by kernel size by rolling
        # it into the `alpha` multiplier.
        if $(name == :mean)
            alpha = alpha/prod(kernel_size(pdims))
        end

        # Start with the central region
        w_region, h_region, d_region = central_region
        @inbounds for batch_idx in 1:size(x)[end],
            c in 1:out_c,
            d in d_region,
            h in h_region,
            w in w_region

            # Grab the output at this index for future use
            y_idx = y[w, h, d, c, batch_idx]
            dy_idx = dy[w, h, d, c, batch_idx]
            maxpool_already_chose = false

            for kd in 1:kernel_d,
                kh in 1:kernel_h,
                kw in 1:kernel_w

                input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
                input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
                input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w

                # This conditional will be optimized away at compile time,
                # or my name isn't shengdan jingyu
                x_idxs = (input_kw, input_kh, input_kd, c, batch_idx)
                if $(name == :max)
                    # If it's equal; this is the one we chose. We only choose one per
                    # kernel window, all other elements of dx must be zero.
                    if y_idx == x[x_idxs...] && !maxpool_already_chose
                        dx[x_idxs...] = dy_idx*alpha + beta*dx[x_idxs...]
                        maxpool_already_chose = true
                    # Maxpooling does not support `beta` right now.  :(
                    #else
                    #    dx[x_idxs...] = T(0) + beta*dx[x_idxs...]
                    end
                elseif $(name == :mean)
                    # Either does meanpool :(
                    dx[x_idxs...] = dy_idx*alpha + dx[x_idxs...]
                else
                    error("Unimplemented codegen path")
                end
            end
        end

        # Next, the padded regions
        @inbounds for (w_region, h_region, d_region) in padded_regions
            for batch_idx in 1:size(x)[end],
                c in 1:out_c,
                d in d_region,
                h in h_region,
                w in w_region

                # Grab the incoming gradient at this index for future use
                y_idx = dy[w, h, d, c, batch_idx]
                dy_idx = dy[w, h, d, c, batch_idx]
                maxpool_already_chose = false

                # In these loops, we have to check that we're not reaching off the edge,
                # we do so by putting in a bunch of conditionals.  :/
                for kd in 1:kernel_d
                    input_kd = project(d, stride_d, pad_d_lo) + (kd - 1)*dil_d
                    if input_kd <= 0 || input_kd > depth
                        continue
                    end

                    for kh in 1:kernel_h
                        input_kh = project(h, stride_h, pad_h_lo) + (kh - 1)*dil_h
                        if input_kh <= 0 || input_kh > height
                            continue
                        end

                        for kw in 1:kernel_w
                            input_kw = project(w, stride_w, pad_w_lo) + (kw - 1)*dil_w
                            if input_kw <= 0 || input_kw > width
                                continue
                            end

                            # Same as above
                            x_idxs = (input_kw, input_kh, input_kd, c, batch_idx)
                            if $(name == :max)
                                if y_idx == x[x_idxs...] && !maxpool_already_chose
                                    dx[x_idxs...] = dy_idx*alpha + beta*dx[x_idxs...]
                                    maxpool_already_chose = true
                                #else
                                #    dx[x_idxs...] = T(0) + beta*dx[x_idxs...]
                                end
                            elseif $(name == :mean)
                                dx[x_idxs...] += dy_idx*alpha + beta*dx[x_idxs...]
                            else
                                error("Unimplemented codegen path")
                            end
                        end
                    end
                end
            end
        end

        # Return `dx`
        return dx
    end
end
