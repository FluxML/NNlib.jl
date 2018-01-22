
## Convolutions on CPU. Almost all the code is borrowed from Knet.jl
## For GPU versions see CUDNN.jl


## helper macros & functions

macro nnlib_call(fun, x...)       # error if nnlib_call missing, nothing if run
    if libnnlib != ""
        fx = Expr(:call, :ccall, ("$fun", libnnlib), :Void, x...)
        err = gensym()
        esc(:($err=$fx; $err))
    else
        Expr(:call,:error,"Cannot find nnlib, please rerun Pkg.build(\"NNlib\").")
    end
end


function cdims(w,x; padding=0, stride=1, o...)
    N = ndims(x)
    ntuple(N) do i
        if i < N-1
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            1 + div(size(x,i) - size(w,i) + 2*pi, si)
        elseif i == N-1
            size(w,N)
        else # i == N
            size(x,N)
        end
    end
end


function pdims(x; window=2, padding=0, stride=window, o...)
    N = ndims(x)
    ntuple(N) do i
        if i < N-1
            wi = (if isa(window,Number); window; else window[i]; end)
            pi = (if isa(padding,Number); padding; else padding[i]; end)
            si = (if isa(stride,Number); stride; else stride[i]; end)
            1 + div(size(x,i) + 2*pi - wi, si)
        else
            size(x,i)
        end
    end
end


# convert padding etc. size to an Int array of the right dimension
function psize(p, x)
    nd = ndims(x)-2
    if isa(p,Number)
        fill(Int(p),nd)
    elseif length(p)==nd
        collect(Int,p)
    else
        throw(DimensionMismatch("psize: $p $nd"))
    end
end


## convolution

function conv2d{T}(x::Array{T,4}, w::Array{T,4};
                  padding=0, stride=1, upscale=1, mode=0, alpha=1,
                  o...) # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
    if upscale != 1; throw(ArgumentError("CPU conv2d only supports upscale=1.")); end
    if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    if Cx!=C1; throw(DimensionMismatch()); end
    Wy,Hy,Cy,Ny = cdims(w,x;padding=padding,stride=stride)
    # @assert Cy==C2 && Ny==Nx
    y = similar(x, (Wy,Hy,Cy,Ny))
    x2dims = im2col_dims(w,y)
    x2 = similar(x, x2dims)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    M,N,K,Y = Wy*Hy,Cy,Ww*Hw*Cx,Wy*Hy*Cy
    alpha,beta,yidx = T(alpha),T(0),1
    @inbounds for n in 1:Nx
        im2col!(w, x, x2, n, p1, p2, s1, s2, mode)
        gemm!('N','N',M,N,K,alpha,pointer(x2),pointer(w),beta,pointer(y,yidx))
        yidx += Y
    end
    return y
end

function conv2d_grad_w{T}(x::Array{T,4}, w::Array{T,4}, dy::Array{T,4};
                   padding=0, stride=1, upscale=1, mode=0, alpha=1,
                   o...) # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
    # dw = x'*dy
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    Wy,Hy,Cy,Ny = size(dy)
    # if upscale != 1; throw(ArgumentError("CPU conv2d only supports upscale=1.")); end
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    # @assert Cx==C1 && Cy==C2 && Ny==Nx
    dw = zeros(w)
    x2dims = im2col_dims(w,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Cy,Ww*Hw*Cx,Cy,Wy*Hy
    alpha,beta = T(alpha),T(1)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    dyi = 1
    @inbounds for n in 1:Nx
        im2col!(w, x, x2, n, p1, p2, s1, s2, mode)
        gemm!('T','N',M,N,K,alpha,pointer(x2),pointer(dy,dyi),beta,pointer(dw))
        dyi += Y
    end
    return dw
end

function conv2d_grad_x{T}(x::Array{T,4}, w::Array{T,4}, dy::Array{T,4};
                   padding=0, stride=1, upscale=1, mode=0, alpha=1,
                   o...) # Ignoring handle, algo, workSpace, workSpaceSizeInBytes
    # dx = dy*w'
    Wx,Hx,Cx,Nx = size(x)
    Ww,Hw,C1,C2 = size(w)
    Wy,Hy,Cy,Ny = size(dy)
    # if upscale != 1; throw(ArgumentError("CPU conv2d only supports upscale=1.")); end
    # if mode != 0 && mode != 1; throw(ArgumentError("conv2d only supports mode=0 or 1.")); end
    @assert Cx==C1 && Cy==C2 && Ny==Nx
    dx = similar(x)
    x2dims = im2col_dims(w,dy)
    x2 = similar(x, x2dims)
    # op(A) is an m-by-k matrix, op(B) is a k-by-n matrix, C is an m-by-n matrix.
    Y,M,N,K = Wy*Hy*Cy,Wy*Hy,Ww*Hw*Cx,Cy
    alpha,beta = T(alpha),T(0)
    (p1,p2) = psize(padding,x)
    (s1,s2) = psize(stride,x)
    dyi = 1
    @inbounds for n in 1:Nx
        gemm!('N','T',M,N,K,alpha,pointer(dy,dyi),pointer(w),beta,pointer(x2))
        col2im!(w,dx,x2,n,p1,p2,s1,s2,mode)
        dyi += Y
    end
    return dx
end

im2col_dims(w,y)=(size(y,1)*size(y,2), size(w,1)*size(w,2)*size(w,3))



# Functions from conv.cpp:

for (T,S) in ((Float32,32), (Float64,64)); @eval begin

    function im2col!(w::Array{$T,4}, x::Array{$T,4}, x2::Array{$T,2},
                     n::Int, p1::Int, p2::Int, s1::Int, s2::Int, mode::Int)
        Wx,Hx,Cx,Nx = size(x)
        Ww,Hw,C1,C2 = size(w)
        xn = pointer(x, Wx*Hx*Cx*(n-1)+1)
        @nnlib_call($("im2col$S"),
               (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
               xn,x2,Wx,Hx,Cx,Ww,Hw,p1,p2,s1,s2,mode)
        return x2
    end

    function col2im!(w::Array{$T,4}, x::Array{$T,4}, x2::Array{$T,2},
                     n::Int, p1::Int, p2::Int, s1::Int, s2::Int, mode::Int)
        Wx,Hx,Cx,Nx = size(x)
        Ww,Hw,C1,C2 = size(w)
        xn = pointer(x, Wx*Hx*Cx*(n-1)+1)
        @nnlib_call($("col2im$S"),
               (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
               x2,xn,Wx,Hx,Cx,Ww,Hw,p1,p2,s1,s2,mode)
        return x
    end

    ### CPU pooling from Mocha.jl

    function pool(x::Array{$T,4}; window=2, padding=0, stride=window, mode=0,
                  maxpoolingNanOpt=0, alpha=1, handle=nothing)
        if maxpoolingNanOpt!=0
            throw(ArgumentError("CPU pool only supports maxpoolingNanOpt=0"))
        end
        Wx,Hx,Cx,Nx = size(x);
        Wy,Hy,Cy,Ny = pdims(x;window=window,padding=padding,stride=stride)
        y = similar(x, (Wy,Hy,Cy,Ny))
        (w1,w2) = psize(window, x)
        (p1,p2) = psize(padding, x)
        (s1,s2) = psize(stride, x)
        if mode == 0
            @nnlib_call($("max_pooling_fwd$S"),
                   (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                   x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        elseif mode == 1 || (mode == 2 && p1==p2==0)
            @nnlib_call($("mean_pooling_fwd$S"),
                   (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                   x,y,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        else
            throw(ArgumentError("mode $mode not supported by cpu pool"))
        end
        if alpha != 1; scale!(alpha,y); end
        return y
    end

    function pool_grad(x::Array{$T,4}, y::Array{$T,4}, dy::Array{$T,4};
                       window=2, padding=0, stride=window, mode=0,
                       maxpoolingNanOpt=0, alpha=1, handle=nothing)
        if maxpoolingNanOpt!=0;
            throw(ArgumentError("CPU pool only supports maxpoolingNanOpt=0"));
        end
        Wx,Hx,Cx,Nx = size(x);
        Wy,Hy,Cy,Ny = size(y);
        dx = similar(x)
        (w1,w2) = psize(window, x)
        (p1,p2) = psize(padding, x)
        (s1,s2) = psize(stride, x)
        if mode == 0
            if alpha != 1; y = y ./ alpha; end
            @nnlib_call($("max_pooling_bwd$S"),
                   (Ptr{$T},Ptr{$T},Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                   x,y,dy,dx,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        elseif mode == 1 || (mode == 2 && p1==p2==0)
            @nnlib_call($("mean_pooling_bwd$S"),
                   (Ptr{$T},Ptr{$T},Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint,Cint),
                   dx,dy,Wx,Hx,Cx,Nx,Wy,Hy,w1,w2,p1,p2,s1,s2)
        else
            throw(ArgumentError("mode $mode not supported by cpu pool"))
        end
        if alpha != 1; scale!(alpha,dx); end
        return dx
    end
end;end

maxpool2d(x, k; pad = 0) = pool(x; window = k, padding = pad, mode = 0)
avgpool2d(x, k; pad = 0) = pool(x; window = k, padding = pad, mode = 1)
