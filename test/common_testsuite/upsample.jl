# Callers that return the result of `upsample_nearest`, non-inlined or guarded. Enzyme
# handed the reverse rule a bogus shadow for `x` in these call shapes on Julia 1.13.
@noinline _upsample_nearest_scale(x, scale) = upsample_nearest(x, scale)
@noinline _upsample_nearest_size(x, size) = upsample_nearest(x; size)
function _upsample_nearest_guarded(x, outsize)
    xsize = size(x)[1:2]
    all(outsize .% xsize .== 0) || throw(ArgumentError("expected output size divisible by input size"))
    return upsample_nearest(x, outsize .÷ xsize)
end

# The matrix of a resized dimension, from the definitions of PyTorch's `upsample_bicubic2d`,
# `_upsample_bicubic2d_aa` and `_upsample_bilinear2d_aa`, in Float64.
function _resize_matrix(n_in, n_out, mode; align_corners, antialias)
    scale = align_corners ? (n_out > 1 ? (n_in - 1) / (n_out - 1) : 0.0) : n_in / n_out
    cubic(x, a) = (x = abs(x); x < 1 ? ((a + 2) * x - (a + 3)) * x^2 + 1 : x < 2 ? a * (((x - 5) * x + 8) * x - 4) : 0.0)
    M = zeros(n_out, n_in)
    for i in 1:n_out
        if antialias
            support = (mode == :bicubic ? 2 : 1) * max(scale, 1)
            center = scale * (i - 0.5)
            for k in max(trunc(Int, center - support + 0.5), 0):min(trunc(Int, center + support + 0.5), n_in) - 1
                u = (k - center + 0.5) / max(scale, 1)
                M[i, k + 1] = mode == :bicubic ? cubic(u, -0.5) : max(1 - abs(u), 0.0)
            end
            M[i, :] ./= sum(M[i, :])
        else
            src = align_corners ? scale * (i - 1) : scale * (i - 0.5) - 0.5
            k = floor(Int, src)
            for (m, u) in zip(k-1:k+2, (src - k + 1, src - k, k + 1 - src, k + 2 - src))
                M[i, clamp(m, 0, n_in - 1) + 1] += cubic(u, -0.75)
            end
        end
    end
    return M
end

function _resize_reference(x, outsize, mode; kws...)
    Mw = _resize_matrix(size(x, 1), outsize[1], mode; kws...)
    Mh = _resize_matrix(size(x, 2), outsize[2], mode; kws...)
    return stack(Mw * x[:, :, c, n] * Mh' for c in axes(x, 3), n in axes(x, 4))
end

function upsample_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    T = Float32 # TODO test against all supported eltypes for each backend.
    atol = T == Float32 ? 1e-3 : 1e-6

    @testset "upsample_nearest, integer scale via reshape" begin
        x = device(reshape(T[1 2; 3 4], (2,2,1,1)))
        @test cpu(upsample_nearest(x, (3,3)))[1,:] == [1,1,1, 2,2,2]

        y = upsample_nearest(x, (2,3))
        @test size(y) == (4,6,1,1)
        y2 = upsample_nearest(x, size=(4,6))
        @test cpu(y) ≈ cpu(y2)

        @test cpu(∇upsample_nearest(y, (2,3)))[:, :, 1, 1] == [6 12; 18 24]
        @test test_gradients(
            x -> upsample_nearest(x, (2,3)),
            rand(T, 2,2,1,1); test_gpu = Backend != CPU, atol)
        @test test_gradients(
            x -> upsample_nearest(x, size=(4,6)),
            rand(T, 2,2,1,1); test_gpu = Backend != CPU, atol)
        # `upsample_nearest` reached through a function that returns its result.
        @test test_gradients(
            x -> _upsample_nearest_scale(x, (2,3)),
            rand(T, 2,2,1,1); test_gpu = Backend != CPU, atol)
        @test test_gradients(
            x -> _upsample_nearest_size(x, (4,6)),
            rand(T, 2,2,1,1); test_gpu = Backend != CPU, atol)
        @test test_gradients(
            x -> _upsample_nearest_guarded(x, (4,6)),
            rand(T, 2,2,1,1); test_gpu = Backend != CPU, atol)

        @test_throws ArgumentError ∇upsample_nearest(y, (2,4))
        @test_throws ArgumentError upsample_nearest(x, (1,2,3,4,5))
        @test_throws ArgumentError upsample_nearest(x, size=(3,4))
    end

    @testset "Linear upsampling (1D)" begin
        x = T[1,2,3,4]
        x = hcat(x,x,x)[:,:,:]

        y = collect(1:1//3:4)
        y = hcat(y,y,y)[:,:,:]

        xd = device(x)
        @test y ≈ cpu(upsample_linear(xd, 2.5))
        @test y ≈ cpu(upsample_linear(xd; size=10))
        @test test_gradients(x -> upsample_linear(x, 2.5), x; test_gpu = Backend != CPU, atol)
        @test cpu(∇upsample_linear(xd; size = 2)) == cpu(∇upsample_linear(xd; size = (2,)))
    end

    # Float16 positions past 2048 pixels used to be off by whole pixels (BFloat16 is
    # tested in gpu/cuda/upsample.jl, rounding on the device).
    Backend == CPU && @testset "Resizing a long Float16 axis" begin
        short, long = randn(Float32, 600, 5, 2, 1), randn(Float32, 2999, 5, 2, 1)
        for (f, x) in ((x -> upsample_bilinear(x; size = (2999, 5)), short),
                       (x -> ∇upsample_bilinear(x; size = (600, 5)), long),
                       (x -> upsample_bicubic(x; size = (2999, 5)), short),
                       (x -> ∇upsample_bicubic(x; size = (600, 5)), long),
                       (x -> upsample_bicubic(x; size = (600, 5), antialias = true), long),
                       (x -> ∇upsample_bicubic(x; size = (2999, 5), antialias = true), short),
                       (x -> upsample_bilinear(x; size = (600, 5), align_corners = false, antialias = true), long))
            x16 = Float16.(x)
            @test Float32.(f(x16)) ≈ f(Float32.(x16)) rtol = 1e-2
        end
    end

    @testset "Bilinear upsampling (2D)" begin
        x = Float32[1 2; 3 4][:,:,:,:]
        x = cat(x,x; dims=3)
        x = cat(x,x; dims=4)

        # this output matches the one of pytorch v1.5.0
        # nn.UpsamplingBilinear2d(scale_factor=(3,2), align_corners=True)
        # for above x
        y_true = Float32[ 1//1  4//3   5//3   2//1;
                          7//5 26//15 31//15 12//5;
                          9//5 32//15 37//15 14//5;
                         11//5 38//15 43//15 16//5;
                         13//5 44//15 49//15 18//5;
                          3//1 10//3  11//3   4//1][:,:,:,:]
        y_true = cat(y_true, y_true; dims=3)
        y_true = cat(y_true, y_true; dims=4)

        xd = device(x)
        y = upsample_bilinear(xd, (3, 2))
        @test size(y) == size(y_true)
        @test eltype(y) == Float32
        @test cpu(y) ≈ y_true

        @test test_gradients(x -> upsample_bilinear(x, (3, 2)), x; test_gpu = Backend != CPU, atol)

        # additional grad check, also compliant with pytorch
        o = ones(Float32,6,4,2,1)
        grad_true = 6*ones(Float32,2,2,2,1)
        @test cpu(∇upsample_bilinear(device(o); size = (2,2))) ≈ grad_true

        # CPU only tests.

        y_true_2 = Rational{Int}[1//1  5//4  6//4  7//4 2//1;
                                 3//2  7//4  8//4  9//4 5//2;
                                 4//2  9//4 10//4 11//4 6//2;
                                 5//2 11//4 12//4 13//4 7//2;
                                 3//1 13//4 14//4 15//4 4//1][:,:,:,:]
        y_true_2 = cat(y_true_2, y_true_2; dims=3)
        y_true_2 = cat(y_true_2, y_true_2; dims=4)

        # check for real-valued single-number argument and type stability for rationals
        y_rational = upsample_bilinear(Rational{Int}.(x), 2.5)
        @test eltype(y_rational) == Rational{Int}
        @test y_rational == y_true_2

        # check Integer support for forward pass
        # grads are always assumed to be floats, so no extension there
        x = UInt8[1 3; 3 5][:,:,:,:]
        y_true_int = UInt8[1 2 3; 2 3 4; 3 4 5][:,:,:,:]
        y = upsample_bilinear(x, 1.5)

        @test eltype(y) == UInt8
        @test y == y_true_int
    end

    @testset "Bicubic and antialiased resizing (2D)" begin
        x = T[mod((i + 2j)^2, 7) - 3 for i in 1:3, j in 1:2, c in 1:1, n in 1:1]
        x6 = T[mod((i + 2j)^2, 7) - 3 for i in 1:6, j in 1:5, c in 1:1, n in 1:1]
        # PyTorch 2.13 (CPU): F.interpolate of the same images in the NCHW layout.
        y_torch = T[-1.24424 -0.8183425 0.107666016 1.0336756 1.4595727;
                    -1.2650003 -1.1860539 -1.0144043 -0.8427546 -0.76380867;
                    -0.013523961 -0.47697026 -1.4846191 -2.492269 -2.9557154;
                    1.6082944 0.79789674 -0.9641113 -2.7261212 -3.5365186]
        @test cpu(upsample_bicubic(device(x); size = (4, 5)))[:, :, 1, 1] ≈ y_torch rtol = 1e-6
        y_torch = T[-1.0 0.0 1.0;
                    -1.1875 -0.796875 -0.40625;
                    -1.0 -1.5 -2.0;
                    0.0 -1.390625 -2.78125;
                    1.0 -1.0 -3.0]
        @test cpu(upsample_bicubic(device(x); size = (5, 3), align_corners = true))[:, :, 1, 1] ≈ y_torch rtol = 1e-6
        y_torch = T[-0.9680162 -1.1688496 -0.52510136;
                    -1.5228745 -0.5194615 -1.4633604]
        @test cpu(upsample_bicubic(device(x6); size = (2, 3), antialias = true))[:, :, 1, 1] ≈ y_torch rtol = 1e-6
        y_torch = T[-1.038961 -0.6428571;
                    -1.1988637 -0.965909;
                    -1.0129871 -1.1363636]
        @test cpu(upsample_bilinear(device(x6); size = (3, 2), align_corners = false, antialias = true))[:, :, 1, 1] ≈ y_torch rtol = 1e-6

        sizes = [((5, 7), (10, 14)), ((8, 8), (3, 5)), ((6, 9), (13, 4)), ((1, 1), (3, 2)),
                 ((2, 3), (1, 1)), ((3, 2), (7, 5)), ((17, 13), (4, 3)), ((4, 4), (4, 4))]
        for (insize, outsize) in sizes, mode in (:bicubic, :bilinear), align_corners in (false, true), antialias in (false, true)
            mode == :bilinear && !antialias && continue
            f = mode == :bicubic ? upsample_bicubic : upsample_bilinear
            ∇f = mode == :bicubic ? ∇upsample_bicubic : ∇upsample_bilinear
            x = rand(T, insize..., 2, 3)
            Δ = rand(T, outsize..., 2, 3)
            y = f(device(x); size = outsize, align_corners, antialias)
            @test y isa typeof(device(x)) && size(y) == size(Δ)
            @test cpu(y) ≈ _resize_reference(x, outsize, mode; align_corners, antialias) rtol = 1e-5
            dx = ∇f(device(Δ); size = insize, align_corners, antialias)
            dx_ref = stack(_resize_matrix(insize[1], outsize[1], mode; align_corners, antialias)' * Δ[:, :, c, n] *
                           _resize_matrix(insize[2], outsize[2], mode; align_corners, antialias) for c in 1:2, n in 1:3)
            @test cpu(dx) ≈ dx_ref rtol = 1e-5
        end

        x = device(rand(T, 6, 5, 2, 1))
        @test upsample_bicubic(x, 2) == upsample_bicubic(x; size = (12, 10))
        @test upsample_bicubic(x, (0.5, 1.5); antialias = true) == upsample_bicubic(x; size = (3, 7), antialias = true)
        @test upsample_bilinear(x, (0.5, 1.5); antialias = true) == upsample_bilinear(x; size = (3, 7), antialias = true)
        @test_throws ArgumentError upsample_bicubic(x; size = (0, 3))

        @test test_gradients(x -> upsample_bicubic(x, (2, 1.5)), rand(T, 3, 4, 2, 1); test_gpu = Backend != CPU, atol)
        @test test_gradients(x -> upsample_bicubic(x; size = (3, 2), antialias = true), rand(T, 7, 5, 2, 1);
                             test_gpu = Backend != CPU, atol)
        @test test_gradients(x -> upsample_bilinear(x; size = (2, 4), align_corners = false, antialias = true),
                             rand(T, 5, 9, 1, 2); test_gpu = Backend != CPU, atol)
    end

    @testset "Trilinear upsampling (3D)" begin
        # Layout: WHDCN, where D is depth
        # we generate data which is constant along W & H and differs in D
        # then we upsample along all dimensions
        x = ones(T, 3,3,3,1,1)
        x[:,:,1,:,:] .= 1.
        x[:,:,2,:,:] .= 2.
        x[:,:,3,:,:] .= 3.

        y_true = ones(T, 5,5,5,1,1)
        y_true[:,:,1,:,:] .= 1.
        y_true[:,:,2,:,:] .= 1.5
        y_true[:,:,3,:,:] .= 2.
        y_true[:,:,4,:,:] .= 2.5
        y_true[:,:,5,:,:] .= 3.

        xd = device(x)
        y = upsample_trilinear(xd; size=(5,5,5))

        @test size(y) == size(y_true)
        @test eltype(y) == T
        @test collect(y) ≈ collect(y_true)

        @test test_gradients(
            x -> upsample_trilinear(x, (2,2,2)), x;
            test_gpu = Backend != CPU, atol=(T == Float32) ? 1e-2 : 1e-5)

        # This test only works when `align_corners=false`.
        o = device(ones(Float32,8,8,8,1,1))
        grad_true = 8 * ones(Float32,4,4,4,1,1)
        @test cpu(∇upsample_trilinear(o; size=(4,4,4), align_corners=false)) ≈ grad_true
    end

    @testset "pixel_shuffle" begin
        x = reshape(1:16, (2, 2, 4, 1))
        # [:, :, 1, 1] =
        #     1  3
        #     2  4
        # [:, :, 2, 1] =
        #     5  7
        #     6  8
        # [:, :, 3, 1] =
        #     9  11
        #     10  12
        # [:, :, 4, 1] =
        #     13  15
        #     14  16

        y_true = [1  9 3 11
                  5 13 7 15
                  2 10 4 12
                  6 14 8 16][:,:,:,:]

        y = pixel_shuffle(device(x), 2)
        @test size(y) == size(y_true)
        @test y_true == cpu(y)

        x = reshape(1:32, (2, 2, 8, 1))
        y_true = zeros(Int, 4, 4, 2, 1)
        y_true[:,:,1,1] .= [ 1   9  3  11
                             5  13  7  15
                             2  10  4  12
                             6  14  8  16 ]

        y_true[:,:,2,1] .= [ 17  25  19  27
                             21  29  23  31
                             18  26  20  28
                             22  30  24  32]

        y = pixel_shuffle(device(x), 2)
        @test size(y) == size(y_true)
        @test y_true == cpu(y)

        x = reshape(1:4*3*27*2, (4,3,27,2))
        y = pixel_shuffle(device(x), 3)
        @test size(y) == (12, 9, 3, 2)

        # batch dimension is preserved
        x1 = x[:,:,:,[1]]
        x2 = x[:,:,:,[2]]
        y1 = pixel_shuffle(device(x1), 3)
        y2 = pixel_shuffle(device(x2), 3)
        @test cpu(cat(y1, y2, dims=4)) == cpu(y)

        for d in [1, 2, 3]
            r = rand(1:5)
            n = rand(1:5)
            c = rand(1:5)
            insize = rand(1:5, d)
            x = rand(Float32, insize..., r^d*c, n)   # Float32: Metal has no Float64
            xd = device(x)

            y = pixel_shuffle(xd, r)
            @test size(y) == ((r .* insize)..., c, n)
            @test test_gradients(x -> pixel_shuffle(x, r), x; test_gpu = Backend != CPU)
        end
    end

    @testset "Complex-valued upsample" begin
        for (d, method) in zip([1, 2, 3], [upsample_linear, upsample_bilinear, upsample_trilinear])
            for (k, interp) in zip((2, ntuple(_ -> 2,  d)), [method, upsample_nearest])
                x = device(randn(Complex{Float32}, (4,8,12)[1:d]..., 1, 1))

                upsize = (8, 16, 24)[1:d]
                xup = interp(x, k)
                @test size(xup)[1:d] == upsize
                @test cpu(real(xup)) == cpu(interp(real(x), k))
                @test cpu(imag(xup)) == cpu(interp(imag(x), k))

                upsize = (8,24,48)[1:d]
                xup = interp(x; size=upsize)
                @test size(xup)[1:d] == upsize
                @test cpu(real(xup)) == cpu(interp(real(x), size=upsize))
                @test cpu(imag(xup)) == cpu(interp(imag(x), size=upsize))
            end
        end
    end
end
