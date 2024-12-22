function spectral_testsuite(Backend)
    cpu(x) = adapt(CPU(), x)
    device(x) = adapt(Backend(), x)
    gradtest_fn = Backend == CPU ? gradtest : gputest

    @testset "Window functions" begin
        for window_fn in (hann_window, hamming_window)
            @inferred window_fn(10, Float32)
            @inferred window_fn(10, Float64)

            w = window_fn(10)
            @test length(w) == 10
            @test eltype(w) == Float32

            wp = window_fn(10; periodic=false)
            @test wp[1:5] ≈ reverse(wp[6:10])

            @test window_fn(10; periodic=true) ≈ window_fn(10 + 1; periodic=false)[1:10]
        end
    end

    @testset "STFT" for batch in ((), (3,))
        @testset "Grads" begin
            if Backend != CPU
                x = rand(Float32, 16, batch...)
                window = hann_window(16)

                gradtest_fn(s -> abs.(stft(s; n_fft=16)), x)
                gradtest_fn((s, w) -> abs.(stft(s; n_fft=16, window=w)), x, window)

                x = rand(Float32, 2045, batch...)
                n_fft = 256
                window = hann_window(n_fft)
                gradtest_fn((s, w) -> abs.(stft(s; n_fft, window=w)), x, window)
                gradtest_fn((s, w) -> abs.(stft(s; n_fft, window=w, center=false)), x, window)
                gradtest_fn((s, w) -> abs.(stft(s; n_fft, window=w, normalized=true)), x, window)
            end
        end

        @testset "Batch $batch" begin
            x = device(ones(Float32, 16, batch...))
            # TODO fix type stability for pad_reflect
            # @inferred stft(x; n_fft=16)

            bd = ntuple(_ -> Colon(), length(batch))

            y = stft(x; n_fft=16)
            @test size(y) == (9, 5, batch...)
            @test all(real(cpu(y))[1, :, bd...] .≈ 16)

            xx = istft(y; n_fft=16)
            @test size(xx) == (16, batch...)
            @test cpu(x) ≈ cpu(xx)

            # Test multiple hops.
            x = device(rand(Float32, 2048, batch...))
            y = stft(x; n_fft=1024)
            xx = istft(y; n_fft=1024)
            @test cpu(x) ≈ cpu(xx)

            # Test odd sizes.
            x = device(rand(Float32, 1111, batch...))
            y = stft(x; n_fft=256)
            xx = istft(y; n_fft=256, original_length=size(x, 1))
            @test cpu(x) ≈ cpu(xx)

            # Output from inverse is cropped on the right
            # without knowing the original size.
            xx = istft(y; n_fft=256)
            @test length(xx) < length(x)
            @test cpu(x)[[1:s for s in size(xx)]...] ≈ cpu(xx)

            # Test different options.

            # Normalized.
            x = device(rand(Float32, 1234, batch...))
            y = stft(x; n_fft=512, normalized=true)
            xx = istft(y; n_fft=512, normalized=true, original_length=size(x, 1))
            @test cpu(x) ≈ cpu(xx)

            # With window.
            window = device(hann_window(512))
            y = stft(x; n_fft=512, window)
            xx = istft(y; n_fft=512, window, original_length=size(x, 1))
            @test cpu(x) ≈ cpu(xx)

            # Hop.
            for hop_length in (32, 33, 255, 256, 511, 512)
                y = stft(x; n_fft=512, hop_length)
                xx = istft(y; n_fft=512, hop_length, original_length=size(x, 1))
                @test cpu(x) ≈ cpu(xx)
            end

            # N FFT.
            for n_fft in (32, 33, 64, 65, 128, 129, 512)
                y = stft(x; n_fft)
                xx = istft(y; n_fft, original_length=size(x, 1))
                @test cpu(x) ≈ cpu(xx)
            end
        end
    end

    @testset "Spectrogram" begin
        x = device(rand(Float32, 1024))
        window = device(hann_window(1024))

        y = stft(x;
            n_fft=1024, hop_length=128, window,
            center=true, normalized=false)
        spec = spectrogram(x;
            n_fft=1024, hop_length=128, window,
            center=true, normalized=false)
        @test abs.(y).^2 ≈ spec

        # Gradient with `0`s in spectrogram.
        # We add small ϵ to spectrogram before computing power
        # to prevent `NaN` in gradient due to `abs(0)`.
        x = device(ones(Float32, 1024))
        g = Zygote.gradient(x) do x
            sum(spectrogram(x;
                n_fft=1024, hop_length=128, window,
                center=true, normalized=false))
        end
        @test !any(isnan.(g[1]))

        # Batched.
        x = device(rand(Float32, 1024, 3))
        spec = spectrogram(x;
            n_fft=1024, hop_length=128, window,
            center=true, normalized=false)
        for i in 1:3
            y = stft(x[:, i];
                n_fft=1024, hop_length=128, window,
                center=true, normalized=false)
            @test abs.(y).^2 ≈ spec[:, :, i]
        end

        if Backend != CPU
            @testset "Grads" begin
                for batch in ((), (3,))
                    x = rand(Float32, 2045, batch...)
                    n_fft = 256
                    window = hann_window(n_fft)
                    gradtest_fn((s, w) -> spectrogram(s; n_fft, hop_length=n_fft ÷ 4, window=w), x, window)
                    gradtest_fn((s, w) -> spectrogram(s; n_fft, hop_length=n_fft ÷ 4, window=w, center=false), x, window)
                    gradtest_fn((s, w) -> spectrogram(s; n_fft, hop_length=n_fft ÷ 4, window=w, normalized=true), x, window)
                end
            end
        end
    end

    @testset "Power to dB" begin
        x = device(rand(Float32, 1024))
        window = device(hann_window(1024))
        spec = spectrogram(x; pad=0, n_fft=1024, hop_length=128, window)

        @test spec ≈ NNlib.db_to_power(NNlib.power_to_db(spec))
        @inferred NNlib.power_to_db(spec)
        @inferred NNlib.db_to_power(NNlib.power_to_db(spec))
    end
end
