using NNlib

function spectral_testsuite(Backend)
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

    @testset "STFT" begin
        cpu(x) = adapt(CPU(), x)
        device(x) = adapt(Backend(), x)

        for batch in ((), (3,))
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
    end
end
