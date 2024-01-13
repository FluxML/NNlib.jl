function upsample_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    gradtest_fn = Backend == CPU ? gradtest : gputest
    T = Float32


    @testset "Image Rotation" begin
        @testset "SImple test" begin
            arr = device(zeros((6, 6, 1, 1))); 
            arr[3:4, 4, 1, 1] .= 1;
            @test all(cpu(NNlib.imrotate(arr, deg2rad(45))) .≈ [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.29289321881345254 0.585786437626905 0.0; 0.0 0.0 0.08578643762690495 1.0 0.2928932188134524 0.0; 0.0 0.0 0.0 0.08578643762690495 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0])
        end
    end


    @testset "Compare with ImageTransformations" begin
        arr = device(zeros(T, (51, 51, 1, 1)))
        arr[15:40, 15:40, :, :] .= device(1 .+ randn((26, 26)))                                                                       
        
        arr2 = device(zeros(T, (51, 51, 1, 5)))
        arr2[15:40, 15:40, :, :] .= device(arr[15:40, 15:40, :, :])


        for method in [:nearest, :bilinear]
            for angle in deg2rad.([0, 35, 45, 90, 135, 170, 180, 270, 360])
                res1 = cpu(NNlib.imrotate(arr, angle; method))
                res3 = cpu(NNlib.imrotate(arr2, angle; method))
                if method == :nearest
                    res2 = ImageTransformations.imrotate(cpu(arr)[:, :, 1, 1], angle, axes(arr)[1:2], method=Constant(), fillvalue=0)
                elseif method == :bilinear
                    res2 = ImageTransformations.imrotate(cpu(arr)[:, :, 1, 1], angle, axes(arr)[1:2], fillvalue=0)
                end
                @test all(1 .+ res1[20:35, 20:35, :, :] .≈ 1 .+ res2[20:35, 20:35])
                @test all(1 .+ res1[20:35, 20:35, :, :] .≈ 1 .+ res3[20:35, 20:35, :, 1])
                @test all(1 .+ res1[20:35, 20:35, :, :] .≈ 1 .+ res3[20:35, 20:35, :, 2])
            end
        end
        
        arr = device(zeros(T, (52, 52, 1, 1)))
        arr[15:40, 15:40, :, :] .= device(1 .+ randn((26, 26)))
        
        arr2 = device(zeros(T, (52, 52, 5, 1)))
        arr2[15:40, 15:40, :, 1] .= device(arr[15:40, 15:40, :, 1])

        for method in [:nearest, :bilinear]
            for angle in deg2rad.([0, 35,  90, 170, 180, 270, 360])
                res1 = cpu(NNlib.imrotate(arr, angle; method, midpoint=size(arr) .÷2 .+0.5))
                res3 = cpu(NNlib.imrotate(arr2, angle; method, midpoint=size(arr) .÷2 .+0.5))
                if method == :nearest
                    res2 = ImageTransformations.imrotate(cpu(arr)[:, :, 1, 1], angle, axes(arr)[1:2], method=Constant(), fillvalue=0)
                elseif method == :bilinear
                    res2 = ImageTransformations.imrotate(cpu(arr)[:, :, 1, 1], angle, axes(arr)[1:2], fillvalue=0)
                end
                @test all(1 .+ res1[20:35, 20:35, :, :] .≈ 1 .+ res2[20:35, 20:35])
                @test all(1 .+ res1[20:35, 20:35, :, :] .≈ 1 .+ res3[20:35, 20:35, :, :])
                @test all(1 .+ res1[20:35, 20:35, :, :] .≈ 1 .+ res3[20:35, 20:35, :, :])
            end
        end

    end

    @testset "Compare for plausibilty" begin
        arr = zeros(T, (10, 10, 1, 3))
        arr[6, 6, :, 1] .= 1
        arr[6, 6, :, 2] .= 2
        arr[6, 6, :, 3] .= 2

        for method in [:bilinear, :nearest]
            @test all(.≈(arr , NNlib.imrotate(arr, deg2rad(0); method)))
            @test all(.≈(arr , NNlib.imrotate(arr, deg2rad(90); method)))
            @test all(.≈(arr , NNlib.imrotate(arr, deg2rad(180); method)))
            @test all(.≈(arr , NNlib.imrotate(arr, deg2rad(270); method)))
            @test all(.≈(arr , NNlib.imrotate(arr, deg2rad(360); method)))
        end
    end


    @testset "Test gradients" begin

        for method in [:nearest, :bilinear]
            for angle in deg2rad.([0, 45, 90, 137, 180, 270, 360])
                f(x) = sum(abs2.(NNlib.imrotate(x, deg2rad(angle); method)))
       
                img2 = randn(T, (14, 14, 1, 1));
                grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(7, 1), f, img2)[1]
                grad2 = Zygote.gradient(f, img2)[1];
                @test all(.≈(1 .+ grad, 1 .+ grad2, rtol=1f-7))
                
                img2 = randn(T, (9, 9, 1, 2));
                grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(7, 1), f, img2)[1]
                grad2 = Zygote.gradient(f, img2)[1];
                @test all(.≈(1 .+ grad, 1 .+ grad2, rtol=1f-7))

                img2 = randn(T, (8, 12, 1, 1));
                grad = FiniteDifferences.grad(FiniteDifferences.central_fdm(7, 1), f, img2)[1]
                grad2 = Zygote.gradient(f, img2)[1];
                @test all(.≈(1 .+ grad, 1 .+ grad2, rtol=1f-7))
            end
        end
    end
end
