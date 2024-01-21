function rotation_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    gradtest_fn = Backend == CPU ? gradtest : gputest
    T = Float64
    atol = T == Float32 ? 1e-3 : 1e-6
    rtol = T == Float32 ? 1f-3 : 1f-6
    angles = deg2rad.([0, 0.0001, 35, 90, -90, -90.0123, 170, 180, 270, 360, 450, 1234.1234]) 

    @testset "imrotate" begin
        @testset "Simple test" begin
            arr = device(zeros((6, 6, 1, 1))); 
            arr[3:4, 4, 1, 1] .= 1;
            @test all(cpu(NNlib.imrotate(arr, deg2rad(45))) .≈ [0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.29289321881345254 0.585786437626905 0.0; 0.0 0.0 0.08578643762690495 1.0 0.2928932188134524 0.0; 0.0 0.0 0.0 0.08578643762690495 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0])
        end


        @testset "Compare with ImageTransformations" begin
            for sz in [(51,51,1,1), (52,52,1,1), (51,52,1,1), (52,51,1,1)]
                rotation_center = (sz[1:2] .+ 1) ./ 2  
                arr1 = device(zeros(T, sz))
                arr1[15:40, 15:40, :, :] .= device(1 .+ randn((26, 26)))                                                                       
                arr2 = device(zeros(T, (sz[1], sz[2], sz[3], 3)))
                arr2[15:40, 15:40, :, :] .= device(arr1[15:40, 15:40, :, :])

                for method in [:nearest, :bilinear]
                    @testset "$method" begin
                        for angle in angles
                            res1 = cpu(NNlib.imrotate(arr1, angle; method, rotation_center=rotation_center))
                            res2 = cpu(NNlib.imrotate(arr2, angle; method, rotation_center=rotation_center))
                            if method == :nearest
                                res_IT = ImageTransformations.imrotate(cpu(arr1)[:, :, 1, 1], angle, axes(arr1)[1:2], method=Constant(), fillvalue=0)
                            elseif method == :bilinear
                                res_IT = ImageTransformations.imrotate(cpu(arr1)[:, :, 1, 1], angle, axes(arr1)[1:2], fillvalue=0)
                            end
                            if method == :nearest
                                @test ≈(1 .+ res1[:, :, :, :], 1 .+ res_IT[:, :], rtol=0.5)
                                @test ≈(1 .+ res1[:, :, :, :], 1 .+ res2[:, :,:, 1], rtol=0.5)
                                @test ≈(1 .+ res1[:, :, :, :], 1 .+ res2[:, :,:, 2], rtol=0.5)
                            else
                                @test all(.≈(1 .+ res1[:, :, :, :], 1 .+ res_IT[:, :], rtol=rtol))
                                @test all(.≈(1 .+ res1[:, :, :, :], 1 .+ res2[:, :,:, 1], rtol=rtol))
                                @test all(.≈(1 .+ res1[:, :, :, :], 1 .+ res2[:, :,:, 2], rtol=rtol))
                            end
                        end
                    end
                end
            end
        end
            
        @testset "Compare for plausibilty" begin
            @testset "Special cases of rotation" begin
                arr = device(zeros(T, (10, 10, 1, 3)))
                arr[6, 6, :, 1] .= 1
                arr[6, 6, :, 2] .= 2
                arr[6, 6, :, 3] .= 3

                for method in [:bilinear, :nearest]
                    @test all(.≈(arr , NNlib.imrotate(arr, deg2rad(0); method)))
                    @test all(.≈(arr , NNlib.imrotate(arr, deg2rad(90); method)))
                    @test all(.≈(arr , NNlib.imrotate(arr, deg2rad(180); method)))
                    @test all(.≈(arr , NNlib.imrotate(arr, deg2rad(270); method)))
                    @test all(.≈(arr , NNlib.imrotate(arr, deg2rad(360); method)))
                end
            end
        end

        @testset "Test gradients" begin
            for method in [:nearest, :bilinear]
                for angle in angles 
                    gradtest_fn(
                        x -> NNlib.imrotate(x, angle; method),
                        device(rand(T, 11,11,1,1)); atol)
                    gradtest_fn(
                        x -> NNlib.imrotate(x, angle; method),
                        device(rand(T, 10,10,1,1)); atol)        
                end
            end
        end
    end
end
