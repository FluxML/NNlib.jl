ys = [3, 4, 5, 6, 7]
us = ones(Int, 3, 4)
xs_int = [1 2 3 4;
          4 2 1 3;
          3 5 5 3]
xs_tup = [(1,) (2,) (3,) (4,);
          (4,) (2,) (1,) (3,);
          (3,) (5,) (5,) (3,)]
types = [UInt8, UInt16, UInt32, UInt64, UInt128,
         Int8, Int16, Int32, Int64, Int128, BigInt,
         Float16, Float32, Float64, BigFloat, Rational]

for T = types
    @testset "$T" begin
        PT = promote_type(T, Int)
        @testset "scatter_add!" begin
            ys_ = [5, 6, 9, 8, 9]
            for xs in [xs_int, xs_tup]
                @test scatter_add!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_add!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_add!(copy(ys), T.(us), xs) == PT.(ys_)
            end
        end

        @testset "scatter_sub!" begin
            ys_ = [1, 2, 1, 4, 5]
            for xs in [xs_int, xs_tup]
                @test scatter_sub!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_sub!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_sub!(copy(ys), T.(us), xs) == PT.(ys_)
            end
        end

        @testset "scatter_max!" begin
            ys_ = [3, 4, 5, 6, 7]
            for xs in [xs_int, xs_tup]
                @test scatter_max!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_max!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_max!(copy(ys), T.(us), xs) == PT.(ys_)
            end
        end

        @testset "scatter_min!" begin
            ys_ = [1, 1, 1, 1, 1]
            for xs in [xs_int, xs_tup]
                @test scatter_min!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_min!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_min!(copy(ys), T.(us), xs) == PT.(ys_)
            end
        end

        @testset "scatter_mul!" begin
            ys_ = [3, 4, 5, 6, 7]
            for xs in [xs_int, xs_tup]
                @test scatter_mul!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_mul!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_mul!(copy(ys), T.(us), xs) == PT.(ys_)
            end
        end
    end
end

for T = [Float16, Float32, Float64, BigFloat, Rational]
    @testset "$T" begin
        PT = promote_type(T, Float64)
        @testset "scatter_div!" begin
            us_div = us .* 2
            ys_ = [0.75, 1., 0.3125, 1.5, 1.75]
            for xs in [xs_int, xs_tup]
                @test scatter_div!(T.(copy(ys)), T.(us_div), xs) == T.(ys_)
                @test scatter_div!(T.(copy(ys)), us_div, xs) == PT.(ys_)
                @test scatter_div!(copy(ys), T.(us_div), xs) == PT.(ys_)
            end
        end

        @testset "scatter_mean!" begin
            ys_ = [4., 5., 6., 7., 8.]
            for xs in [xs_int, xs_tup]
                @test scatter_mean!(T.(copy(ys)), T.(us), xs) == T.(ys_)
                @test scatter_mean!(T.(copy(ys)), us, xs) == PT.(ys_)
                @test scatter_mean!(copy(ys), T.(us), xs) == PT.(ys_)
            end
        end
    end
end
