dst = [3, 4, 5, 6, 7]
src = ones(Int, 3, 4)
src2 = [1 2 3 4;
       1 2 3 4;
       1 2 3 4]
idx_int = [1 2 3 4;
          4 2 1 3;
          3 5 5 3]
idx_tup = [(1,) (2,) (3,) (4,);
          (4,) (2,) (1,) (3,);
          (3,) (5,) (5,) (3,)]
types = [UInt8, UInt16, UInt32, UInt64, UInt128,
         Int8, Int16, Int32, Int64, Int128,
         Float16, Float32, Float64, BigFloat, Rational]

for T = types
    @testset "$T" begin
        PT = promote_type(T, Int)
        @testset "scatter_add!" begin
            dst_ = [5, 6, 9, 8, 9]
            for idx in [idx_int, idx_tup]
                @test scatter!(+, T.(copy(dst)), T.(src), idx, dims=0) == T.(dst_)
                @test scatter!(+, T.(copy(dst)), src, idx, dims=0) == PT.(dst_)
                @test scatter!(+, copy(dst), T.(src), idx, dims=0) == PT.(dst_)
                @test scatter(+, T.(src2), idx, dims=0) == T[4, 4, 12, 5, 5]
            end
        end

        @testset "scatter_sub!" begin
            dst_ = [1, 2, 1, 4, 5]
            for idx in [idx_int, idx_tup]
                @test scatter!(-, T.(copy(dst)), T.(src), idx, dims=0) == T.(dst_)
                @test scatter!(-, T.(copy(dst)), src, idx, dims=0) == PT.(dst_)
                @test scatter!(-, copy(dst), T.(src), idx, dims=0) == PT.(dst_)
                if !(T in [UInt8, UInt16, UInt32, UInt64, UInt128])
                    @test scatter(-, T.(src2), idx, dims=0) == T[-4, -4, -12, -5, -5]
                end
            end
        end

        @testset "scatter_max!" begin
            dst_ = [3, 4, 5, 6, 7]
            for idx in [idx_int, idx_tup]
                @test scatter!(max, T.(copy(dst)), T.(src), idx, dims=0) == T.(dst_)
                @test scatter!(max, T.(copy(dst)), src, idx, dims=0) == PT.(dst_)
                @test scatter!(max, copy(dst), T.(src), idx, dims=0) == PT.(dst_)
                @test scatter(max, T.(src2), idx, dims=0) == T[3, 2, 4, 4, 3]
            end
        end

        @testset "scatter_min!" begin
            dst_ = [1, 1, 1, 1, 1]
            for idx in [idx_int, idx_tup]
                @test scatter!(min, T.(copy(dst)), T.(src), idx, dims=0) == T.(dst_)
                @test scatter!(min, T.(copy(dst)), src, idx, dims=0) == PT.(dst_)
                @test scatter!(min, copy(dst), T.(src), idx, dims=0) == PT.(dst_)
                @test scatter(min, T.(src2), idx, dims=0) == T[1, 2, 1, 1, 2]
            end
        end

        @testset "scatter_mul!" begin
            dst_ = [3, 4, 5, 6, 7]
            for idx in [idx_int, idx_tup]
                @test scatter!(*, T.(copy(dst)), T.(src), idx, dims=0) == T.(dst_)
                @test scatter!(*, T.(copy(dst)), src, idx, dims=0) == PT.(dst_)
                @test scatter!(*, copy(dst), T.(src), idx, dims=0) == PT.(dst_)
                @test scatter(*, T.(src2), idx, dims=0) == T[3, 4, 48, 4, 6]
            end
        end
    end
end

for T = [Float16, Float32, Float64, BigFloat, Rational]
    @testset "$T" begin
        PT = promote_type(T, Float64)
        @testset "scatter_div!" begin
            us_div = src .* 2
            dst_ = [0.75, 1., 0.3125, 1.5, 1.75]
            for idx in [idx_int, idx_tup]
                @test scatter!(/, T.(copy(dst)), T.(us_div), idx, dims=0) == T.(dst_)
                @test scatter!(/, T.(copy(dst)), us_div, idx, dims=0) == PT.(dst_)
                @test scatter!(/, copy(dst), T.(us_div), idx, dims=0) == PT.(dst_)
                @test scatter(/, T.(src2), idx, dims=0) == T[1//3, 1//4, 1//48, 1//4, 1//6]
            end
        end

        @testset "scatter_mean!" begin
            dst_ = [4., 5., 6., 7., 8.]
            for idx in [idx_int, idx_tup]
                @test scatter!(mean, T.(copy(dst)), T.(src), idx, dims=0) == T.(dst_)
                @test scatter!(mean, T.(copy(dst)), src, idx, dims=0) == PT.(dst_)
                @test scatter!(mean, copy(dst), T.(src), idx, dims=0) == PT.(dst_)
                @test scatter(mean, T.(src2), idx, dims=0) == T[2, 2, 3, 2.5, 2.5]
            end
        end
    end
end

dst = [3. 3. 4. 4. 5.;
      5. 5. 6. 6. 7.]
src = 2*ones(2, 3, 4)
idx = [1 2 3 4;
      4 2 1 3;
      3 5 5 3]

∇y_mul = [4. 4. 16. 4. 4.; 4. 4. 16. 4. 4.]
∇y_div = [.25 .25 .0625 .25 .25; .25 .25 .0625 .25 .25]
∇u_mean = cat([.5 .5 .25; .5 .5 .25], [.5 .5 .5; .5 .5 .5],
              [.25 .5 .5; .25 .5 .5], [.5 .25 .25; .5 .25 .25], dims=3)

@testset "∇scatter" begin
    @test Zygote.gradient(x -> sum(scatter!(+, x, src, idx, dims=1)), dst) == (ones(2, 5),)
    @test Zygote.gradient(x -> sum(scatter!(+, copy(dst), x, idx, dims=1)), src) == (ones(2, 3, 4),)
    @test Zygote.gradient(x -> sum(scatter!(+, copy(dst), src, x, dims=1)), idx) == (nothing,)

    @test Zygote.gradient(x -> sum(scatter!(-, x, src, idx, dims=1)), dst) == (ones(2, 5),)
    @test Zygote.gradient(x -> sum(scatter!(-, copy(dst), x, idx, dims=1)), src) == (-ones(2, 3, 4),)
    @test Zygote.gradient(x -> sum(scatter!(-, copy(dst), src, x, dims=1)), idx) == (nothing,)

    @test Zygote.gradient(x -> sum(scatter!(max, x, src, idx, dims=1)), dst) == (ones(2, 5),)
    @test Zygote.gradient(x -> sum(scatter!(max, copy(dst), x, idx, dims=1)), src) == (zeros(2, 3, 4),)
    @test Zygote.gradient(x -> sum(scatter!(max, copy(dst), src, x, dims=1)), idx) == (nothing,)

    @test Zygote.gradient(x -> sum(scatter!(min, x, src, idx, dims=1)), dst) == (zeros(2, 5),)
    @test Zygote.gradient(x -> sum(scatter!(min, copy(dst), x, idx, dims=1)), src) == (ones(2, 3, 4),)
    @test Zygote.gradient(x -> sum(scatter!(min, copy(dst), src, x, dims=1)), idx) == (nothing,)

    @test Zygote.gradient(x -> sum(scatter!(*, x, src, idx, dims=1)), dst) == (∇y_mul,)
    @test Zygote.gradient(x -> sum(scatter!(*, copy(dst), x, idx, dims=1)), src) == (2048*gather(dst, idx),)
    @test Zygote.gradient(x -> sum(scatter!(*, copy(dst), src, x, dims=1)), idx) == (nothing,)

    @test Zygote.gradient(x -> sum(scatter!(/, x, src, idx, dims=1)), dst) == (∇y_div,)
    @test Zygote.gradient(x -> sum(scatter!(/, copy(dst), x, idx, dims=1)), src) == (-gather(dst, idx)/8192,)
    @test Zygote.gradient(x -> sum(scatter!(/, copy(dst), src, x, dims=1)), idx) == (nothing,)

    @test Zygote.gradient(x -> sum(scatter!(mean, x, src, idx, dims=1)), dst) == (ones(2, 5),)
    @test Zygote.gradient(x -> sum(scatter!(mean, copy(dst), x, idx, dims=1)), src) == (∇u_mean,)
    @test Zygote.gradient(x -> sum(scatter!(mean, copy(dst), src, x, dims=1)), idx) == (nothing,)
end
