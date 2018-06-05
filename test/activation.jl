ACTIVATION_FUNCTIONS = [σ, relu, leakyrelu, elu, swish, selu, softplus, softsign];

function test_value_float_precision_preserving(a)
    @testset "$(a): " begin
        for T in [Float32, Float64]
            for val in [-10, -1, 0, 1, 10]
                val = @inferred a(T(val))
                @test typeof(val) == T
            end
        end
    end
end

function test_value_int_input_forces_float64(a)
    @testset "$(a): " begin
        for T in [Int32, Int64]
            for val in [-10, -1, 0, 1, 10]
                val = @inferred a(T(val))
                @test typeof(val) == Float64
            end
        end
    end
end

# if Base.find_in_path("ForwardDiff") ≠ nothing
#     using ForwardDiff
#     function test_value_duals(a)
#         @testset "$(a): " begin
#         for T in [Float32, Float64, Int32, Int64]
#           val = @inferred a(ForwardDiff.Dual(float(T(1)), one(float(T))))
#           @test typeof(val) == ForwardDiff.Dual{Nothing,float(T),1}
#         end
#         end
#     end
#
#     test_value_duals.(ACTIVATION_FUNCTIONS)
# end

@testset "Activation Functions" begin

  @test σ(0.0) == 0.5
  @test relu(0.0) == 0.0
  @test leakyrelu(0.0) == 0.0
  @test elu(0.0) == 0.0
  @test swish(0.0) == 0.0
  @test softplus(0.0) ≈ log(2.0)
  @test softsign(0.0) == 0.0
  @test selu(0.0) == 0.0

  @test σ(1.0) == 1.0 / (1.0 + exp(-1.0))
  @test relu(1.0) == 1.0
  @test leakyrelu(1.0) == 1.0
  @test elu(1.0) == 1.0
  @test swish(1.0) == 1.0 / (1.0 + exp(-1.0))
  @test softplus(1.0) ≈ log(exp(1.0) + 1.0)
  @test softsign(1.0) == 0.5
  @test selu(1.0) == 1.0507009873554804934193349852946

  @test σ(-1.0) == 1.0 / (1.0 + exp(1.0))
  @test relu(-1.0) == 0.0
  @test leakyrelu(-1.0) == -0.01
  @test elu(-1.0) == exp(-1.0) - 1.0
  @test swish(-1.0) == -1.0 / (1.0 + exp(1.0))
  @test softplus(-1.0) ≈ log(exp(-1.0) + 1.0)
  @test softsign(-1.0) == -0.5
  @test selu(-1.0) == 1.0507009873554804934193349852946 * 1.6732632423543772848170429916717 * (exp(-1.0) - 1.0)

  @testset "Float inference" begin
    test_value_float_precision_preserving.(ACTIVATION_FUNCTIONS)
  end

  @testset "Test Integer64 and Integer32 inputs will force Float64 outputs" begin
    test_value_int_input_forces_float64.(filter(x -> x != relu, ACTIVATION_FUNCTIONS))

    @testset "relu: " begin
      # relu doesn't have to force floating point outputs
      @test typeof(relu(Int64(1))) == Int64
      @test typeof(relu(Int32(1))) == Int32
    end
  end


  xs = rand(5,5)

  @test all(sum(softmax(xs), dims = 1) .≈ 1)

  @test sum(softmax(vec(xs))) ≈ 1

  @testset "elu" begin
      @test elu(42) == 42
      @test elu(42.) == 42.

      @test elu(-4) ≈ (exp(-4) - 1)
  end

  @test leakyrelu( 0.4,0.3) ≈  0.4
  @test leakyrelu(-0.4,0.3) ≈ -0.12

end
