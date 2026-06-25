@testset "dropout + Metal" begin
    x1 = Metal.rand(Float32, 3, 4)
    @test size(dropout(x1, 0.1)) == (3, 4)
    @test dropout(x1, 0.1) isa MtlArray{Float32}

    # The `dropout` rrule must not let a `Float64` `p` leak into the Metal kernel
    # (Metal has no Float64 support). `p = 0.0` is the MultiHeadAttention default.
    @testset "Zygote grad, p=$p" for p in (0.0, 0.3)
        @test gradient(x -> sum(dropout(x, p)), x1)[1] isa MtlArray{Float32}
        @test gradient(x -> sum(dropout(x, p; dims=1)), x1)[1] isa MtlArray{Float32}
    end
end
