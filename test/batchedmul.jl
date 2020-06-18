function bmm_test(a,b; transA = false, transB = false)
    bs = size(a,3)
    transA && (a = permutedims(a, [2,1,3]))
    transB && (b = permutedims(b, [2,1,3]))
    c = []
    for i = 1:bs
        push!(c, a[:,:,i]*b[:,:,i])
    end

    cat(c...; dims = 3)
end

function bmm_adjtest(a,b; adjA = false, adjB = false)
    bs = size(a,3)
    c = []
    for i = 1:bs
        ai = adjA ? adjoint(a[:,:,i]) : a[:,:,i]
        bi = adjB ? adjoint(b[:,:,i]) : b[:,:,i]
        push!(c, ai*bi)
    end

    cat(c...; dims = 3)
end


@testset "batched_mul: Float64 * $TB" for TB in [Float64, Float32]

    A = randn(7,5,3)
    B = randn(TB, 5,7,3)
    C = randn(7,6,3)

    @test batched_mul(A, B) ≈ bmm_test(A, B)
    @test batched_mul(batched_transpose(A), batched_transpose(B)) ≈ bmm_test(A, B; transA = true, transB = true)
    @test batched_mul(batched_transpose(A), C) ≈ bmm_test(A, C; transA = true)
    @test batched_mul(A, batched_transpose(A)) ≈ bmm_test(A, A; transB = true)


    cA = randn(Complex{Float64}, 7,5,3)
    cB = randn(Complex{TB}, 5,7,3)
    cC = randn(Complex{Float64}, 7,6,3)

    @test batched_mul(cA, cB) ≈ bmm_adjtest(cA, cB)
    @test batched_mul(batched_adjoint(cA), batched_adjoint(cB)) ≈ bmm_adjtest(cA, cB; adjA = true, adjB = true)
    @test batched_mul(batched_adjoint(cA), cC) ≈ bmm_adjtest(cA, cC; adjA = true)
    @test batched_mul(cA, batched_adjoint(cA)) ≈ bmm_adjtest(cA, cA; adjB = true)

    @test batched_transpose(batched_transpose(A)) === A
    @test batched_adjoint(batched_adjoint(cA)) === cA

    TBi = TB==Float64 ? Int64 : Int32
    iA = rand(1:99, 7,5,3)
    iB = TB.(rand(1:99, 5,7,3))
    iC = zeros(Int, 7,6,3)
    @test batched_mul(iA, iB) == bmm_adjtest(iA, iB)
    @test batched_mul(cA, iB) ≈ bmm_adjtest(cA, iB)

    @test_throws DimensionMismatch batched_mul(rand(2,2,2), rand(TB, 2,2,10))
    @test_throws DimensionMismatch batched_mul(rand(2,2,2), rand(TB, 10,2,2))
    @test_throws Exception batched_mul!(zeros(2,2,10), rand(2,2,2), rand(TB, 2,2,2))

end

@testset "BatchedAdjOrTrans interface * $TB" for TB in [Float64, Float32]
    A = randn(7,5,3)
    B = randn(TB, 5,7,3)
    C = randn(7,6,3)

    function interface_tests(X, _X)
        @test length(_X) == length(X)
        @test size(_X) == (size(X, 2), size(X, 1), size(X, 3))
        @test axes(_X) == (axes(X, 2), axes(X, 1), axes(X, 3))
        #
        @test getindex(_X, 2, 3, 3) == getindex(X, 3, 2, 3)
        @test getindex(_X, 5, 4, 1) == getindex(X, 4, 5, 1)
        #
        setindex!(_X, 2.0, 2, 4, 1)
        @test getindex(_X, 2, 4, 1) == 2.0
        setindex!(_X, 3.0, 1, 2, 2)
        @test getindex(_X, 1, 2, 2) == 3.0

        _sim = similar(_X, TB, (2, 3))
        @test size(_sim) == (2, 3)
        @test typeof(_sim) == Array{TB, 2}

        _sim = similar(_X, TB)
        @test length(_sim) == length(_X)
        @test typeof(_sim) == Array{TB, 3}

        _sim = similar(_X, (2, 3))
        @test size(_sim) == (2, 3)
        @test typeof(_sim) == Array{Float64, 2}

        _sim = similar(_X)
        @test length(_sim) == length(_X)
        @test typeof(_sim) == Array{Float64, 3}

        @test parent(_X) == _X.parent
    end

    for (X, _X) in zip([A, B, C], map(batched_adjoint, [A, B, C]))
        interface_tests(X, _X)

        @test -_X == NNlib.BatchedAdjoint(-_X.parent)

        _copyX = copy(_X)
        @test _X == _copyX

        setindex!(_copyX, 2.0, 1, 2, 1)
        @test _X != _copyX
    end

    for (X, _X) in zip([A, B, C], map(batched_transpose, [A, B, C]))
        interface_tests(X, _X)

        @test -_X == NNlib.BatchedTranspose(-_X.parent)

        _copyX = copy(_X)
        @test _X == _copyX

        setindex!(_copyX, 2.0, 1, 2, 1)
        @test _X != _copyX
    end
end
