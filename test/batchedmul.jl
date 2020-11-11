using NNlib, Test, LinearAlgebra
using NNlib: storage_type, storage_typejoin, is_strided,
    batched_mul!, _unbatch, _copy_if_faster, BatchedAdjoint

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

function half_batched_mul(x,y)
    @assert size(y,3) == 1
    d = size(x,2)
    x_mat = reshape(permutedims(x, (1,3,2)),:,d)
    y_mat = reshape(y,d,:)
    z_mat = x_mat * y_mat
    permutedims(reshape(z_mat, size(x,1), size(x,3), :), (1,3,2))
end

@testset "batched_mul: Float64 * $TB" for TB in [Float64, Float32]

    # Real
    A = randn(7,5,3)
    B = randn(TB, 5,7,3)
    C = randn(7,6,3)

    @test batched_mul(A, B) ≈ bmm_test(A, B)
    @test batched_mul(batched_transpose(A), batched_transpose(B)) ≈ bmm_test(A, B; transA = true, transB = true)
    @test batched_mul(batched_transpose(A), C) ≈ bmm_test(A, C; transA = true)
    @test batched_mul(A, batched_transpose(A)) ≈ bmm_test(A, A; transB = true)

    # Complex
    cA = randn(Complex{Float64}, 7,5,3)
    cB = randn(Complex{TB}, 5,7,3)
    cC = randn(Complex{Float64}, 7,6,3)

    @test batched_mul(cA, cB) ≈ bmm_adjtest(cA, cB)
    @test batched_mul(batched_adjoint(cA), batched_adjoint(cB)) ≈ bmm_adjtest(cA, cB; adjA = true, adjB = true)
    @test batched_mul(batched_adjoint(cA), cC) ≈ bmm_adjtest(cA, cC; adjA = true)
    @test batched_mul(cA, batched_adjoint(cA)) ≈ bmm_adjtest(cA, cA; adjB = true)

    # Wrappers which cancel
    @test batched_transpose(batched_transpose(A)) === A
    @test batched_transpose(PermutedDimsArray(A, (2,1,3))) === A
    @test batched_adjoint(batched_adjoint(cA)) === cA
    @test batched_transpose(batched_adjoint(cA)) isa NNlib.BatchedAdjoint

    # Integers
    TBi = TB==Float64 ? Int64 : Int32
    iA = rand(1:99, 7,5,3)
    iB = TB.(rand(1:99, 5,7,3))
    iC = zeros(Int, 7,6,3)
    @test batched_mul(iA, iB) == bmm_adjtest(iA, iB)
    @test batched_mul(cA, iB) ≈ bmm_adjtest(cA, iB)

    # Errors
    @test_throws DimensionMismatch batched_mul(rand(2,2,2), rand(TB, 2,2,10))
    @test_throws DimensionMismatch batched_mul(rand(2,2,2), rand(TB, 10,2,2))
    @test_throws Exception batched_mul!(zeros(2,2,10), rand(2,2,2), rand(TB, 2,2,2))

    # PermutedDimsArrays
    for perm in [(1,3,2), (2,1,3), (3,2,1)], fun in [identity, batched_adjoint], ty in [identity, complex]
        A = randn(ty(Float64), 4,4,4)
        B = randn(ty(TB), 4,4,4)
        @test batched_mul(fun(A), PermutedDimsArray(B, perm)) ≈ batched_mul(fun(A), permutedims(B, perm))
        @test batched_mul(fun(PermutedDimsArray(A, perm)), B) ≈ batched_mul(fun(permutedims(A, perm)), B)
        # when TB=Float64, only the case  perm=(2,1,3); fun=batched_adjoint; ty=complex;  goes to fallback
        # but all the perm=(3,2,1); cases copy their inputs.
    end

    # PermutedDimsArray output
    A′ = randn(4,3,2)
    B′ = batched_adjoint(randn(TB, 5,3,2))
    C1 = batched_mul(A′, B′) # size 4,5,2
    C2 = PermutedDimsArray(zeros(5,2,4), (3,1,2)) # size 4,5,2
    @test C1 ≈ batched_mul!(C2, A′, B′) # Float64: "Debug: transposing C = A * B into Cᵀ = Bᵀ * Aᵀ"
    @test C1 ≈ C2

    # 5-arg mul!
    @test 10 .* C1 ≈ batched_mul!(C2, A′, B′, 10)
    C2 .= 10
    @test C1 .+ 100 ≈ batched_mul!(C2, A′, B′, 1, 10)

    # Trivial batches for B
    D′ = randn(TB, 3,5,1)
    @test size(batched_mul(A′,D′)) == (4,5,2)
    @test batched_mul(A′,D′) ≈ half_batched_mul(A′, D′)
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

@testset "storage_type" begin

    @test storage_type(transpose(reshape(view(rand(10), 2:9),4,:))) == Vector{Float64}
    @test storage_type(transpose(reshape(view(1:10,     2:9),4,:))) == UnitRange{Int}

    @test storage_typejoin(rand(2), rand(Float32, 2)) == Vector{<:Any}
    @test storage_typejoin(rand(2), rand(2,3)', rand(2,3,4)) == Array{Float64}
    @test storage_typejoin([1,2,3], 4:5) == AbstractVector{Int}

end

@testset "is_strided" begin

    M = ones(10,10)

    @test is_strided(M)
    @test is_strided(view(M, 1:2:5,:))
    @test is_strided(PermutedDimsArray(M, (2,1)))

    @test !is_strided(reshape(view(M, 1:2:10,:), 10,:))
    @test !is_strided((M.+im)')
    @test !is_strided(Diagonal(ones(3)))

    A = ones(2,2,2)

    @test is_strided(batched_adjoint(A))
    @test is_strided(batched_transpose(A))
    @test !is_strided(batched_adjoint(A .+ im))
    @test is_strided(batched_transpose(A .+ im))

    #=
    using SparseArrays
    @test !is_strided(sparse(M))
    using NamedDims
    @test is_strided(NamedDimsArray(M,(:a, :b))) # and 0.029 ns, 0 allocations
    =#

end
