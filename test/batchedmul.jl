using NNlib, Test, LinearAlgebra, Logging
using NNlib: storage_type, storage_typejoin, is_strided,
    batched_mul_generic!, _unbatch, _copy_if_faster,
    BatchedAdjoint, BatchedTranspose

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
    @test 10 .* C1 ≈ batched_mul!(C2, A′, B′, 10) rtol=1e-7
    C2 .= 10
    @test C1 .+ 100 ≈ batched_mul!(C2, A′, B′, 1, 10)

    # Trivial batches for B
    D′ = randn(TB, 3,5,1)
    @test size(batched_mul(A′,D′)) == (4,5,2)
    @test batched_mul(A′,D′) ≈ half_batched_mul(A′, D′)

    # Large output, multi-threaded path
    if TB == Float64
        N = 50
        A = rand(N,N,N)
        B = rand(N,N,N)
        C = reshape(reduce(hcat, [vec(A[:,:,k] * B[:,:,k]) for k in 1:N]), N,N,N)
        @test C ≈ A ⊠ B

        D = rand(N,N,1)
        E = reshape(reduce(hcat, [vec(A[:,:,k] * D[:,:,1]) for k in 1:N]), N,N,N)
        @test E ≈ A ⊠ D
    end
end

perm_12(A) = PermutedDimsArray(A, (2,1,3))
perm_23(A) = PermutedDimsArray(A, (1,3,2))

@testset "batched_mul: trivial dimensions & unit strides, $T" for T in [Float64, ComplexF64]
    @testset "$tA(rand$((sA...,3))) ⊠ $tB(rand$((sB...,3)))" for
    tA in [identity, batched_adjoint, batched_transpose, perm_12, perm_23], sA in [(1,1), (1,3), (3,1), (3,3)],
    tB in [identity, batched_adjoint, batched_transpose, perm_12, perm_23], sB in [(1,1), (1,3), (3,1), (3,3)]

        A = tA(rand(T, sA..., 3))
        B = tB(rand(T, sB..., 3))
        size(A,2) == size(B,1) && size(A,3) == size(B,3) == 3 || continue

        C = cat(A[:,:,1] * B[:,:,1], A[:,:,2] * B[:,:,2], A[:,:,3] * B[:,:,3]; dims=3)
        @test A ⊠ B ≈ C
        @test_logs min_level=Logging.Debug A ⊠ B

        # In-place batched_mul!
        α, β = rand(T), rand(T)
        D = rand(T, size(C))
        @test batched_mul!(copy(D), A, B, α, β) ≈ α .* C .+ β .* D
        @test batched_mul_generic!(copy(D), A, B, α, β) ≈ α .* C .+ β .* D

        # ... and with weird LHS -- all to batched_mul_generic! right now
        C2 = batched_transpose(permutedims(C, (2,1,3)))
        C3 = batched_adjoint(permutedims(conj(C), (2,1,3)))
        @test C2 == C3 == C
        C2 .= D
        C3 .= D
        @test batched_mul!(C2, A, B, α, β) ≈ α .* C .+ β .* D
        @test C2 ≈ α .* C .+ β .* D
        @test batched_mul!(C3, A, B, α, β) ≈ α .* C .+ β .* D
        @test C3 ≈ α .* C .+ β .* D
    end
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

@testset "batched_mul(ndims < 3), $TM" for TM in [ComplexF64, Int8]
    A = randn(ComplexF64, 3,3,3)
    M = rand(TM, 3,3) .+ im
    V = rand(TM, 3)

    # These are all reshaped and sent to batched_mul(3-array, 3-array)
    @test batched_mul(A, M) ≈ cat([A[:,:,k] * M for k in 1:3]...; dims=3)
    @test batched_mul(A, M') ≈ cat([A[:,:,k] * M' for k in 1:3]...; dims=3)
    @test A ⊠ transpose(M) ≈ cat([A[:,:,k] * transpose(M) for k in 1:3]...; dims=3)

    @test batched_mul(M, A) ≈ cat([M * A[:,:,k] for k in 1:3]...; dims=3)
    @test batched_mul(M', A) ≈ cat([M' * A[:,:,k] for k in 1:3]...; dims=3)
    @test transpose(M) ⊠ A ≈ cat([transpose(M) * A[:,:,k] for k in 1:3]...; dims=3)

    # batched_vec
    @test batched_vec(A, M) ≈ hcat([A[:,:,k] * M[:,k] for k in 1:3]...)
    @test batched_vec(A, M') ≈ hcat([A[:,:,k] * (M')[:,k] for k in 1:3]...)
    @test batched_vec(A, V) ≈ hcat([A[:,:,k] * V for k in 1:3]...)
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

end

FiniteDifferences.to_vec(x::BatchedAdjoint) = FiniteDifferences.to_vec(collect(x))
FiniteDifferences.to_vec(x::BatchedTranspose) = FiniteDifferences.to_vec(collect(x))

@testset "AutoDiff" begin
    M, P, Q = 13, 7, 11
    B = 3
    # Two 3-arrays
    gradtest(batched_mul, randn(rng, M, P, B), randn(rng, P, Q, B))
    gradtest(batched_mul, batched_adjoint(randn(rng, P, M, B)), randn(rng, P, Q, B))
    gradtest(batched_mul, randn(rng, M, P, B), batched_transpose(randn(rng, Q, P, B)))

    # One a matrix...
    gradtest(batched_mul, randn(rng, M, P), randn(rng, P, Q, B))
    gradtest(batched_mul, adjoint(randn(rng, P, M)), randn(rng, P, Q, B))
    gradtest(batched_mul, randn(rng, M, P), batched_adjoint(randn(rng, Q, P, B)))

    gradtest(batched_mul, randn(rng, M, P, B), randn(rng, P, Q))
    gradtest(batched_mul, batched_transpose(randn(rng, P, M, B)), randn(rng, P, Q))
    gradtest(batched_mul, randn(rng, M, P, B), transpose(randn(rng, Q, P)))

    # ... or equivalent to a matrix
    gradtest(batched_mul, randn(rng, M, P, 1), randn(rng, P, Q, B))
    gradtest(batched_mul, batched_transpose(randn(rng, P, M, 1)), randn(rng, P, Q, B))
    gradtest(batched_mul, randn(rng, M, P, 1), batched_transpose(randn(rng, Q, P, B)))

    gradtest(batched_mul, randn(rng, M, P, B), randn(rng, P, Q, 1))
    gradtest(batched_mul, batched_adjoint(randn(rng, P, M, B)), randn(rng, P, Q, 1))
    gradtest(batched_mul, randn(rng, M, P, B), batched_adjoint(randn(rng, Q, P, 1)))

    # batched_vec
    gradtest(batched_vec, randn(rng, M, P, B), randn(rng, P, B))
    gradtest(batched_vec, randn(rng, M, P, B), transpose(randn(rng, B, P)))

    gradtest(batched_vec, randn(rng, M, P, B), randn(rng, P))
end

@testset "batched_vec: N-D batches" begin
    # Test 4D case: A is 4D, B is 3D
    A4d = randn(4, 5, 3, 2)  # (matrix_rows, matrix_cols, batch_dim1, batch_dim2)
    B3d = randn(5, 3, 2)     # (vector_length, batch_dim1, batch_dim2)
    
    C = batched_vec(A4d, B3d)
    @test size(C) == (4, 3, 2)
    
    # Manual verification
    for i in 1:3, j in 1:2
        @test C[:, i, j] ≈ A4d[:, :, i, j] * B3d[:, i, j]
    end
    
    # Test 5D case: A is 5D, B is 4D
    A5d = randn(3, 4, 2, 3, 2)  # (matrix_rows, matrix_cols, batch1, batch2, batch3)
    B4d = randn(4, 2, 3, 2)     # (vector_length, batch1, batch2, batch3)
    
    C5 = batched_vec(A5d, B4d)
    @test size(C5) == (3, 2, 3, 2)
    
    # Manual verification for a few cases
    @test C5[:, 1, 1, 1] ≈ A5d[:, :, 1, 1, 1] * B4d[:, 1, 1, 1]
    @test C5[:, 2, 3, 2] ≈ A5d[:, :, 2, 3, 2] * B4d[:, 2, 3, 2]
    
    # Test dimension mismatch errors
    @test_throws DimensionMismatch batched_vec(randn(3, 4, 2), randn(4, 3))  # ndims mismatch
    @test_throws DimensionMismatch batched_vec(randn(3, 4, 2, 3), randn(4, 2, 2))  # batch size mismatch
    
end
