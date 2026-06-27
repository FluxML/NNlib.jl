# Tests for batched_mul on Metal, see https://github.com/FluxML/NNlib.jl/issues/581
using NNlib: batched_mul, batched_mul!, batched_vec, batched_adjoint, batched_transpose

@testset "batched_mul" begin
    A = randn(Float32, 3, 4, 5)
    B = randn(Float32, 4, 6, 5)

    # plain, and with batched transpose / adjoint wrappers on either side
    @test gputest(DEVICE, batched_mul, A, B; atol=1f-3)
    @test gputest(DEVICE, (a, b) -> batched_mul(batched_transpose(a), b),
                  randn(Float32, 4, 3, 5), B; atol=1f-3)
    @test gputest(DEVICE, (a, b) -> batched_mul(a, batched_adjoint(b)),
                  A, randn(Float32, 6, 4, 5); atol=1f-3)
    @test gputest(DEVICE, (a, b) -> batched_mul(batched_transpose(a), batched_adjoint(b)),
                  randn(Float32, 4, 3, 5), randn(Float32, 6, 4, 5); atol=1f-3)

    # PermutedDimsArray(_, (2,1,3)) is understood as a batched transpose
    @test gputest(DEVICE, (a, b) -> batched_mul(PermutedDimsArray(a, (2, 1, 3)), b),
                  randn(Float32, 4, 3, 5), B; atol=1f-3)

    # one operand lacking a batch index reshapes/broadcasts the batch dimension,
    # which MPS cannot do natively (falls back to the generic per-slice path)
    @test gputest(DEVICE, batched_mul, A, randn(Float32, 4, 6); atol=1f-3)
    @test gputest(DEVICE, batched_mul, randn(Float32, 7, 3), A; atol=1f-3)
    @test gputest(DEVICE, batched_mul, randn(Float32, 3, 4, 1), B; atol=1f-3)  # broadcast batch

    # regard B as a batch of vectors: A[:,:,k] * b[:,k]
    @test gputest(DEVICE, batched_vec, randn(Float32, 4, 5, 3), randn(Float32, 5, 3); atol=1f-3)

    # tiny arrays previously tripped MPS bugs (the original report)
    for n in (1, 2, 3)
        @test gputest(DEVICE, batched_mul, randn(Float32, n, n, 2), randn(Float32, n, n, 2); atol=1f-3)
    end

    # Float16 goes through the generic per-slice path
    @test gputest(DEVICE, batched_mul, randn(Float16, 8, 8, 4), randn(Float16, 8, 8, 4);
                  checkgrad=false)
end

@testset "batched_mul! (5-arg, in-place)" begin
    A = randn(Float32, 3, 4, 5)
    B = randn(Float32, 4, 6, 5)
    C = randn(Float32, 3, 6, 5)
    Ag, Bg, Cg = DEVICE(A), DEVICE(B), DEVICE(C)

    Cref = copy(C)
    batched_mul!(Cref, A, B, 2f0, 3f0)
    batched_mul!(Cg, Ag, Bg, 2f0, 3f0)
    @test Array(Cg) ≈ Cref atol=1f-3
end

@testset "BatchedAdjOrTrans display & conversion" begin
    x = rand(Float32, 3, 4, 2)
    y = DEVICE(x)

    # the array body, dropping the summary line whose eltype/container differs
    body(a) = @view split(sprint((io, v) -> show(io, MIME"text/plain"(), v), a), '\n')[2:end]

    for f in (batched_adjoint, batched_transpose)
        @test Array(f(y)) == Array(f(x))
        @test collect(f(y)) == collect(f(x))
        @test sprint(show, f(y)) == sprint(show, f(x))
        @test body(f(y)) == body(f(x))
        # reshape of a batched adjoint/transpose still avoids scalar indexing
        @test Array(reshape(f(y), 12, 2)) == Array(reshape(f(x), 12, 2))
        @test body(reshape(f(y), 12, 2)) == body(reshape(f(x), 12, 2))
    end
end
