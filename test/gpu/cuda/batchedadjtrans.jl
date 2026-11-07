function print_array_strs(x)
    str = sprint((io, x)->show(io, MIME"text/plain"(), x), x)
    return @view split(str, '\n')[2:end]
end

@testset "BatchedAdjOrTrans" begin
    x = randn(Float32, 3,4,2)
    y = cu(x)

    bax = batched_adjoint(x)
    btx = batched_transpose(x)
    bay = batched_adjoint(y)
    bty = batched_transpose(y)

    @test sprint(show, bax) == sprint(show, bay)
    @test sprint(show, btx) == sprint(show, bty)

    @test print_array_strs(bax) == print_array_strs(bay)
    @test print_array_strs(btx) == print_array_strs(bty)

    @test Array(bax) == Array(bay)
    @test collect(bax) == collect(bay)
    @test Array(btx) == Array(bty)
    @test collect(btx) == collect(bty)
    
    for shape in (:, (12, 2))
        rbax = reshape(bax, shape)
        rbtx = reshape(btx, shape)
        rbay = reshape(bay, shape)
        rbty = reshape(bty, shape)

        @test sprint(show, rbax) == sprint(show, rbay)
        @test sprint(show, rbtx) == sprint(show, rbty)
    
        @test print_array_strs(rbax) == print_array_strs(rbay)
        @test print_array_strs(rbtx) == print_array_strs(rbty)
    
        @test Array(rbax) == Array(rbay)
        @test collect(rbax) == collect(rbay)
        @test Array(rbtx) == Array(rbty)
        @test collect(rbtx) == collect(rbty)
    end

end

@testset "BatchedAdjOrTrans reshaped gradients" begin
    # Zygote adds the two cotangents of the transpose, each a reshaped `BatchedTranspose`.
    B, C = randn(Float32, 3, 4, 2, 2), randn(Float32, 3, 5, 4)
    f(B, C) = sum(batched_mul(batched_transpose(reshape(B, 3, 4, :)), C))
    g(B, C) = f(B, C) + f(B, C)
    @test Array(Zygote.gradient(g, cu(B), cu(C))[1]) ≈ Zygote.gradient(g, B, C)[1]
    @test reshape(batched_transpose(cu(randn(Float32, 3, 4, 2))), 4, :) .+ 1 isa CuArray
end
