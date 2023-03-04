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
