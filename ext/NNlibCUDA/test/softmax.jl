@testset "softmax" begin
    for (sz, dims) in [((5,), :), ((5,), 1), ((5,5), :), ((5,5), 1), ((5,5), 2)]
        x = randn(Float64, sz)
        y = softmax(x, dims=dims)
        dy = randn(Float64, sz)
        gputest(softmax, x, dims=dims)
        gputest(∇softmax, dy, x, y, dims=dims, checkgrad=false)
        y = logsoftmax(x, dims=dims)
        gputest(logsoftmax, x, dims=dims)
        gputest(∇logsoftmax, dy, x, y, dims=dims, checkgrad=false) 
    end
end
