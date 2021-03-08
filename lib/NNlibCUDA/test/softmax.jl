@testset "softmax" begin
    for dims in [(5,5), (5,)]
        x = randn(Float64, dims)
        y = softmax(x)
        dy = randn(Float64, dims)
        gputest(softmax, x)
        gputest(∇softmax, dy, x, y, checkgrad=false)
        y = logsoftmax(x)
        gputest(logsoftmax, x)
        gputest(∇logsoftmax, dy, x, y, checkgrad=false) 
    end
end
