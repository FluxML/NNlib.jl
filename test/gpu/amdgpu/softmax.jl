@testset "Compare CPU & GPU" begin
    for (T, atol) in ((Float16, 1f-2), (Float32, 1f-5))
        for (sz, dims) in [
            ((5,), :), ((5,), 1),
            ((5, 5), :), ((5, 5), 1), ((5, 5), 2),
            ((5, 5, 5, 5), (2, 3)), ((5, 5, 5, 5), (2, 4)),
        ]
            if T == Float16
                x = ones(T, sz) # Really low precision.
            else
                x = randn(T, sz)
            end
            gputest(NNlib.softmax, x; atol)
            gputest(NNlib.logsoftmax, x; atol)
        end
    end
end
