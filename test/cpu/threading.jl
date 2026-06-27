# Runs single-threaded in normal CI workers, and on multithreaded workers in the
# dedicated `NNLIB_TEST_THREADED` job (see `runtests.jl`), where the multi-thread
# branch below is exercised.
if Threads.nthreads(:default) > 1
    @test NNlib.should_use_spawn()
    NNlib.@disallow_spawns begin
        @test NNlib.should_use_spawn() == false
    end
else
    @test NNlib.should_use_spawn() == false
end
