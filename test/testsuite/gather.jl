using NNlib: gather, gather!
import EnzymeTestUtils
using EnzymeCore

function gather_testsuite(Backend)
    device(x) = adapt(Backend(), x)
    gradtest_fn = Backend == CPU ? gradtest : gputest
    T = Float32

    @testset "gather scalar index" begin
        ## 1d src, 2d index of ints -> 2d output
        src = device(T[3, 4, 5, 6, 7])
        index = device([
            1 2 3 4;
            4 2 1 3;
            3 5 5 3])
        output = T[
            3 4 5 6;
            6 4 3 5;
            5 7 7 5]

        y = cpu(gather(src, index))
        @test y isa Array{T,2}
        @test size(y) == size(index)
        @test y == output

        dst = device(T.(zero(index)))
        @test cpu(gather!(dst, src, index)) == output
        dst = device(zeros(T, 3, 5))
        @test_throws ArgumentError gather!(dst, src, index)

        if Backend == CPU
            index2 = [1 2 3 4;
                      4 2 1 3;
                      3 6 5 3]
            @test_throws BoundsError gather!(T.(zero(index)), src, index2)
        end

        ## 1d src, 3d index of ints -> 3d output
        src = device(T[3, 4, 5, 6, 7])
        index = device([
            1 2 3 4;
            4 2 1 3;
            3 5 5 3][:,:,1:1])
        output = T[
            3 4 5 6;
            6 4 3 5;
            5 7 7 5][:,:,1:1]

        y = cpu(gather(src, index))
        @test y isa Array{T,3}
        @test size(y) == size(index)
        @test y == output

        ## 2d src, 2d index of ints -> 3d output
        src = device(T[
            3 5 7
            4 6 8])
        index = device([
            1 2 3;
            2 2 1;
            3 1 3])

        output = zeros(T, 2, 3, 3)
        output[:,:,1] = [
            3 5 7
            4 6 8]
        output[:,:,2] = [
            5 5 3
            6 6 4]
        output[:,:,3] = [
            7 3 7
            8 4 8]

        y = cpu(gather(src, index))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa Array{T,3}
        @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...)
        @test y == output
    end

    @testset "gather tuple index" begin
        ## 2d src, 1d index of 2-tuples -> 1d output
        src = device(T[
            3 5 7
            4 6 8])
        index = device([(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)])
        output = T[3, 5, 7, 4, 6, 8]

        y = cpu(gather(src, index))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa Array{T,1}
        @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...)
        @test y == output

        ## 3d src, 2d index of 2-tuples -> 3d output
        n1, nsrc, nidx = 2, 3, 6
        src = device(rand(T, n1, nsrc, nsrc))
        index = device([
            (rand(1:nsrc), rand(1:nsrc)) for i=1:nidx, j=1:nidx])

        y = cpu(gather(src, index))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa Array{T,3}
        @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...)
    end

    @testset "gather cartesian index" begin
        ## 2d src, 1d index of 2-tuples -> 1d output
        src = device(T[
            3 5 7
            4 6 8])
        index = device(CartesianIndex.([(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]))
        output = T[3, 5, 7, 4, 6, 8]

        y = cpu(gather(src, index))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa Array{T,1}
        @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...)
        @test y == output

        ## 3d src, 2d index of 2-tuples -> 3d output
        n1, nsrc, nidx = 2, 3, 6
        src = device(rand(Float32, n1, nsrc, nsrc))
        index = device([
            CartesianIndex((rand(1:nsrc), rand(1:nsrc))) for i=1:nidx, j=1:nidx])

        y = cpu(gather(src, index))
        M = NNlib.typelength(eltype(index))
        Nsrc = ndims(src)
        @test y isa Array{T,3}
        @test size(y) == (size(src)[1:Nsrc-M]..., size(index)...)
    end

    @testset "gather gradient for scalar index" begin
        src = device(Float64[3, 4, 5, 6, 7])
        idx = device([
            1 2 3 4;
            4 2 1 3;
            3 5 5 3])
        dst = device(Float64[
            3 4 5 6;
            6 4 3 5;
            5 7 7 5])
        Backend == CPU ?
            gradtest_fn(xs -> gather!(dst, xs, idx), src) :
            gradtest_fn((d, s, i) -> gather!(d, s, i), dst, src, idx)
        Backend == CPU ?
            gradtest_fn(xs -> gather(xs, idx), src) :
            gradtest_fn((s, i) -> gather(s, i), src, idx)
    end

    @static if Test_Enzyme

    @testset "EnzymeRules: gather! gradient for scalar index" begin
        src = device(Float64[3, 4, 5, 6, 7])
        idx = device([
            1 2 3 4;
            4 2 1 3;
            3 5 5 3])
        dst = gather(src, idx)
        for Tret in (EnzymeCore.Const, EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
            Tdst in (EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated),
            Tsrc in (EnzymeCore.Duplicated, EnzymeCore.BatchDuplicated)

            Tret == EnzymeCore.Const && continue # ERROR
            EnzymeTestUtils.are_activities_compatible(Tret, Tdst, Tsrc) || continue

            EnzymeTestUtils.test_reverse(gather!, Tret, (dst, Tdst), (src, Tsrc), (idx, EnzymeCore.Const))
        end
    end

    end

    @testset "gather gradient for tuple index" begin
        src = device(Float64[
            3 5 7
            4 6 8])
        idx = device([(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)])
        dst = device(Float64[3, 5, 7, 4, 6, 8])
        Backend == CPU ?
            gradtest_fn(xs -> gather!(dst, xs, idx), src) :
            gradtest_fn((d, s, i) -> gather!(d, s, i), dst, src, idx)
        Backend == CPU ?
            gradtest_fn(xs -> gather(xs, idx), src) :
            gradtest_fn((s, i) -> gather(s, i), src, idx)
    end

    @testset "gather(src, IJK...)" begin
        x = device(reshape([1:15;], 3, 5))
        i, j = device([1,2]), device([2,4])
        y = gather(x, i, j)
        @test cpu(y) == [4, 11]
        y = gather(x, device([1, 2]))
        @test cpu(y) == [
            1 4
            2 5
            3 6]
    end
end

