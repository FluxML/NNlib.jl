using NNlib, Test, Statistics, Random
using ChainRulesCore, ChainRulesTestUtils
using Base.Broadcast: broadcasted
import FiniteDifferences
import ForwardDiff
import Zygote
using Zygote: gradient
using StableRNGs
using Documenter
using Adapt
using KernelAbstractions
DocMeta.setdocmeta!(NNlib, :DocTestSetup, :(using NNlib, UnicodePlots); recursive=true)

ENV["NNLIB_TEST_CUDA"] = true # uncomment to run CUDA tests
# ENV["NNLIB_TEST_AMDGPU"] = true # uncomment to run AMDGPU tests

const rng = StableRNG(123)
include("test_utils.jl")

macro conditional_testset(name, skip_tests, expr)
    esc(quote
        @testset $name begin
            if $name ∉ $skip_tests
                $expr
            else
                @test_skip false
            end
        end
    end)
end

cpu(x) = adapt(CPU(), x)

include("gather.jl")
include("scatter.jl")
include("upsample.jl")

function nnlib_testsuite(Backend; skip_tests = Set{String}())
    @conditional_testset "Upsample" skip_tests begin
        upsample_testsuite(Backend)
    end
    @conditional_testset "Gather" skip_tests begin
        gather_testsuite(Backend)
    end
    @conditional_testset "Scatter" skip_tests begin
        scatter_testsuite(Backend)
    end
end

@testset "NNlib.jl" verbose=true begin
    @testset verbose=true "Test Suite" begin
        @testset "CPU" begin
            nnlib_testsuite(CPU)
        end

        if get(ENV, "NNLIB_TEST_CUDA", "false") == "true"
            using CUDA
            if CUDA.functional()
                @testset "CUDABackend" begin
                    nnlib_testsuite(CUDABackend; skip_tests=Set(("Scatter", "Gather")))
                end
            else
                @info "CUDA.jl is not functional. Skipping test suite for CUDABackend."
            end
        else
            @info "Skipping CUDA tests, set NNLIB_TEST_CUDA=true to run them."
        end

        if get(ENV, "NNLIB_TEST_AMDGPU", "false") == "true"
            import Pkg
            test_info = Pkg.project()
            # Add MIOpen_jll to AMDGPU.
            Pkg.develop("AMDGPU")
            Pkg.activate(joinpath(Pkg.devdir(), "AMDGPU"))
            Pkg.add("MIOpen_jll")
            Pkg.update()
            # Update test project.
            Pkg.activate(test_info.path)
            Pkg.update()

            using AMDGPU
            if AMDGPU.functional()
                @testset "ROCBackend" begin
                    nnlib_testsuite(ROCBackend)
                end
            else
                @info "AMDGPU.jl is not functional. Skipping test suite for ROCBackend."
            end
        else
            @info "Skipping AMDGPU tests, set NNLIB_TEST_AMDGPU=true to run them."
        end
    end

    @testset verbose=true "Tests" begin
        if get(ENV, "NNLIB_TEST_CUDA", "false") == "true"
            using CUDA
            if CUDA.functional()
                @testset "CUDA" begin
                    include("ext_cuda/runtests.jl")
                end
            else
                @info "Insufficient version or CUDA not found; Skipping CUDA tests"
            end
        else
            @info "Skipping CUDA tests, set NNLIB_TEST_CUDA=true to run them"
        end

        if get(ENV, "NNLIB_TEST_AMDGPU", "false") == "true"
            import Pkg
            test_info = Pkg.project()
            # Add MIOpen_jll to AMDGPU.
            Pkg.develop("AMDGPU")
            Pkg.activate(joinpath(Pkg.devdir(), "AMDGPU"))
            Pkg.add("MIOpen_jll")
            Pkg.update()
            # Update test project.
            Pkg.activate(test_info.path)
            Pkg.update()

            using AMDGPU
            AMDGPU.versioninfo()
            if AMDGPU.functional() && AMDGPU.functional(:MIOpen)
                @show AMDGPU.MIOpen.version()
                @testset "AMDGPU" begin
                    include("ext_amdgpu/runtests.jl")
                end
            else
                @info "AMDGPU.jl package is not functional. Skipping AMDGPU tests."
            end
        else
            @info "Skipping AMDGPU tests, set NNLIB_TEST_CUDA=true to run them."
        end

        @testset "Doctests" begin
            doctest(NNlib, manual=false)
        end

        @testset "Activation Functions" begin
            include("activations.jl")
        end

        @testset "Attention" begin
            include("attention.jl")
        end

        @testset "Batched Multiplication" begin
            include("batchedmul.jl")
        end

        @testset "Convolution" begin
            include("conv.jl")
            include("conv_bias_act.jl")
        end

        @testset "CTC Loss" begin
            include("ctc.jl")
        end

        @testset "Dropout" begin
            include("dropout.jl")
        end

        @testset "Fold/Unfold" begin
            include("fold.jl")
        end

        @testset "Inference" begin
            include("inference.jl")
        end

        @testset "Pooling" begin
            include("pooling.jl")
        end

        @testset "Padding" begin
            include("padding.jl")
        end

        @testset "Softmax" begin
            include("softmax.jl")
        end

        @testset "Utilities" begin
            include("utils.jl")
        end

        @testset "Grid Sampling" begin
            include("sampling.jl")
        end

        @testset "Functions" begin
            include("functions.jl")
        end
    end
end
