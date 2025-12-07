using NNlib, Test, Statistics, Random
using ChainRulesCore, ChainRulesTestUtils
using Base.Broadcast: broadcasted
import EnzymeTestUtils
using EnzymeCore
import FiniteDifferences
import ForwardDiff
import Zygote
using Zygote: gradient
using StableRNGs
using Documenter
using Adapt
using ImageTransformations
using Interpolations: Constant
using KernelAbstractions
using FFTW
import ReverseDiff as RD        # used in `pooling.jl`
import Pkg
using SpecialFunctions

const Test_Enzyme = VERSION <= v"1.12-"

DocMeta.setdocmeta!(NNlib, :DocTestSetup, :(using NNlib, UnicodePlots); recursive=true)

# ENV["NNLIB_TEST_CUDA"] = "true" # uncomment to run CUDA tests
# ENV["NNLIB_TEST_AMDGPU"] = "true" # uncomment to run AMDGPU tests
# ENV["NNLIB_TEST_METAL"] = "true" # uncomment to run Metal tests
# ENV["NNLIB_TEST_CPU"] = "false" # uncomment to skip CPU tests

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

include("testsuite/gather.jl")
include("testsuite/scatter.jl")
include("testsuite/upsample.jl")
include("testsuite/rotation.jl")
include("testsuite/spectral.jl")
include("testsuite/fold.jl")

function nnlib_testsuite(Backend; skip_tests = Set{String}())
    @conditional_testset "Upsample" skip_tests begin
        upsample_testsuite(Backend)
    end
    @conditional_testset "rotation" skip_tests begin
        rotation_testsuite(Backend)
    end
    @conditional_testset "Gather" skip_tests begin
        gather_testsuite(Backend)
    end
    @conditional_testset "Scatter" skip_tests begin
        scatter_testsuite(Backend)
    end
    @conditional_testset "Spectral" skip_tests begin
        spectral_testsuite(Backend)
    end
    @conditional_testset "Fold" skip_tests begin
        fold_testsuite(Backend)
    end
end

@testset verbose=true "NNlib.jl" begin

    if get(ENV, "NNLIB_TEST_CPU", "true") == "true"
        @testset "CPU" begin
            @testset "Doctests" begin
                doctest(NNlib, manual=false)
            end

            nnlib_testsuite(CPU)

            if Threads.nthreads(:default) > 1
                @test NNlib.should_use_spawn()
                NNlib.@disallow_spawns begin
                    @test NNlib.should_use_spawn() == false
                end
            else
                @test NNlib.should_use_spawn() == false
            end

            @testset "Activation Functions" begin
                include("activations.jl")
                include("bias_act.jl")
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
    else
        @info "Skipping CPU tests, set NNLIB_TEST_CPU=true to run them."
    end

    if get(ENV, "NNLIB_TEST_CUDA", "false") == "true"
        Pkg.add(["CUDA", "cuDNN"])

        using CUDA
        if CUDA.functional()
            @testset "CUDA" begin
                nnlib_testsuite(CUDABackend; skip_tests=Set(("Scatter", "Gather")))

                include("ext_cuda/runtests.jl")
            end
        else
            @info "Insufficient version or CUDA not found; Skipping CUDA tests"
        end
    else
        @info "Skipping CUDA tests, set NNLIB_TEST_CUDA=true to run them"
    end

    if get(ENV, "NNLIB_TEST_AMDGPU", "false") == "true"
        Pkg.add("AMDGPU")

        using AMDGPU
        AMDGPU.versioninfo()
        if AMDGPU.functional() && AMDGPU.functional(:MIOpen)
            @testset "AMDGPU" begin
                nnlib_testsuite(ROCBackend)
                AMDGPU.synchronize(; blocking=false, stop_hostcalls=true)

                include("ext_amdgpu/runtests.jl")
                AMDGPU.synchronize(; blocking=false, stop_hostcalls=true)
            end
        else
            @info "AMDGPU.jl package is not functional. Skipping AMDGPU tests."
        end
    else
        @info "Skipping AMDGPU tests, set NNLIB_TEST_AMDGPU=true to run them."
    end

    if get(ENV, "NNLIB_TEST_METAL", "false") == "true"
        Pkg.add("Metal")

        using Metal
        if Metal.functional()
            @testset "Metal" begin
                # nnlib_testsuite(MetalBackend)
                include("ext_metal/runtests.jl")
            end
        else
            @info "Insufficient version or Metal not found; Skipping Metal tests"
        end
    else
        @info "Skipping Metal tests, set NNLIB_TEST_METAL=true to run them"
    end
end
