using NNlib
using NNlib: σ, swish, softplus, logσ, mish, logcosh, tanh_fast, sigmoid_fast,
             lisht, tanhshrink

# Holomorphic activations that accept complex inputs (see `test/activations.jl`).
const COMPLEX_ACTS = [σ, swish, softplus, logσ, mish, logcosh,
                      tanh_fast, sigmoid_fast, lisht, tanhshrink]

function activations_testsuite(Backend)
    cpu(x) = adapt(CPU(), x)
    device(x) = adapt(Backend(), x)

    # A spread of points in ℂ: both signs of the real part exercise the
    # overflow-safe branches in σ/softplus, and nonzero imaginary parts make
    # sure the gradient `conj` convention is correct on the device.
    zs = ComplexF32[0.4f0 + 0.6f0im, -0.7f0 + 0.2f0im,
                    1.2f0 - 0.9f0im, -0.3f0 - 1.1f0im]

    @testset "complex $f" for f in COMPLEX_ACTS
        x = device(zs)

        # Forward: device result must match the CPU result elementwise.
        y = f.(x)
        @test y isa AbstractArray{ComplexF32}
        @test cpu(y) ≈ f.(zs)

        # Gradient: take a real loss (`abs2`) so Zygote is happy with the
        # complex output, then compare the device gradient against the CPU one.
        loss(x) = sum(abs2, f.(x))
        g = gradient(loss, x)[1]
        g_cpu = gradient(loss, zs)[1]
        @test g isa AbstractArray{ComplexF32}
        @test cpu(g) ≈ g_cpu
    end

    # The numerically-stable branches must stay finite on the device where the
    # naive formula would overflow.
    @testset "numerical stability" begin
        big = device(ComplexF32[600 + 2im, -600 + 2im, 80 - 50im, -80 + 50im])
        for f in (σ, softplus, swish)
            @test all(isfinite, cpu(f.(big)))
        end
    end
end
