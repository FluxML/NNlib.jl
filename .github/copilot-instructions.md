# NNlib.jl Copilot Instructions

## Repository Overview

NNlib.jl is a library providing fundamental neural network operations and primitives for Julia. It is primarily used by Flux.jl but can be used independently. The library provides:

- Activation functions (sigmoid, relu, gelu, etc.)
- Convolution and pooling operations
- Attention mechanisms
- Batched matrix operations
- Neural network utilities (dropout, normalization, etc.)
- GPU acceleration support (CUDA, AMDGPU)

## Project Structure

```
NNlib.jl/
├── src/              # Core library implementation
│   ├── NNlib.jl      # Main module file
│   ├── activations.jl # Activation functions
│   ├── attention.jl   # Attention mechanisms
│   ├── conv.jl        # Convolution operations
│   ├── pooling.jl     # Pooling operations
│   ├── batched/       # Batched operations
│   └── impl/          # Implementation details
├── ext/              # Package extensions for GPU backends
│   ├── NNlibCUDAExt/      # CUDA-specific implementations
│   ├── NNlibAMDGPUExt/    # AMDGPU-specific implementations
│   └── NNlibCUDACUDNNExt/ # cuDNN-specific implementations
├── test/             # Test suite
└── docs/             # Documentation
```

## Julia Version

- Minimum Julia version: 1.10
- CI tests on: minimum julia version, latest stable (1.x), and pre-release versions

## Coding Standards

### Julia Conventions

1. **Naming**:
   - Functions: lowercase with underscores (e.g., `dot_product_attention`)
   - Types: PascalCase (e.g., `ConvDims`, `PoolDims`)
   - Constants: UPPERCASE with underscores (e.g., `ACTIVATIONS`)

2. **Documentation**:
   - Use Julia docstrings (""" ... """) for all exported functions
   - Include examples in docstrings where appropriate
   - Keep documentation up-to-date with implementation changes

3. **Type Annotations**:
   - Use type parameters and abstract types for generic implementations
   - Leverage Julia's multiple dispatch for specialized implementations
   - Define clear type hierarchies (e.g., `DenseConvDims`, `DepthwiseConvDims`)

4. **Performance**:
   - Prefer in-place operations where appropriate (functions ending with `!`)
   - Use `@inbounds` judiciously when bounds checking is verified
   - Consider thread safety for multi-threaded operations
   - Use `NNlib.@disallow_spawns` to control threading behavior

### Code Organization

1. **Core Implementations**: CPU implementations go in `src/`
2. **GPU Extensions**: GPU-specific code belongs in `ext/` as package extensions
3. **Tests**: Mirror the structure of `src/` in `test/`
4. **Gradients**: Define gradients using ChainRules.jl (`rrule` functions)

## Testing

### Test Infrastructure

- Uses the standard Julia `Test` framework
- Tests are organized to mirror the source structure
- GPU tests are conditional (controlled by environment variables)

### Running Tests

```julia
# Run all CPU tests
julia --project -e 'using Pkg; Pkg.test()'

# Run tests with threading
JULIA_NUM_THREADS=4 julia --project -e 'using Pkg; Pkg.test()'
```

### Test Patterns

1. **Activation Functions**: Test at specific values (0.0, 1.0, -1.0) and verify expected outputs
2. **Gradient Tests**: Use `ChainRulesTestUtils` for gradient correctness
3. **Type Stability**: Use `@inferred` where appropriate
4. **GPU Tests**: Conditional testing based on environment variables:
   - `ENV["NNLIB_TEST_CUDA"]` for CUDA tests
   - `ENV["NNLIB_TEST_AMDGPU"]` for AMDGPU tests

### Writing New Tests

- Include tests for edge cases (zero inputs, negative values, boundary conditions)
- Test both forward pass and gradients (using ChainRulesTestUtils)
- For array operations, test multiple dimensions and batch sizes
- Include tests for type stability when performance-critical

## Dependencies

### Core Dependencies

- **ChainRulesCore**: For automatic differentiation support
- **KernelAbstractions**: For GPU kernel abstractions
- **Adapt**: For moving data between CPU/GPU
- **GPUArraysCore**: GPU array interface

### Weak Dependencies (Extensions)

- **CUDA.jl/cuDNN**: NVIDIA GPU support
- **AMDGPU.jl**: AMD GPU support
- **FFTW**: Fast Fourier transforms
- **ForwardDiff**: Forward-mode AD support
- **EnzymeCore**: Enzyme AD support

### Adding New Dependencies

- Consider whether the dependency should be a weak dependency (extension)
- Update `Project.toml` with version constraints
- Ensure compatibility with supported Julia versions
- Run full test suite after adding dependencies

## GPU Support

NNlib uses Julia's package extension system for GPU backends:

1. **CUDA**: Load with `using NNlib, CUDA, cuDNN`
2. **AMDGPU**: Load with `using NNlib, AMDGPU`

### GPU Implementation Guidelines

- Keep GPU-specific code in appropriate extensions (`ext/` directory)
- Provide CPU fallback implementations in `src/`
- Test GPU implementations separately (conditional on hardware availability)
- Use KernelAbstractions for portable GPU kernels when possible

## Build and CI/CD

### Continuous Integration

- **CI Workflow**: `.github/workflows/ci.yml`
  - Tests on Linux (always), Windows, and macOS
  - Tests with different Julia versions (LTS, stable, pre-release)
  - Tests with different thread counts
  
### Additional Workflows

- **TagBot**: Automatic release tagging
- **CompatHelper**: Dependency compatibility updates
- **Downstream**: Tests dependent packages
- **BenchmarkTrigger**: Performance regression testing

## Common Tasks

### Adding a New Activation Function

1. Add function to `src/activations.jl`
2. Add to `ACTIVATIONS` tuple for automatic export
3. Define gradient with `@scalar_rule` or `rrule`
4. Add tests in `test/activations.jl` at key values
5. Document with docstring and example

### Adding a New Operation

1. Implement in appropriate file in `src/`
2. Export from `src/NNlib.jl`
3. Define gradients using ChainRules
4. Add comprehensive tests
5. Add GPU implementations in extensions if applicable
6. Document in appropriate file in `docs/src/`

### Modifying Existing Functions

1. Check for dependent code in Flux.jl and other downstream packages
2. Maintain backward compatibility or document breaking changes
3. Update tests to cover new behavior
4. Update gradients if needed
5. Consider performance implications

## Performance Considerations

1. **Memory Allocation**: Minimize allocations in hot paths
2. **Threading**: NNlib uses Julia threads for parallel operations
   - Control with `NNlib.@disallow_spawns` if needed
   - Thread count controlled by `JULIA_NUM_THREADS`
3. **GPU Kernels**: Optimize kernel launch parameters and memory access patterns
4. **Type Stability**: Ensure type-stable code for performance-critical paths

## Documentation

- Documentation source: `docs/src/`
- Built with Documenter.jl
- Includes API reference, examples, and guides
- Documentation tests run via `DocTestSetup`

## Related Projects

- **Flux.jl**: Primary consumer of NNlib
- **Zygote.jl**: Automatic differentiation (uses ChainRules)
- **ChainRules.jl**: Gradient definitions
- **KernelAbstractions.jl**: GPU kernel abstraction

## Getting Help

- Documentation: https://fluxml.ai/NNlib.jl/dev/
- Issues: https://github.com/FluxML/NNlib.jl/issues
- FluxML community: https://github.com/FluxML
