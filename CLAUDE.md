# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Building the Project

```bash
# Setup requirements symlink (required once)
cd third_party/pip_requirements && ln -sf requirements-dev.txt requirements.txt

# Build all targets
bazel build //...

# Build specific CUDA targets
bazel build //d4ft/native/obara_saika:obara_saika

# Build with specific configuration
bazel build //... --config=test
bazel build //... --config=release
```

### Running Tests

```bash
# Run all tests
bazel test //...

# Run specific test
bazel test //tests/native/xla:eri_kernel_test

# Run with test output
bazel test --test_output=all //tests/native/xla:example_test
```

### Linting and Formatting

```bash
# Run all linters
make lint

# Format code
make format

# Python formatting
make py-format-fix

# C++ formatting
make clang-format-fix

# Bazel file formatting
make buildifier-fix
```

## High-Level Architecture

### Core Components

1. **d4ft/integral/** - Implementation of quantum chemistry integrals
   - `obara_saika/` - Obara-Saika scheme for integral computation
   - `gto/` - Gaussian-type orbital functions
   - `pw/` - Plane wave basis functions
   - `quadrature/` - Numerical quadrature methods

2. **d4ft/native/** - CUDA/C++ implementations for performance-critical code
   - `obara_saika/` - CUDA kernels for electron repulsion integrals
   - `gamma/` - Special functions (gamma, digamma, etc.)
   - `xla/` - XLA custom calls for JAX integration

3. **d4ft/hamiltonian/** - Hamiltonian construction
   - Nuclear, kinetic, and electron repulsion components
   - Orthogonalization routines

4. **d4ft/solver/** - SCF and optimization solvers
   - Direct minimization with gradient descent
   - Self-consistent field (SCF) iterations
   - PySCF wrapper for benchmarking

5. **d4ft/system/** - Molecular system definitions
   - Geometry readers (XYZ, cccdbd, refdata)
   - Basis set management
   - Occupation number handling

### Build System

The project uses Bazel with WORKSPACE-based dependency management. Key dependencies:
- `rules_cuda` - CUDA compilation support
- `rules_python` - Python rules and pip integration
- `pybind11` - Python/C++ bindings

### CUDA Configuration

The project uses `rules_cuda` from bazel-contrib. CUDA paths are configured in `.bazelrc`:
- CUDA_DIR and CUDA_PATH environment variables
- gcc-12 compiler for CUDA compatibility

### Python Environment

- Requires Python 3.9-3.11 (3.13+ has compatibility issues with older setuptools)
- Virtual environment setup recommended as per README
- JAX with CUDA support for GPU acceleration

## CUDA Issues Fixed (2025-06-15)

The following CUDA-related issues have been resolved:

1. **Updated rules_cuda**: Migrated from deprecated TensorFlow runtime location to official bazel-contrib/rules_cuda v0.2.1
2. **Fixed dependencies**: Added required bazel_skylib and rules_cc dependencies
3. **Updated GCC version**: Changed from gcc-12 to gcc-13 (available on system)
4. **Fixed Python version**: Updated to Python 3.11 for compatibility
5. **Migrated CUDA targets**: Updated all @cuda references to use @rules_cuda//cuda:runtime

## CUDA Issues Resolved ✅

All CUDA configuration issues have been successfully resolved:

1. **CUDA Path Detection**: Updated .bazelrc to point to correct CUDA installation at `/opt/cuda`
2. **rules_cuda Version**: Updated to v0.2.3 with correct checksum
3. **Double Precision Atomic Operations**: Added custom `atomicAdd` implementation for double precision on older GPU architectures
4. **Full Build Success**: All targets including CUDA kernels now build successfully

The project is now fully functional with CUDA support.

## Important Notes

1. When updating CUDA configuration, ensure:
   - `rules_cuda` version matches strip_prefix in workspace0.bzl
   - Python version is compatible (3.9-3.11)
   - Requirements symlink exists in third_party/pip_requirements/
   - GCC version matches what's available on system

2. The project uses both Obara-Saika and quadrature methods for integrals
   - CUDA kernels optimize the most expensive ERI calculations
   - XLA custom calls integrate with JAX's JIT compilation

3. Direct minimization (SGD) and SCF are two main solver approaches
   - Direct minimization works on the wavefunction directly
   - SCF iterates the Fock matrix to self-consistency