# Contributing to Baremetal AI Inference Engine

Thank you for your interest in contributing! This project aims to be the most accessible, lightweight, and hackable C99 inference engine for MoE models. 

## Project Philosophy

1. **Zero External Dependencies:** We rely only on standard C99 (`<stdio.h>`, `<stdlib.h>`, etc.), OpenMP for CPU threading, and standard POSIX/Win32 APIs. Do not introduce external libraries (e.g., `libcurl`, `OpenBLAS`, `zlib`) without extensive discussion.
2. **C99 Strictness:** Code must compile with `-std=c99`. Use MSVC or GCC/Clang.
3. **No Heap Allocations in Hot Paths:** The forward pass (`forward_runtime`) must execute with zero dynamic memory allocation (`malloc`/`free`). 
4. **Memory Constraint First:** This engine is designed for systems with <4 GB RAM. If a feature drastically increases baseline memory overhead, it belongs in an optional flag or macro.

## Versioning Policy

We use [Semantic Versioning (SemVer)](https://semver.org/):
- **MAJOR:** Incompatible API changes to the C wrapper (`baremetal.h`) or breaking changes to the CLI interface.
- **MINOR:** Backwards-compatible new features (e.g., new backend support, new quantization types, HTTP server endpoints).
- **PATCH:** Backwards-compatible bug fixes, optimizations, and internal refactors.

## Pull Request Process

1. **Format Code:** Use `clang-format` if available, or match the surrounding code style (4 spaces for indentation, brackets on the same line).
2. **Update Tests:** If you add a new kernel or quantization type, add a corresponding test in `tests/tests.c`.
3. **Run CI Locally:** Ensure `make test` passes before pushing.
4. **Review:** PRs will be reviewed for SIMD correctness, memory usage regressions, and C99 compliance.

## Reporting Bugs

Please include:
1. The model file used (or Hugging Face URL).
2. OS, CPU, and RAM specifications.
3. The exact command-line arguments used.
4. If it's a crash, a GDB backtrace if possible.
