# Triton-LLM-Kernels

Hand-written [Triton](https://github.com/openai/triton) GPU kernels for common LLM building blocks, built for learning and benchmarking against PyTorch's native ops.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Kernels](#kernels)

## Background

This repo collects standalone Triton kernel implementations of operations that show up in LLM training and inference (matmul, attention, normalization, etc.), one directory per kernel. Each kernel is self-contained and ships with a correctness check against the equivalent PyTorch/`torch.compile` implementation, so it can be read, run, and benchmarked in isolation.

## Install

Requires a CUDA-capable GPU.

```bash
pip install torch triton
```

## Usage

Each kernel lives in its own directory and can be run directly:

```bash
python GEMM/GEMM.py
```

Running a kernel's script executes its `__main__` correctness check, printing the max absolute error against a PyTorch reference implementation.

## Kernels

| Kernel | Description |
| --- | --- |
| [GEMM](GEMM/) | Tiled matrix multiplication (`C = A @ B`) with masked boundary handling for non-block-aligned shapes. |