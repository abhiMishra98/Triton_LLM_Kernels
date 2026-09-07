# GEMM

A tiled, Triton-JIT'd matrix multiplication kernel (`C = A @ B`), with masking so it stays correct on shapes that aren't multiples of the tile size.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)

## Background

`matmul_kernel` computes `C = A @ B` for `A: (M, K)` and `B: (K, N)` using a classic blocked-GEMM strategy: each program instance owns a `BLOCK_M x BLOCK_N` output tile, accumulates over `K` in `BLOCK_K`-sized steps in a float32 accumulator, and calls `tl.dot` per step. Boundary masks on the loads and store make the kernel correct even when `M`, `N`, or `K` aren't multiples of their respective block sizes, at the cost of some wasted work at the edges.

## Install

Requires a CUDA-capable GPU.

```bash
pip install torch triton
```

## Usage

Run the script directly to execute its built-in correctness check, which multiplies deliberately non-block-aligned matrices and compares against a PyTorch reference:

```bash
python GEMM.py
```

## API

### `matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor`

Multiplies `a` and `b` on the GPU and returns the result as a `float32` tensor. `a.shape[1]` must equal `b.shape[0]`. Internally launches `matmul_kernel` on a 2D grid of `(ceil(M / BLOCK_M), ceil(N / BLOCK_N))` program instances, with `BLOCK_M=64, BLOCK_N=64, BLOCK_K=32`.