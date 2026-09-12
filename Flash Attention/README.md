# FlashAttention

A tiled, Triton-JIT'd FlashAttention-2 forward pass for a single sequence and head, using an online softmax so the full `seq_len x seq_len` score matrix is never materialized.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Optimizations](#optimizations)
- [Benchmarks](#benchmarks)

## Background

`flash_attn_fwd_kernel` computes attention for `Q, K, V: (seq_len, head_dim)` using the FlashAttention-2 strategy. Each program instance owns a `BLOCK_M`-sized tile of queries and streams `K`/`V` in `BLOCK_N`-sized tiles, keeping a running max (`m_i`) and running denominator (`l_i`) per query row. The output accumulator is rescaled whenever the running max updates, and division by `l_i` is deferred to the end. This avoids storing the full attention matrix while still producing an exact softmax. Padding masks on the loads and store keep the kernel correct when `seq_len` is not a multiple of the block sizes. An optional causal mode restricts each query tile to keys at or before it.

## Install

Requires a CUDA-capable GPU.

```bash
pip install torch triton
```

## Usage

Run the script directly to execute its correctness checks (non-causal and causal, against an unfused PyTorch reference) followed by a latency benchmark against `F.scaled_dot_product_attention`:

```bash
source ~/venvs/cute/bin/activate
python FlashAttention.py
```

For occupancy/warp-stall numbers, profile a single forward pass with:

```bash
ncu --set full -o flash_attn_profile python FlashAttention.py --profile-once
```

## API

### `flash_attention(q, k, v, sm_scale=None, num_stages=2, dot_precision="ieee", is_causal=False) -> torch.Tensor`

Runs the forward kernel and returns a `(seq_len, head_dim)` float16 tensor.

- `q, k, v`: `(seq_len, head_dim)` float16 tensors on GPU.
- `sm_scale`: softmax scale; defaults to `1/sqrt(head_dim)`.
- `num_stages`: KV-loop software-pipelining depth, forwarded to `tl.range`.
- `dot_precision`: `input_precision` forwarded to `tl.dot`.
- `is_causal`: if `True`, query `i` only attends to keys `<= i`.

Internally launches `flash_attn_fwd_kernel` on a 1D grid of `ceil(seq_len / BLOCK_M)` program instances, with `BLOCK_M=64, BLOCK_N=64, BLOCK_D=next_power_of_2(head_dim)`.

## Optimizations

Beyond the baseline online-softmax loop, four optimizations are applied:

1. **`exp2` instead of `exp`** — the softmax scale is pre-multiplied by `log2(e)` so `tl.math.exp2` can replace `tl.exp`, cheaper on most GPU backends and mathematically identical.
2. **Explicit software pipelining** — the K/V loop uses `tl.range(..., num_stages=NUM_STAGES)` instead of plain `range`, letting the compiler overlap the load for iteration *i+1* with compute for iteration *i*.
3. **Explicit dot precision** — both `tl.dot` calls take `input_precision=DOT_PRECISION` and `out_dtype=tl.float32` instead of relying on Triton's defaults.
4. **Causal tile skipping** — a causal query tile starting at `start_m` can never see keys past `start_m + BLOCK_M - 1`, so the K/V loop bound is clamped to `kv_hi = min(seq_len, start_m + BLOCK_M)`: tiles entirely in the future are never loaded or computed. Only the diagonal tile needs the extra element-wise triangular mask (`offs_m >= offs_n`).

## Benchmarks

**Correctness** (`seq_len=256, head_dim=64`, vs. an unfused fp32 PyTorch reference): max abs diff `~2e-4` (non-causal) / `~2e-3` (causal), both under the `1e-2` tolerance used for fp16.

**Latency vs. SDPA** (causal, fp16, `head_dim=64`, single head, on an NVIDIA L40S):

| seq_len | triton (ms) | sdpa (ms) | speedup |
|--------:|------------:|----------:|--------:|
|     512 |      0.0128 |    0.0116 |   0.91x |
|    1024 |      0.0220 |    0.0131 |   0.60x |
|    2048 |      0.0391 |    0.0193 |   0.49x |
|    4096 |      0.0734 |    0.0305 |   0.42x |

SDPA (cuDNN/CUTLASS-backed, warp-specialized) pulls further ahead as `seq_len` grows, as expected for a fixed `BLOCK_M = BLOCK_N = 64` reference kernel compared to a production-tuned one.

### Why the causal tile skip doesn't show up at these sizes

Comparing causal vs. non-causal at the same `seq_len` (instead of vs. SDPA) exposes a load-balancing issue:

| seq_len | grid (programs) | causal (ms) | non-causal (ms) | ratio |
|--------:|-----------------:|------------:|-----------------:|------:|
|    4096 |               64 |      0.0737 |           0.0737 |  1.00 |
|    8192 |              128 |      0.1397 |           0.1424 |  0.98 |
|   16384 |              256 |      0.2988 |           0.3827 |  0.78 |
|   32768 |              512 |      1.0257 |           1.5808 |  0.65 |

The tile-skip logic is correct — it halves the average number of K/V tiles visited — but the grid launches one CTA per `BLOCK_M`-sized query tile with no work redistribution. When the grid fits in a single wave of the GPU's SMs (L40S has 142 SMs; `seq_len=4096` gives 64 programs), wall-clock time is set by the slowest program in that wave. For causal attention that is the last query tile, which still walks almost the entire K/V range, same as non-causal. Savings from the earlier, cheaper tiles are hidden by that straggler. Only once the grid exceeds one wave (`seq_len >= 16384`) does the FLOP reduction start cutting the number of waves needed, and the ratio drops toward the expected ~0.5.

Production FlashAttention kernels avoid this by interleaving/swizzling query-tile assignment across CTAs, or splitting each tile's K/V range across multiple CTAs and reducing afterward (as in flash-decoding), so every wave is balanced regardless of grid size. Not implemented here.

### Possible follow-ups

- Load-balanced causal grid (tile interleaving or split-K reduction) so the tile-skip savings show up in wall-clock time at moderate `seq_len`.
- Batch/head dimensions folded into the grid instead of one sequence at a time.
- Backward pass (would also need the logsumexp saved out of the forward pass).
