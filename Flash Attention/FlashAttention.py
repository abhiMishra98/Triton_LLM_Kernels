import torch
import triton
import triton.language as tl
import torch.nn.functional as F

LOG2E = tl.constexpr(1.4426950408889634)

# ---------------------------------------------------------------------------
# Flash Attention forward kernel (SEGMENTs 1-7), non-causal or causal,
# single-head. Q, K, V are [seq_len, head_dim]; batch/head are handled by the
# grid, one program per BLOCK_M-sized slice of queries.
# ---------------------------------------------------------------------------


@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_om, stride_od,
    seq_len, head_dim,
    sm_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Single-head, single-sequence Flash Attention-2 forward pass.

    Online-softmax over K/V tiles of size BLOCK_N, one program per BLOCK_M
    query rows. Args mirror the tensor layout passed in from
    ``flash_attention``:

      stride_{q,k,v,o}{m,d}: row (seq) / col (head_dim) strides per tensor.
      NUM_STAGES: software-pipelining depth for the KV loop (``tl.range``).
      DOT_PRECISION: ``input_precision`` forwarded to ``tl.dot``.
      IS_CAUSAL: bounds and masks the KV loop to the causal triangle.
    """
    pid_m = tl.program_id(0)
    start_m = pid_m * BLOCK_M

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    fast_scale = sm_scale * LOG2E

    m_i = tl.full((BLOCK_M,), value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # SEGMENT 7: causal tile skip. A query in this program can only see keys
    # up to start_m + BLOCK_M - 1, so tiles entirely past that are never
    # entered -- this is the actual compute saving, not the in-loop mask
    # below, which only handles the diagonal tile.
    if IS_CAUSAL:
        kv_hi = tl.minimum(seq_len, start_m + BLOCK_M)
    else:
        kv_hi = seq_len

    for start_n in tl.range(0, kv_hi, BLOCK_N, num_stages=NUM_STAGES):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        qk = tl.dot(
            q,
            tl.trans(k),
            input_precision=DOT_PRECISION,
            out_dtype=tl.float32,
        ) * fast_scale

        # Padding mask, widened with the causal triangle on the diagonal
        # tile. On fully-past tiles (already excluded from the loop range
        # above) the triangle term would be all-True; it's only evaluated
        # for tiles this loop actually visits.
        valid = offs_n[None, :] < seq_len
        if IS_CAUSAL:
            valid = valid & (offs_m[:, None] >= offs_n[None, :])
        qk = tl.where(valid, qk, float("-inf"))

        # Online softmax update (base-2): rescale the running accumulator
        # to the new max instead of renormalizing after every tile.
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(
            p.to(v.dtype),
            v,
            input_precision=DOT_PRECISION,
            out_dtype=tl.float32,
        )

        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = O_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    o_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, acc.to(tl.float16), mask=o_mask)


def flash_attention(q, k, v, sm_scale=None, num_stages=2, dot_precision="ieee", is_causal=False):
    """Launch the forward kernel.

    Args:
        q, k, v: [seq_len, head_dim] float16 tensors on GPU.
        sm_scale: softmax scale; defaults to 1/sqrt(head_dim).
        num_stages: KV-loop pipelining depth, forwarded to ``tl.range``.
        dot_precision: ``input_precision`` forwarded to ``tl.dot``.
        is_causal: mask query i to keys <= i.

    Returns:
        [seq_len, head_dim] float16 tensor.
    """
    seq_len, head_dim = q.shape
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    o = torch.empty_like(q)

    BLOCK_M, BLOCK_N = 64, 64
    BLOCK_D = triton.next_power_of_2(head_dim)

    grid = (triton.cdiv(seq_len, BLOCK_M),)
    flash_attn_fwd_kernel[grid](
        q, k, v, o,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        seq_len, head_dim,
        sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        NUM_STAGES=num_stages, DOT_PRECISION=dot_precision,
        IS_CAUSAL=is_causal,
    )
    return o


def _naive_attention(q, k, v, is_causal):
    """Unfused fp32 reference attention, used to check kernel correctness."""
    seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim ** 0.5)
    scores = (q.float() @ k.float().T) * scale
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v.float()).to(torch.float16)


# ---------------------------------------------------------------------------
# SEGMENT 8: correctness check against the unfused reference, both masking
# modes. SEGMENT 9: latency vs. F.scaled_dot_product_attention.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    seq_len, head_dim = 256, 64

    q = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)

    for causal_flag in (False, True):
        out_triton = flash_attention(q, k, v, is_causal=causal_flag)
        out_ref = _naive_attention(q, k, v, is_causal=causal_flag)

        max_diff = (out_triton - out_ref).abs().max().item()
        label = "causal" if causal_flag else "non-causal"
        print(f"[{label}] max abs diff: {max_diff}")
        assert max_diff < 1e-2, f"Mismatch too large ({label})!"
    print("Correctness tests passed.")

    print("\nBenchmark (causal, fp16, head_dim=64):")
    print(f"{'seq_len':>8}  {'triton (ms)':>12}  {'sdpa (ms)':>10}  {'speedup':>8}")

    for bench_seq_len in (512, 1024, 2048, 4096):
        bq = torch.randn(bench_seq_len, head_dim, device="cuda", dtype=torch.float16)
        bk = torch.randn(bench_seq_len, head_dim, device="cuda", dtype=torch.float16)
        bv = torch.randn(bench_seq_len, head_dim, device="cuda", dtype=torch.float16)

        triton_ms = triton.testing.do_bench(
            lambda: flash_attention(bq, bk, bv, is_causal=True)
        )

        # SDPA expects a batch/head dim; unsqueeze to [1, 1, seq_len, head_dim]
        sq, sk, sv = (t.unsqueeze(0).unsqueeze(0) for t in (bq, bk, bv))
        sdpa_ms = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(sq, sk, sv, is_causal=True)
        )

        speedup = sdpa_ms / triton_ms
        print(f"{bench_seq_len:>8}  {triton_ms:>12.4f}  {sdpa_ms:>10.4f}  {speedup:>7.2f}x")