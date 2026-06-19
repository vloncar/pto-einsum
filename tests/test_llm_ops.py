"""LLM-specific einsum patterns.

The contraction/broadcast shapes that show up in transformer workloads, kept
separate from the general-capability cases in test_einsum.py:

  * multi-batch-axis (N-D) attention contractions -- scores, context, per-head
    rotation, and the K!=C / blocked-transpose variants of the same machinery;
  * the linear-attention chain (a 4-way einsum run as three two-operand steps);
  * pure broadcast / scaling ops (rms_norm, token_pos, rope, alibi).
"""
import pytest
import torch
import os

TEST_DEVICE = os.getenv("EINSUM_TEST_DEVICE", "npu")

from pto_einsum import einsum, EinsumBuilder


def _tols(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-4
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 1e-7, 1e-7


# Multi-batch-axis contractions (two or more inplace axes, e.g. batch + heads): the
# operands need an N-D non-identity transpose into the [batch, free, contract] matmul
# layout. These exercise both N-D transpose paths -- the innermost-preserved strided
# gather (inputs/output whose inner axis is unchanged) and the batched 2D TTRANS (an
# input whose inner axis moves) -- plus a non-identity *output* permutation. d=64 keeps
# the matmul exact in fp16 too.
ATTENTION_CASES = [
    ("bshd, bthd -> bsht", (4, 16, 8, 64), (4, 16, 8, 64)),  # attention scores
    ("bsht, bthd -> bshd", (4, 16, 8, 16), (4, 16, 8, 64)),  # attention context
    ("bshd, hdk -> bshk", (4, 16, 8, 64), (8, 64, 64)),      # per-head rotation
    # Non-16-aligned contraction (K != C) on an N-D *transposed* input.
    # In `bhs` the contract `h` is not the innermost source axis, so input0 takes the
    # batched-2D TTRANS path, whose innermost output axis IS the contract dim -- now
    # padded h=20 -> K=32. Single-batch (b=1, I=1) and multi-batch (I=b>1) both exercise
    # the K-padded transposed-row stride. (input1 `bht` stays identity.)
    ("bhs, bht -> bst", (1, 20, 16), (1, 20, 24)),  # input0 batched-2D K!=C, I=1
    ("bhs, bht -> bst", (4, 20, 16), (4, 20, 24)),  # input0 batched-2D K!=C, I>1
    # K != C with a *non-identity* input1 and batching (I = b*h > 1). input1's matmul
    # layout is [batch, C, L1]; the contract C is the row axis, so each batch's C valid
    # rows must land at the front of its K-row slot (transpose_inline_rowpad). `bcht`: t
    # (L1) stays innermost -> case (a) row-padded gather; `bhtc`: c (C) is innermost so
    # the inner axis moves -> case (b) batched-2D TTRANS with a K-padded per-batch block.
    # c=20 -> K=32. (input0 `bhsc` is identity + K!=C padded copy.)
    ("bhsc, bcht -> bsht", (2, 8, 16, 20), (2, 20, 8, 24)),  # input1 case (a), K!=C, I>1
    ("bhsc, bhtc -> bhst", (2, 8, 16, 20), (2, 8, 24, 20)),  # input1 case (b), K!=C, I>1
    # Large per-batch transpose tile (exceeds UB): the batched-2D TTRANS block is tiled
    # into 64x64 sub-blocks distributed across the vector lanes. The `bhs` per-batch
    # block is [h, s]; h=128/130, s=256 push it past the 184 KB UB so it must block.
    # h=128 -> K==C (no pad); h=130 -> K=144 exercises blocking *with* the input0
    # innermost contract pad. The input1 case-(b) variant (c=130, t=256) blocks the
    # [t, c] tile *with* the input1 K-row pad. fp16 keeps tol at 1e-2.
    ("bhs, bht -> bst", (2, 128, 256), (2, 128, 64)),        # input0 blocked, K==C
    ("bhs, bht -> bst", (2, 130, 256), (2, 130, 64)),        # input0 blocked + pad
    ("bhsc, bhtc -> bhst", (2, 4, 16, 130), (2, 4, 256, 130)),  # input1 case (b) blocked + pad
]


@pytest.mark.parametrize("equation, shape0, shape1", ATTENTION_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_attention_contraction(equation, shape0, shape1, dtype):
    if TEST_DEVICE == "cpu" and dtype == torch.float16:
        pytest.skip("float16 is not supported on CPU")

    inp0 = torch.rand(shape0, dtype=dtype, device=TEST_DEVICE)
    inp1 = torch.rand(shape1, dtype=dtype, device=TEST_DEVICE)

    result = einsum(equation, inp0, inp1, device=TEST_DEVICE)
    expected = torch.einsum(equation, inp0, inp1).to(dtype=torch.float32)

    rtol, atol = _tols(dtype)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("T, S, N, P", [
    (96, 96, 64, 128),     # all free dims single-tile / aligned (baseline)
    (320, 320, 64, 128),   # T=S=320 -> partial M/N grid (operand-padded to 384)
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linear_attention_chain(dtype, T, S, N, P):
    # Linear attention `einsum("tn,sn,sp,ts->tp", Q,K,V,L)` expressed as the three
    # two-way contractions it decomposes into. The middle step is the L-mask
    # Hadamard `ts,ts->ts`, which exercises the elementwise kernel; the outer two
    # are ordinary matmuls. Validates the whole chain against torch's 4-way einsum.
    # The 320 case drives the two matmuls with a non-128-multiple seq length (a
    # partial output-tile grid), end-to-end alongside the Hadamard L-mask.
    if TEST_DEVICE == "cpu" and dtype == torch.float16:
        pytest.skip("float16 is not supported on CPU")

    Q = torch.rand(T, N, dtype=dtype, device=TEST_DEVICE)
    K = torch.rand(S, N, dtype=dtype, device=TEST_DEVICE)
    V = torch.rand(S, P, dtype=dtype, device=TEST_DEVICE)
    L = torch.tril(torch.ones(T, S, dtype=dtype, device=TEST_DEVICE))

    A = einsum("tn, sn -> ts", Q, K, device=TEST_DEVICE).to(dtype)
    A = einsum("ts, ts -> ts", A, L, device=TEST_DEVICE).to(dtype)   # elementwise step
    O = einsum("ts, sp -> tp", A, V, device=TEST_DEVICE)

    ref = torch.einsum("tn,sn,sp,ts->tp", Q.float(), K.float(), V.float(), L.float())
    rtol, atol = (1e-3, 1e-3) if dtype == torch.float32 else (5e-2, 5e-2)
    torch.testing.assert_close(O.float(), ref, rtol=rtol, atol=atol)


# Broadcast / scaling ops: no contracted index, one operand carries every output
# axis (the "full" operand, in output order) and the other a strict subset that is
# broadcast over the axes it lacks. Routed to the Vector broadcast-multiply kernel.
# These cover both kernel modes: mode 0 (ColExpand, B varies along the inner cols:
# rms_norm / token_pos / rope) and mode 1 (scalar, B constant per group: alibi).
BROADCAST_CASES = [
    ("bsd, d -> bsd",     (2, 8, 64),    (64,)),       # rms_norm-like  (mode 0, Cc=64,  single group)
    ("bsd, sd -> bsd",    (2, 8, 64),    (8, 64)),      # token_pos      (mode 0, Cc=512, broadcast over b)
    ("bshd, sd -> bshd",  (2, 8, 4, 16), (8, 16)),      # rope-like      (mode 0, interior-strided broadcast)
    ("hqk, h -> hqk",     (8, 16, 16),   (8,)),         # alibi          (mode 1, scalar per outer row)
    # Larger configs that spread rows across multiple Vector cores with a partial
    # trailing block (rb < RB): the GM row extent must be the runtime rb, else the
    # last lane over-reads past the operand buffer (was NaN before the runtime-dim fix).
    ("bsd, d -> bsd",     (2, 16, 512),  (512,)),       # rms_norm Cc=512, Rr=32 over ~8 lanes
    ("bshd, sd -> bshd",  (4, 32, 8, 64),(32, 64)),     # rope Cc=64, Outer=128, multi-lane
]


@pytest.mark.parametrize("equation, shape0, shape1", BROADCAST_CASES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_broadcast_correctness(equation, shape0, shape1, dtype):
    # The broadcast classifier is NPU-only (it returns None on CPU), so these route
    # to the Vector broadcast kernel only on device.
    if TEST_DEVICE != "npu":
        pytest.skip("broadcast/scaling ops are routed only on NPU")

    inp0 = torch.rand(shape0, dtype=dtype, device=TEST_DEVICE)
    inp1 = torch.rand(shape1, dtype=dtype, device=TEST_DEVICE)

    result = einsum(equation, inp0, inp1, device=TEST_DEVICE)
    expected = torch.einsum(equation.replace(" ", ""), inp0.float(), inp1.float())

    rtol, atol = (1e-3, 1e-3) if dtype == torch.float32 else (5e-2, 5e-2)
    torch.testing.assert_close(result.float(), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_broadcast_multi_op_sequence(dtype):
    # Regression guard for the single-core store-vs-readback race: distinct broadcast
    # kernels built and run back-to-back (with cleanup between) used to leave the last
    # block's GM store in flight, so a small single-core op could read back stale data
    # from a reused buffer. Same seed/shape across ops makes a stale readback look like
    # "B not applied". Drained by a kernel-exit pipe_barrier; this exercises that path.
    if TEST_DEVICE != "npu":
        pytest.skip("broadcast/scaling ops are routed only on NPU")

    rtol, atol = (1e-3, 1e-3) if dtype == torch.float32 else (5e-2, 5e-2)
    for name, (equation, shape0, shape1) in enumerate(BROADCAST_CASES):
        torch.manual_seed(0)
        inp0 = torch.rand(shape0, dtype=dtype, device=TEST_DEVICE)
        inp1 = torch.rand(shape1, dtype=dtype, device=TEST_DEVICE)
        builder = EinsumBuilder(equation, [shape0, shape1], dtype, device=TEST_DEVICE)
        result = builder.build()(inp0, inp1)
        expected = torch.einsum(equation.replace(" ", ""), inp0.float(), inp1.float())
        torch.testing.assert_close(result.float(), expected, rtol=rtol, atol=atol)
        builder.cleanup()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
