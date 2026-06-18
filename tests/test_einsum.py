import pytest
import torch
import os

TEST_DEVICE = os.getenv("EINSUM_TEST_DEVICE", "npu")

from pto_einsum import einsum, EinsumBuilder

# Test parameterized equations and shapes
@pytest.mark.parametrize("equation, shape0, shape1", [
    # standard matrix multiplication (aligned for Cube)
    ("ij, jk -> ik", (32, 64), (64, 128)),
    # outer product
    ("i, j -> ij", (64,), (96,)),
    # dot product (scalar contraction)
    ("i, i -> ", (256,), (256,)),
    # batch matrix multiplication (batch dim > 20 for core distribution)
    ("bij, bjk -> bik", (32, 16, 32), (32, 32, 64)),
    # tensor contraction
    #("ijk, jkl -> il", (32, 48, 96), (48, 96, 128)),
    # custom index layouts
    ("ai, ja -> ij", (32, 16), (64, 32)),
    ("abc, cd -> abd", (32, 16, 32), (32, 64)),
    # non-16-aligned contraction (K != C): exercises the fused kernel's K-padded
    # transpose destinations. j=20 -> K=32: identity-copy inputs (ij,jk) and
    # a=20 -> K=32: 2D-TTRANS inputs (ai,ja).
    ("ij, jk -> ik", (32, 20), (20, 128)),
    ("ai, ja -> ij", (20, 16), (64, 20)),
    # batched non-16-aligned contraction (K != C and I > 1): exercises the
    # per-batch K-row repack of ws1 (batched_pad_copy_inline). j=20 -> K=32.
    ("bij, bjk -> bik", (8, 16, 20), (8, 20, 64)),
    # multi-batch-axis contractions (two inplace axes, e.g. batch + heads): the
    # operands need an N-D non-identity transpose into the [batch, free, contract]
    # matmul layout. Exercises both N-D transpose paths -- innermost-
    # preserved strided gather (inputs/output whose inner axis is unchanged) and
    # the batched 2D TTRANS (an input whose inner axis moves) -- plus a non-
    # identity *output* permutation. d=64 keeps it exact in fp16 too.
    ("bshd, bthd -> bsht", (4, 16, 8, 64), (4, 16, 8, 64)),  # attention scores
    ("bsht, bthd -> bshd", (4, 16, 8, 16), (4, 16, 8, 64)),  # attention context
    ("bshd, hdk -> bshk", (4, 16, 8, 64), (8, 64, 64)),      # per-head rotation
    # Non-16-aligned contraction (K != C) on an N-D *transposed* input.
    # Part A: in `bhs` the contract `h` is not the innermost source axis, so input0
    # takes the batched-2D TTRANS path, whose innermost output axis IS the contract
    # dim -- now padded h=20 -> K=32. Single-batch (b=1, I=1) and multi-batch (I=b>1)
    # both exercise the K-padded transposed-row stride. (input1 `bht` stays identity.)
    ("bhs, bht -> bst", (1, 20, 16), (1, 20, 24)),  # input0 batched-2D K!=C, I=1
    ("bhs, bht -> bst", (4, 20, 16), (4, 20, 24)),  # input0 batched-2D K!=C, I>1
    # Part B: K != C with a *non-identity* input1 and batching (I = b*h > 1). input1's
    # matmul layout is [batch, C, L1]; the contract C is the row axis, so each batch's
    # C valid rows must land at the front of its K-row slot (transpose_inline_rowpad).
    # `bcht`: t (L1) stays innermost -> case (a) row-padded gather; `bhtc`: c (C) is
    # innermost so the inner axis moves -> case (b) batched-2D TTRANS with a K-padded
    # per-batch block. c=20 -> K=32. (input0 `bhsc` is identity + K!=C padded copy.)
    ("bhsc, bcht -> bsht", (2, 8, 16, 20), (2, 20, 8, 24)),  # input1 case (a), K!=C, I>1
    ("bhsc, bhtc -> bhst", (2, 8, 16, 20), (2, 8, 24, 20)),  # input1 case (b), K!=C, I>1
    # Large per-batch transpose tile (exceeds UB): the batched-2D TTRANS
    # block is tiled into 64x64 sub-blocks distributed across the vector lanes. The
    # `bhs` per-batch block is [h, s]; h=128/130, s=256 push it past the 184 KB UB so
    # it must block. h=128 -> K==C (no pad); h=130 -> K=144 exercises blocking *with*
    # the input0 innermost contract pad. The input1 case-(b) variant (c=130, t=256)
    # blocks the [t, c] tile *with* the input1 K-row pad. The matmul-facing free/
    # contract dims (s/t=256, h/c=128|130) keep the padded M/N/K tile-divisible (the
    # partial-tile constraint is a separate roadmap item). fp16 keeps tol at 1e-2.
    ("bhs, bht -> bst", (2, 128, 256), (2, 128, 64)),        # input0 blocked, K==C
    ("bhs, bht -> bst", (2, 130, 256), (2, 130, 64)),        # input0 blocked + pad
    ("bhsc, bhtc -> bhst", (2, 4, 16, 130), (2, 4, 256, 130)),  # input1 case (b) blocked + pad
    # elementwise (Hadamard) multiply: no contracted index, identical index order on
    # both inputs and the output. Routed to the dedicated elementwise kernel instead
    # of a batch of degenerate 1x1x1 matmuls. Sizes chosen to exercise the streaming
    # tail (1000 < one 2048 tile) and the multi-chunk / multi-lane path (3072 > tile).
    ("i, i -> i", (1000,), (1000,)),               # 1D, sub-tile tail
    ("ts, ts -> ts", (64, 48), (64, 48)),          # 2D, spans two chunks
    ("bshd, bshd -> bshd", (2, 16, 8, 64), (2, 16, 8, 64)),  # N-D Hadamard
    # non-16-aligned *free* dims (identity layout, no transpose): the padded M/N
    # exceed the real L0/L1, so the matmul emits partial output tiles whose
    # dynamic SetValidRow/SetValidCol extents differ across tiles (e.g. M tiles
    # 128 then 122). Exercises the dynamic-valid-extent path of the Cube matmul.
    ("ij, jk -> ik", (250, 128), (128, 256)),   # partial M
    ("ij, jk -> ik", (256, 128), (128, 250)),   # partial N
    ("ij, jk -> ik", (250, 128), (128, 250)),   # both
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_einsum_correctness(equation, shape0, shape1, dtype):
    if TEST_DEVICE == "cpu" and dtype == torch.float16:
        pytest.skip("float16 is not supported on CPU")
        
    # Generate random input tensors
    inp0 = torch.rand(shape0, dtype=dtype, device=TEST_DEVICE)
    inp1 = torch.rand(shape1, dtype=dtype, device=TEST_DEVICE)
    
    # Compute using our pre-compiled C++ einsum and torch einsum
    result = einsum(equation, inp0, inp1, device=TEST_DEVICE)
    expected = torch.einsum(equation, inp0, inp1).to(dtype=torch.float32)
    
    # Set assertion tolerance based on precision
    if dtype == torch.float32:
        rtol = 1e-4
        atol = 1e-4
    elif dtype == torch.float16:
        rtol = 1e-2
        atol = 1e-2
    else:
        rtol = 1e-7
        atol = 1e-7
    
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_linear_attention_chain(dtype):
    # Linear attention `einsum("tn,sn,sp,ts->tp", Q,K,V,L)` expressed as the three
    # two-way contractions it decomposes into. The middle step is the L-mask
    # Hadamard `ts,ts->ts`, which exercises the elementwise kernel; the outer two
    # are ordinary matmuls. Validates the whole chain against torch's 4-way einsum.
    if TEST_DEVICE == "cpu" and dtype == torch.float16:
        pytest.skip("float16 is not supported on CPU")

    T, S, N, P = 96, 96, 64, 128   # TS = 9216 >= the elementwise kernel threshold
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


def test_builder_reuse():
    # Verify that we can reuse the builder's return Callable to run multiple evaluations
    equation = "ij, jk -> ik"
    shape0 = (32, 64)
    shape1 = (64, 32)
    dtype = torch.float32
    
    runner = EinsumBuilder(equation, [shape0, shape1], dtype, device=TEST_DEVICE).build()
    
    for _ in range(3):
        inp0 = torch.rand(shape0, dtype=dtype, device=TEST_DEVICE)
        inp1 = torch.rand(shape1, dtype=dtype, device=TEST_DEVICE)
        
        result = runner(inp0, inp1)
        expected = torch.einsum(equation, inp0, inp1)
        
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_builder_reuse_kneqc():
    # The reused runner allocates its K-padded workspace once and zeros the
    # contraction-pad regions once. This exercises a K != C equation across many
    # calls with fresh data to ensure the pad stays zero (no stale carry-over) and
    # the persistent workspace yields correct results every time. (j=20 -> K=32.)
    equation = "bij, bjk -> bik"
    shape0, shape1 = (8, 16, 20), (8, 20, 64)
    runner = EinsumBuilder(equation, [shape0, shape1], torch.float32, device=TEST_DEVICE).build()

    for _ in range(10):
        inp0 = torch.rand(shape0, dtype=torch.float32, device=TEST_DEVICE)
        inp1 = torch.rand(shape1, dtype=torch.float32, device=TEST_DEVICE)
        result = runner(inp0, inp1)
        expected = torch.einsum(equation, inp0, inp1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


def test_validation_errors():
    equation = "ij, jk -> ik"
    shape0 = (32, 64)
    shape1 = (64, 32)
    
    # Mismatched operands length
    inp0 = torch.rand(shape0, dtype=torch.float32, device=TEST_DEVICE)
    inp1 = torch.rand(shape1, dtype=torch.float32, device=TEST_DEVICE)
    with pytest.raises(ValueError, match="Only exactly two operands are supported."):
        einsum(equation, inp0)
        
    # Mismatched datatypes between inputs
    inp1_mismatch = inp1.to(torch.int32)
    with pytest.raises(ValueError, match="Operands must have the same datatype."):
        einsum(equation, inp0, inp1_mismatch, device=TEST_DEVICE)

    # Builder validation checks for shapes
    runner = EinsumBuilder(equation, [shape0, shape1], torch.float32, device=TEST_DEVICE).build()
    
    bad_inp0 = torch.rand((32, 32), dtype=torch.float32, device=TEST_DEVICE)
    with pytest.raises(ValueError, match="do not match built shapes"):
        runner(bad_inp0, inp1)

    # Builder validation checks for datatypes
    with pytest.raises(TypeError, match="do not match built datatype"):
        runner(inp0, inp1_mismatch)


def test_cleanup():
    import os
    equation = "ij, jk -> ik"
    shape0 = (16, 32)
    shape1 = (32, 16)
    dtype = torch.float32
    
    builder = EinsumBuilder(equation, [shape0, shape1], dtype, device=TEST_DEVICE)
    runner = builder.build()
    
    # Verify the build files exist
    assert builder.target_dir is not None
    assert os.path.exists(builder.target_dir)
    assert os.path.exists(builder.cpp_filename)
    assert os.path.exists(builder.so_filename)
    
    # Run cleanup
    builder.cleanup()
    
    # Verify that they are deleted
    assert not os.path.exists(builder.target_dir)


if __name__ == "__main__":
    import sys
    # Programmatically execute pytest on this file when run directly
    sys.exit(pytest.main([__file__]))

