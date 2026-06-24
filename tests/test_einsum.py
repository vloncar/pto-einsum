"""Base einsum suite: general end-to-end correctness, aiming to exercise every
kernel routing/tiling branch. The cases are grouped by the path they hit:
identity-layout matmul, transposed inputs (2D / blocked TTRANS), non-identity
output (Phase C), batched (I>1), non-16-aligned contraction (K!=C), elementwise
Hadamard, degenerate free dims (outer/dot/matrix-vector), thin+large-K split-K,
and partial output tiles / tile grids -- plus builder reuse, input validation,
cleanup, and the known-unsupported cases (skipped, with their loud-failure
contract pinned). LLM-shaped contractions (multi-batch-axis attention,
linear-attention chain, broadcast/scaling) live in test_llm_ops.py; the standalone
transpose and matmul kernels are covered by test_transpose.py and test_matmul.py.
"""
import pytest
import torch
import os

TEST_DEVICE = os.getenv("EINSUM_TEST_DEVICE", "npu")

from pto_einsum import einsum, EinsumBuilder

# Test parameterized equations and shapes. The cases are grouped by the kernel
# code path they exercise; together they aim to cover every routing/tiling branch.
@pytest.mark.parametrize("equation, shape0, shape1", [
    # --- identity-layout matmul (no input transpose, identity output) ---
    # standard matrix multiplication (aligned for Cube)
    ("ij, jk -> ik", (32, 64), (64, 128)),
    # multi-tile output grid (both free dims span several 128 tiles)
    ("ij, jk -> ik", (256, 256), (256, 384)),
    # outer product (C == 1: degenerate contraction, rank-1 update)
    ("i, j -> ij", (64,), (96,)),
    # dot product (L0 == L1 == 1, scalar output; thin+large-K -> split-K path)
    ("i, i -> ", (256,), (256,)),
    # matrix-vector / vector-matrix (one free dim is degenerate, L1 or L0 == 1)
    ("ij, j -> i", (32, 64), (64,)),               # L1 == 1
    ("i, ij -> j", (32,), (32, 64)),               # L0 == 1
    # thin matmul with a large contraction -> split-K over otherwise-idle cores
    ("ij, jk -> ik", (16, 2048), (2048, 16)),
    # multi-axis contraction: both j and k are contracted (C = j*k = 512, K == C)
    ("ijk, jkl -> il", (8, 16, 32), (16, 32, 64)),

    # --- transposed inputs (2D TTRANS into the [free, contract] layout) ---
    ("ji, jk -> ik", (64, 32), (64, 128)),         # input0 transposed
    ("ij, kj -> ik", (32, 64), (128, 64)),         # input1 transposed
    ("ji, kj -> ik", (64, 32), (128, 64)),         # both inputs transposed
    # large transposed input that exceeds the UB -> blocked 2D transpose (and a
    # partial M grid from i=500), distributed across all Vector lanes.
    ("ji, jk -> ik", (512, 500), (512, 64)),

    # --- non-identity output permutation (Phase C transpose runs) ---
    ("ij, jk -> ki", (32, 64), (64, 48)),          # output transposed (free1 not innermost
                                                   # -> non-fusible, keeps Phase C)
    # custom index layouts (transposed inputs and/or output)
    ("ai, ja -> ij", (32, 16), (64, 32)),
    ("abc, cd -> abd", (32, 16, 32), (32, 64)),    # multi-axis free0 -> non-fusible

    # --- fused output permutation (free0/free1 single axes, free1 innermost in res) ---
    # The Cube store lands each tile straight into res, dropping Phase C. The dominant
    # attention contractions: batched over (b,h), so the store row stride is H*T (!= L1).
    ("bshd, bthd -> bsht", (2, 128, 4, 64), (2, 128, 4, 64)),   # ->bsht, K==C
    ("bshd, bthd -> bsht", (2, 64, 4, 32), (2, 96, 4, 32)),     # asymmetric S!=T, sub-tile
    ("bsht, bthd -> bshd", (2, 128, 4, 128), (2, 128, 4, 64)),  # ->bshd (contract t)
    ("shd, thd -> sht", (200, 1, 64), (128, 1, 64)),            # single-batch partial-M tile

    # --- batched matmul (I > 1) ---
    # batch dim > 20 so the per-batch tiles distribute across cores
    ("bij, bjk -> bik", (32, 16, 32), (32, 32, 64)),
    # batched with transposed per-batch input block (batched 2D transpose), I > 20
    ("bji, bjk -> bik", (24, 32, 16), (24, 32, 64)),
    # batched-tiny 128^3: full tiles, nKd==1, total_tiles(=64) >= MIN_TILES -> the
    # cross-tile pipelined loop (matmul_tile_loop_pipelined). Identity output.
    ("bij, bjk -> bik", (64, 128, 128), (64, 128, 128)),
    # same regime with a fused output permutation: batch (b,h)=2*16=32 tiles, free0/free1
    # full 128 tiles, contract d=64 -> nKd==1 -> pipelined loop with FUSE_OUT.
    ("bshd, bthd -> bsht", (2, 128, 16, 64), (2, 128, 16, 64)),

    # --- non-16-aligned contraction (K != C): K-padded transpose destinations ---
    # j=20 -> K=32: identity-copy inputs (ij,jk) and a=20 -> K=32: 2D-TTRANS (ai,ja).
    ("ij, jk -> ik", (32, 20), (20, 128)),
    ("ai, ja -> ij", (20, 16), (64, 20)),
    ("ji, jk -> ik", (20, 32), (20, 128)),         # transposed input0 + K != C
    # batched K != C (I > 1): per-batch K-row repack of ws1 (batched_pad_copy_inline).
    ("bij, bjk -> bik", (8, 16, 20), (8, 20, 64)),

    # --- elementwise (Hadamard): no contraction, identical index order ---
    # Routed to the streaming Vector kernel, not a batch of 1x1x1 matmuls. Sizes hit
    # the sub-tile tail (1000 < one 2048 tile) and the multi-chunk / multi-lane path.
    ("i, i -> i", (1000,), (1000,)),               # 1D, sub-tile tail
    ("ts, ts -> ts", (64, 48), (64, 48)),          # 2D, spans two chunks
    ("bshd, bshd -> bshd", (2, 16, 8, 64), (2, 16, 8, 64)),  # N-D Hadamard

    # --- partial output tiles: free dim not a multiple of 16 (single batch) ---
    # The padded M/N exceed the real L0/L1, so boundary tiles carry dynamic
    # SetValidRow/SetValidCol extents that differ across tiles (e.g. 128 then 122).
    ("ij, jk -> ik", (250, 128), (128, 256)),   # partial M
    ("ij, jk -> ik", (256, 128), (128, 250)),   # partial N
    ("ij, jk -> ik", (250, 128), (128, 250)),   # both

    # --- partial *tile grid*: padded free dim not a multiple of the 128 tile ---
    # A boundary tile would leave fully-empty trailing fractal blocks, so the operand
    # is padded up to a whole tile (Ma rows / Na cols, zero-filled); the store clamps
    # to the real dim. 200 -> pad 208 -> tile-pad 256; 1000 -> Mpad 1008 -> 1024.
    ("ij, jk -> ik", (200, 64), (64, 128)),    # partial M grid
    ("ij, jk -> ik", (128, 64), (64, 200)),    # partial N grid
    ("ij, jk -> ik", (200, 64), (64, 200)),    # both
    ("ij, jk -> ik", (1000, 64), (64, 128)),   # large partial M grid
    ("ij, jk -> ik", (200, 20), (20, 200)),    # partial M+N with K!=C (j=20 -> K=32)
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


# Known-unsupported equations/shapes. These are kept here (skipped) to document the
# gaps and so that, when a path lands, the case is unskipped into a real correctness
# check. Each `skip` reason points at the limitation; the matching loud-failure
# contract (the clean error raised today) is pinned in test_unsupported_raises below.
@pytest.mark.parametrize("equation, shape0, shape1", [
    # Batched (I>1) partial output tile: a free dim not a multiple of the matmul tile
    # needs the per-batch row/col block padded; only single-batch (I==1) is done today.
    # Currently rejected with a clean ValueError (see test_unsupported_raises).
    pytest.param("bij, bjk -> bik", (4, 200, 64), (4, 64, 128),
                 marks=pytest.mark.skip(reason="batched partial output tiles (I>1) not yet supported; see roadmap")),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_unsupported_correctness(equation, shape0, shape1, dtype):
    # Skipped today; the body is the correctness check to run once the path lands.
    if TEST_DEVICE == "cpu" and dtype == torch.float16:
        pytest.skip("float16 is not supported on CPU")
    inp0 = torch.rand(shape0, dtype=dtype, device=TEST_DEVICE)
    inp1 = torch.rand(shape1, dtype=dtype, device=TEST_DEVICE)
    result = einsum(equation, inp0, inp1, device=TEST_DEVICE)
    expected = torch.einsum(equation, inp0, inp1).to(dtype=torch.float32)
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(result, expected, rtol=rtol, atol=atol)


def test_unsupported_raises():
    # Pin the loud-failure contract: unsupported configs must raise a clean error,
    # not silently miscompute or crash in the kernel. When a feature lands, the
    # corresponding entry moves to test_unsupported_correctness (unskipped).
    if TEST_DEVICE != "npu":
        pytest.skip("device-specific dtype/partition guards are NPU-only")

    # Single-operand reduction axis (`i` is only in in0 and not in the output) ->
    # rejected before codegen instead of silently miscomputing.
    a_red = torch.rand(32, 64, dtype=torch.float32, device=TEST_DEVICE)
    b_red = torch.rand(64, 128, dtype=torch.float32, device=TEST_DEVICE)
    with pytest.raises(ValueError, match="single-operand reduction is not supported"):
        einsum("ij, jk -> k", a_red, b_red, device=TEST_DEVICE)

    # Batched (I>1) partial output tile -> rejected before codegen.
    a = torch.rand(4, 200, 64, dtype=torch.float32, device=TEST_DEVICE)
    b = torch.rand(4, 64, 128, dtype=torch.float32, device=TEST_DEVICE)
    with pytest.raises(ValueError, match="Partial tiles are .*single-batch"):
        einsum("bij, bjk -> bik", a, b, device=TEST_DEVICE)

    # Unsupported dtype (only float32/float16 are generated).
    a16 = torch.rand(32, 64, dtype=torch.bfloat16, device=TEST_DEVICE)
    b16 = torch.rand(64, 32, dtype=torch.bfloat16, device=TEST_DEVICE)
    with pytest.raises(TypeError, match="Unsupported torch dtype"):
        einsum("ij, jk -> ik", a16, b16, device=TEST_DEVICE)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_pipelined_matmul_deterministic(dtype):
    # The cross-tile pipelined loop (matmul_tile_loop_pipelined) double-buffers L1
    # operands and ping-pongs the L0C accumulator across tile boundaries with a
    # non-blocking store. An earlier cross-tile design raced (needed a PIPE_ALL
    # barrier); this one is meant to be race-free. A data race would surface as
    # run-to-run variation on identical input, so pin bit-exact determinism here --
    # this is the guard that the WAR/RAW flag ordering stays correct under codegen
    # changes. 128^3 batched (64 tiles) is squarely in the pipelined regime.
    if TEST_DEVICE == "cpu":
        pytest.skip("pipelined Cube path is NPU-only")

    equation = "bij, bjk -> bik"
    shape0, shape1 = (64, 128, 128), (64, 128, 128)
    runner = EinsumBuilder(equation, [shape0, shape1], dtype, device=TEST_DEVICE).build()

    inp0 = torch.rand(shape0, dtype=dtype, device=TEST_DEVICE)
    inp1 = torch.rand(shape1, dtype=dtype, device=TEST_DEVICE)

    ref = runner(inp0, inp1).clone()
    # Also pin correctness so a deterministic-but-wrong result can't pass silently.
    expected = torch.einsum(equation, inp0, inp1).to(dtype=torch.float32)
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
    torch.testing.assert_close(ref, expected, rtol=rtol, atol=atol)

    for _ in range(8):
        out = runner(inp0, inp1)
        # Bit-exact: same kernel, same input -> identical output, every time.
        assert torch.equal(out, ref), "pipelined matmul is non-deterministic (suspect a cross-tile race)"


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
    sys.exit(pytest.main([__file__]))

