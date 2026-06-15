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

