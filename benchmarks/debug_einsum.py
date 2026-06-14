import torch
import sys
from pto_einsum import einsum

def test():
    equation = "ij, jk -> ik"
    shape0 = (32, 64)
    shape1 = (64, 128)
    dtype = torch.float16

    inp0 = torch.rand(shape0, dtype=dtype, device="npu")
    inp1 = torch.rand(shape1, dtype=dtype, device="npu")

    print(f"Running pto einsum: {equation} with shapes {shape0} x {shape1}...", flush=True)
    result = einsum(equation, inp0, inp1)

    print("Running torch einsum...", flush=True)
    expected = torch.einsum(equation, inp0, inp1).to(dtype=torch.float32)

    print("Checking correctness...", flush=True)
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
    print("Success! Test passed.")

if __name__ == "__main__":
    test()
