"""Micro-benchmark for transposing einsums (exercises the 2D transpose path).

`ai, ja -> ij` requires a 2D transpose of *each* input into the canonical matmul
layout, so its runtime is dominated by the transpose kernel — unlike the identity
einsums in bench_einsum.py. Large dims push both inputs through the blocked path.
"""
import os
import time
import torch
import torch_npu
from pto_einsum import EinsumBuilder

CASES = [
    ("ai, ja -> ij", (512, 512), (512, 512)),   # blocked path, both inputs
    ("ai, ja -> ij", (256, 256), (256, 256)),   # blocked path, both inputs
    ("ai, ja -> ij", (64, 64), (64, 64)),        # whole-tensor path
]


def bench(eq, s0, s1, dtype=torch.float32, warmup=10, runs=100):
    a = torch.rand(s0, dtype=dtype, device="npu")
    b = torch.rand(s1, dtype=dtype, device="npu")
    runner = EinsumBuilder(eq, [s0, s1], dtype, device="npu").build()
    for _ in range(warmup):
        runner(a, b)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        runner(a, b)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / runs * 1e3


if __name__ == "__main__":
    print(f"{'Equation':16} {'shape0':14} {'shape1':14} {'Custom (ms)':>12}")
    for eq, s0, s1 in CASES:
        ms = bench(eq, s0, s1)
        print(f"{eq:16} {str(s0):14} {str(s1):14} {ms:12.4f}")
