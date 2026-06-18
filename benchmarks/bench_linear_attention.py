"""Benchmark the linear-attention chain `einsum("tn,sn,sp,ts->tp", Q,K,V,L)`
across a sweep of sequence lengths: pto-einsum vs torch.einsum (PyTorch NPU).

Linear attention is a 4-way contraction. pto-einsum takes exactly two operands,
so it is evaluated as the three two-way steps it decomposes into (see
tests/test_einsum.py::test_linear_attention_chain):

    A = QKᵀ        einsum("tn, sn -> ts")   -- matmul
    A = A ⊙ L      einsum("ts, ts -> ts")   -- L-mask Hadamard (elementwise path)
    O = A V        einsum("ts, sp -> tp")   -- matmul

The PyTorch baseline evaluates the whole 4-way `torch.einsum` in one call, so the
comparison is "the full linear-attention op, pto's three fused kernels vs torch's
native contraction".

Sequence length T == S is swept over both powers of two and non-power-of-two
lengths: the matmul now supports arbitrary output dims (it pads a partial boundary
tile's operand up to a whole tile). A *large non-16-aligned* length (e.g. 1000)
still needs the blocked 2D transpose's 16-aligned-dim support -- a separate roadmap
item -- so the chain build is caught and that length is reported Skipped. The
feature dims N (key) and P (value) are fixed (defaults mirror the correctness test).

Note: the pto runner always returns fp32 (the kernel accumulates in fp32), so each
intermediate is cast back to the working dtype before the next step -- exactly the
`.to(dtype)` the correctness test does.
"""

import argparse
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401  (registers the 'npu' device)
from pto_einsum import EinsumBuilder

# Headless matplotlib so the script runs without a display.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results")

WARMUP = 5
RUNS = 50

# Sequence lengths (T == S): powers of two plus non-power-of-two lengths that the
# matmul partial-tile support now handles (320, 768, 1280 are 16-aligned; 200 isn't
# but is small enough for the non-blocked transpose).
SEQ_LENS = [200, 256, 320, 512, 768, 1024, 1280, 2048]

# Fixed feature dims (key dim N, value dim P), mirroring the correctness test.
N_KEY = 64
P_VAL = 128

# The 4-way einsum PyTorch evaluates in one call.
EQ_4WAY = "tn,sn,sp,ts->tp"


class LinearAttentionChain:
    """Three persistent pto-einsum runners for one sequence length.

    Builds (and JIT-compiles) the QKᵀ, L-mask and ·V kernels once; `run` then just
    launches them on the stream. The L mask is a fixed lower-triangular ones matrix.
    """

    def __init__(self, T, S, N, P, dtype, device="npu"):
        self.dtype = dtype
        self._builders = []
        self._qk = self._make("tn, sn -> ts", [(T, N), (S, N)], dtype, device)
        self._mask = self._make("ts, ts -> ts", [(T, S), (T, S)], dtype, device)
        self._av = self._make("ts, sp -> tp", [(T, S), (S, P)], dtype, device)
        self.L = torch.tril(torch.ones(T, S, dtype=dtype, device=device))

    def _make(self, eq, shapes, dtype, device):
        builder = EinsumBuilder(eq, shapes, dtype, device=device)
        runner = builder.build()
        self._builders.append(builder)
        return runner

    def run(self, Q, K, V):
        # Each runner returns fp32; cast back to the working dtype between steps so
        # the next runner's dtype validation passes (and fp16 stays fp16 end-to-end).
        A = self._qk(Q, K).to(self.dtype)
        A = self._mask(A, self.L).to(self.dtype)
        O = self._av(A, V)
        return O

    def cleanup(self):
        for b in self._builders:
            b.cleanup()
        self._builders = []


def _torch_ref(Q, K, V, L):
    return torch.einsum(EQ_4WAY, Q, K, V, L)


def time_seqlen(seqlen, N, P, dtype):
    """Build the chain, validate it, and time pto vs torch. Returns a points dict
    (or None if the build is rejected)."""
    T = S = seqlen
    Q = torch.rand(T, N, dtype=dtype, device="npu")
    K = torch.rand(S, N, dtype=dtype, device="npu")
    V = torch.rand(S, P, dtype=dtype, device="npu")

    t_build = time.perf_counter()
    try:
        chain = LinearAttentionChain(T, S, N, P, dtype, device="npu")
    except (ValueError, RuntimeError):
        # ValueError: config rejected by the builder. RuntimeError: kernel failed to
        # compile -- a non-16-aligned *large* sequence length still needs the blocked
        # 2D transpose's 16-aligned-dim support (a separate roadmap item from the
        # matmul partial-tile work that lifted the power-of-two restriction here).
        return None
    build_s = time.perf_counter() - t_build

    # Correctness gate: the fused chain must match torch's 4-way einsum (fp32 ref).
    ref = torch.einsum(EQ_4WAY, Q.float(), K.float(), V.float(), chain.L.float())
    out = chain.run(Q, K, V).float()
    rtol, atol = (1e-3, 1e-3) if dtype == torch.float32 else (5e-2, 5e-2)
    max_diff = (out - ref).abs().max().item()
    correct = torch.allclose(out, ref, rtol=rtol, atol=atol)

    for _ in range(WARMUP):
        _ = chain.run(Q, K, V)
        _ = _torch_ref(Q, K, V, chain.L)
    torch.npu.synchronize()

    pto_samples, torch_samples = [], []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        _ = chain.run(Q, K, V)
        torch.npu.synchronize()
        pto_samples.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = _torch_ref(Q, K, V, chain.L)
        torch.npu.synchronize()
        torch_samples.append((time.perf_counter() - t0) * 1000.0)

    chain.cleanup()

    pto_ms = statistics.median(pto_samples)
    torch_ms = statistics.median(torch_samples)
    return {
        "label": str(seqlen),
        "pto_ms": pto_ms,
        "torch_ms": torch_ms,
        "speedup": torch_ms / pto_ms if pto_ms > 0 else 0.0,
        "build_s": build_s,
        "correct": correct,
        "max_diff": max_diff,
    }


def run_sweep(dtype, seq_lens, N, P):
    dtype_str = "float16" if dtype == torch.float16 else "float32"
    print("-" * 104)
    print(f"Linear attention  '{EQ_4WAY}'   {dtype_str}   (N={N}, P={P})")
    print(f"{'SeqLen':<10} | {'Build (s)':<9} | {'pto (ms)':<11} | {'PyTorch (ms)':<13} | "
          f"{'Speedup':<8} | {'OK':<4} | {'max|diff|':<10}")
    print("-" * 104)

    points = []
    for seqlen in seq_lens:
        res = time_seqlen(seqlen, N, P, dtype)
        if res is None:
            print(f"{str(seqlen):<10} | {'Skipped':<9} | {'N/A':<11} | {'N/A':<13} | "
                  f"{'N/A':<8} | {'-':<4} | {'-':<10}")
            continue
        ok = "yes" if res["correct"] else "NO"
        print(f"{res['label']:<10} | {res['build_s']:<9.4f} | {res['pto_ms']:<11.4f} | "
              f"{res['torch_ms']:<13.4f} | {res['speedup']:<7.2f}x | {ok:<4} | {res['max_diff']:<10.2e}")
        points.append(res)
    return points


def plot_sweep(dtype, points):
    if not points:
        return
    dtype_str = "float16" if dtype == torch.float16 else "float32"
    labels = [p["label"] for p in points]
    pto = [p["pto_ms"] for p in points]
    torch_t = [p["torch_ms"] for p in points]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(labels)), 6))
    ax.set_yscale('log')
    ax.bar([i - width / 2 for i in x], pto, width, label='pto-einsum (fused)', color='#1f77b4')
    ax.bar([i + width / 2 for i in x], torch_t, width, label='PyTorch NPU (4-way)', color='#ff7f0e')

    ax.set_ylabel('Execution Time (ms) - Log Scale')
    ax.set_xlabel('Sequence length (T = S)')
    ax.set_title(f"Linear attention '{EQ_4WAY}' ({dtype_str})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    for i, p in enumerate(points):
        y = max(p["pto_ms"], p["torch_ms"]) * 1.2
        ax.text(i, y, f"{p['speedup']:.2f}x", ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='purple')

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"linear_attention_{dtype_str}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the linear-attention chain (pto-einsum vs torch.einsum) "
                    "over power-of-two sequence lengths.")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "both"], default="both",
                        help="Precision to benchmark (default: both).")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=SEQ_LENS,
                        help=f"Sequence lengths to sweep (default: {SEQ_LENS}).")
    parser.add_argument("--n-key", type=int, default=N_KEY, help=f"Key dim N (default: {N_KEY}).")
    parser.add_argument("--p-val", type=int, default=P_VAL, help=f"Value dim P (default: {P_VAL}).")
    args = parser.parse_args()

    dtypes = {"fp32": [torch.float32], "fp16": [torch.float16],
              "both": [torch.float32, torch.float16]}[args.dtype]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 104)
    print(f"{'LINEAR ATTENTION CHAIN BENCHMARK -- pto-einsum vs PyTorch NPU':^104}")
    print("=" * 104)
    print(f"Warmup: {WARMUP} | Runs: {RUNS} | Seq lens: {args.seq_lens}")

    for dtype in dtypes:
        points = run_sweep(dtype, args.seq_lens, args.n_key, args.p_val)
        plot_sweep(dtype, points)
    print("=" * 104)


if __name__ == "__main__":
    main()
