import argparse
import os
import statistics
import time

import torch_npu  # noqa: F401  (registers the 'npu' device)
import torch
from pto_einsum import EinsumBuilder

# Use headless matplotlib backend to prevent display errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Where per-benchmark plots and the summary plot are written (benchmarks/bench_results,
# shared across the base/ and complex/ suites).
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_results")

# Timing configuration.
WARMUP = 5
RUNS = 50


# Each benchmark is a *family*: one einsum equation swept over a range of tensor
# sizes, so the output shows how the kernel scales. `make(size)` returns
# `(shape0, shape1, label)` where `label` is the x-axis tick for that point.
#
# Shape constraints (see the case comments in the original suite): the two
# *output* (non-contraction) matrix dims must each be <= 128 or a multiple of
# 128; the contraction axis may be any length (the K-padding path handles it);
# batch axes are unconstrained.
BENCHMARKS = {
    # Plain aligned square Cube matmul (K == C), swept by side length.
    "matmul": {
        "name": "Standard MatMul",
        "equation": "ij, jk -> ik",
        "dtype": torch.float32,
        "param": "N (NxN @ NxN)",
        "sizes": [128, 256, 512, 1024, 2048],
        "make": lambda n: ((n, n), (n, n), str(n)),
    },
    # Same square matmul in float16 (b16 TTRANS + fp16 matmul, fp32 accumulate).
    "matmul-fp16": {
        "name": "Standard MatMul (fp16)",
        "equation": "ij, jk -> ik",
        "dtype": torch.float16,
        "param": "N (NxN @ NxN)",
        "sizes": [128, 256, 512, 1024, 2048],
        "make": lambda n: ((n, n), (n, n), str(n)),
    },
    # Transpose-heavy: input0 "ji" forces a real (non-identity) 2D transpose of a
    # large operand, driving the blocked multi-core TTRANS path. j is the
    # contraction (= 2N); i, k are the output dims (= N).
    "transpose": {
        "name": "Transpose-Heavy",
        "equation": "ji, jk -> ik",
        "dtype": torch.float32,
        "param": "N (output NxN, contract 2N)",
        "sizes": [128, 256, 512, 1024],
        "make": lambda n: ((2 * n, n), (2 * n, n), str(n)),
    },
    # Pure outer product (C == 1 -> K == 16), swept by vector length.
    "outer": {
        "name": "Outer Product",
        "equation": "i, j -> ij",
        "dtype": torch.float32,
        "param": "N (NxN output)",
        "sizes": [256, 512, 1024, 2048],
        "make": lambda n: ((n,), (n,), str(n)),
    },
    # Full reduction to a scalar, swept by contraction length.
    "dot-product": {
        "name": "Dot Product",
        "equation": "i, i -> ",
        "dtype": torch.float32,
        "param": "Length",
        "sizes": [4096, 16384, 49152, 131072],
        "make": lambda n: ((n,), (n,), str(n)),
    },
    # Batched aligned matmul (I > 1, K == C), swept by batch count.
    "batch-matmul": {
        "name": "Batch MatMul",
        "equation": "bij, bjk -> bik",
        "dtype": torch.float32,
        "param": "Batch (64x128 @ 128x64)",
        "sizes": [8, 16, 32, 64, 128],
        "make": lambda b: ((b, 64, 128), (b, 128, 64), str(b)),
    },
    # Batched-tiny full 128^3 tiles (nKd==1) -- the latency-bound regime the cross-tile
    # pipelined loop targets. The batch sweep straddles MIN_TILES (32): below it the deep
    # per-tile path runs, at/above it the pipelined loop kicks in, so the curve tracks
    # the win turning on and guards against it silently regressing vs torch.
    "batch-matmul-tiny": {
        "name": "Batch MatMul (tiny 128^3)",
        "equation": "bij, bjk -> bik",
        "dtype": torch.float32,
        "param": "Batch (128x128 @ 128x128)",
        "sizes": [16, 32, 64, 128, 256],
        "make": lambda b: ((b, 128, 128), (b, 128, 128), str(b)),
    },
    # Same batched-tiny regime in float16 (fp16 matmul, fp32 accumulate).
    "batch-matmul-tiny-fp16": {
        "name": "Batch MatMul (tiny 128^3, fp16)",
        "equation": "bij, bjk -> bik",
        "dtype": torch.float16,
        "param": "Batch (128x128 @ 128x128)",
        "sizes": [16, 32, 64, 128, 256],
        "make": lambda b: ((b, 128, 128), (b, 128, 128), str(b)),
    },
    # Multi-index contraction collapsing two axes, swept by contraction depth.
    "contraction": {
        "name": "Tensor Contraction",
        "equation": "ijk, jkl -> il",
        "dtype": torch.float32,
        "param": "K (contract 64xK)",
        "sizes": [32, 64, 128, 256],
        "make": lambda k: ((64, 64, k), (64, k, 64), str(k)),
    },
    # Custom multi-letter layout with a batched (inplace) output axis, swept by
    # the batch axis 'a'.
    "custom-layout": {
        "name": "Custom Layout",
        "equation": "abc, cd -> abd",
        "dtype": torch.float32,
        "param": "Batch (64x128 @ 128x256)",
        "sizes": [8, 16, 32, 64],
        "make": lambda a: ((a, 64, 128), (128, 256), str(a)),
    },
    # Non-16-aligned contraction (C = N-6 -> K padded up), exercising the
    # K-padded transpose destinations.
    "unaligned": {
        "name": "Unaligned Contract",
        "equation": "ij, jk -> ik",
        "dtype": torch.float32,
        "param": "N (contract N-6)",
        "sizes": [128, 256, 512, 1024],
        "make": lambda n: ((n, n - 6), (n - 6, n), str(n)),
    },
    # Batched non-16-aligned contraction (C = 100 -> K = 112), exercising the
    # per-batch K-row repack of ws1, swept by batch count.
    "batch-unaligned": {
        "name": "Batch Unaligned",
        "equation": "bij, bjk -> bik",
        "dtype": torch.float32,
        "param": "Batch (64x100 @ 100x64)",
        "sizes": [8, 16, 32, 64],
        "make": lambda b: ((b, 64, 100), (b, 100, 64), str(b)),
    },
}


def _time_one(eq, s0, s1, dtype):
    """Build, run and time a single (equation, shape) point.

    Returns (custom_ms, torch_ms, build_s), or None if the shape is unsupported.
    """
    inp0 = torch.rand(s0, dtype=dtype).npu()
    inp1 = torch.rand(s1, dtype=dtype).npu()

    t_build_start = time.perf_counter()
    try:
        builder = EinsumBuilder(eq, [s0, s1], dtype, device="npu")
        runner = builder.build()
    except ValueError:
        return None
    build_s = time.perf_counter() - t_build_start

    for _ in range(WARMUP):
        _ = runner(inp0, inp1)
        _ = torch.einsum(eq, inp0, inp1)
    torch.npu.synchronize()

    custom_samples = []
    torch_samples = []
    for _ in range(RUNS):
        t_start = time.perf_counter()
        _ = runner(inp0, inp1)
        torch.npu.synchronize()
        custom_samples.append((time.perf_counter() - t_start) * 1000.0)

        t_start = time.perf_counter()
        _ = torch.einsum(eq, inp0, inp1)
        torch.npu.synchronize()
        torch_samples.append((time.perf_counter() - t_start) * 1000.0)

    runner = None
    builder.cleanup()

    return statistics.median(custom_samples), statistics.median(torch_samples), build_s


def run_family(key, bench):
    """Sweep one benchmark family over its size range. Returns a points list."""
    name = bench["name"]
    eq = bench["equation"]
    dtype = bench["dtype"]
    dtype_str = "float16" if dtype == torch.float16 else "float32"

    print("-" * 100)
    print(f"{name}  [{key}]   '{eq}'   {dtype_str}")
    print(f"{'Size':<14} | {'Build (s)':<9} | {'Custom (ms)':<13} | {'PyTorch (ms)':<13} | {'Speedup':<8}")
    print("-" * 100)

    points = []
    for size in bench["sizes"]:
        s0, s1, label = bench["make"](size)
        res = _time_one(eq, s0, s1, dtype)
        if res is None:
            print(f"{label:<14} | {'Skipped':<9} | {'N/A':<13} | {'N/A':<13} | {'N/A':<8}")
            continue
        custom_ms, torch_ms, build_s = res
        speedup = torch_ms / custom_ms if custom_ms > 0 else 0.0
        print(f"{label:<14} | {build_s:<9.4f} | {custom_ms:<13.4f} | {torch_ms:<13.4f} | {speedup:<7.2f}x")
        points.append({
            "label": label,
            "custom_ms": custom_ms,
            "torch_ms": torch_ms,
            "speedup": speedup,
        })

    return points


def plot_family(key, bench, points):
    """One grouped-bar figure (custom vs torch over the size sweep)."""
    if not points:
        return
    labels = [p["label"] for p in points]
    custom = [p["custom_ms"] for p in points]
    torch_t = [p["torch_ms"] for p in points]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(labels)), 6))
    ax.set_yscale('log')
    ax.bar([i - width / 2 for i in x], custom, width, label='Custom Pre-compiled C++', color='#1f77b4')
    ax.bar([i + width / 2 for i in x], torch_t, width, label='Native PyTorch NPU', color='#ff7f0e')

    ax.set_ylabel('Execution Time (ms) - Log Scale')
    ax.set_xlabel(bench["param"])
    ax.set_title(f"{bench['name']}: '{bench['equation']}'")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    for i, p in enumerate(points):
        y_pos = max(p["custom_ms"], p["torch_ms"]) * 1.2
        ax.text(i, y_pos, f"{p['speedup']:.2f}x", ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='purple')

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{key}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


def plot_summary(family_results):
    """One figure: per-family mean speedup with min/max range as error bars."""
    items = [(bench["name"], pts) for _, bench, pts in family_results if pts]
    if not items:
        return

    names = [n for n, _ in items]
    speedups = [[p["speedup"] for p in pts] for _, pts in items]
    means = [statistics.mean(s) for s in speedups]
    mins = [min(s) for s in speedups]
    maxs = [max(s) for s in speedups]
    # Asymmetric error bars: distance from mean down to min, up to max.
    yerr_lo = [m - lo for m, lo in zip(means, mins)]
    yerr_hi = [hi - m for m, hi in zip(means, maxs)]

    x = range(len(names))
    fig, ax = plt.subplots(figsize=(max(10, 1.3 * len(names)), 6))
    colors = ['#2ca02c' if m >= 1.0 else '#d62728' for m in means]
    ax.bar(list(x), means, 0.6, color=colors, alpha=0.85)
    ax.errorbar(list(x), means, yerr=[yerr_lo, yerr_hi], fmt='none',
                ecolor='black', elinewidth=1.2, capsize=5)

    ax.axhline(1.0, color='gray', ls='--', lw=1, label='parity (1.0x)')
    ax.set_ylabel('Speedup vs PyTorch (mean, bars = min/max over sweep)')
    ax.set_title('Einsum Speedup Summary (custom pre-compiled C++ vs PyTorch NPU)')
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, axis='y', ls="--", alpha=0.5)

    for i, (m, lo, hi) in enumerate(zip(means, mins, maxs)):
        ax.text(i, hi + 0.02, f"{m:.2f}x\n[{lo:.2f}-{hi:.2f}]", ha='center',
                va='bottom', fontsize=8, fontweight='bold')

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "summary.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved summary plot to: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pto-einsum vs torch.einsum on NPU, sweeping tensor sizes.")
    parser.add_argument(
        "benchmark", nargs="?", default=None,
        help="Benchmark to run (default: all). One of: " + ", ".join(BENCHMARKS))
    args = parser.parse_args()

    if args.benchmark is None:
        selected = list(BENCHMARKS.items())
    elif args.benchmark in BENCHMARKS:
        selected = [(args.benchmark, BENCHMARKS[args.benchmark])]
    else:
        parser.error(f"unknown benchmark '{args.benchmark}'. "
                     f"Choose from: {', '.join(BENCHMARKS)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 100)
    print(f"{'EINSUM PRE-COMPILED C++ VS PYTORCH NPU BENCHMARK (size sweep)':^100}")
    print("=" * 100)
    print(f"Warmup: {WARMUP} | Runs: {RUNS} | Benchmarks: {len(selected)}")

    family_results = []
    for key, bench in selected:
        points = run_family(key, bench)
        plot_family(key, bench, points)
        family_results.append((key, bench, points))

    print("=" * 100)
    plot_summary(family_results)


if __name__ == "__main__":
    main()
