import time
import os
import torch
from pto_einsum import EinsumBuilder

# Use headless matplotlib backend to prevent display errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_benchmark():
    # Benchmark cases, chosen to cover each implemented capability. `dtype`
    # defaults to float32. All transposed/blocked dims are kept 16-aligned (large
    # unaligned 2D transposes are not yet supported); non-16-aligned dims are
    # restricted to the contraction axis, which the K-padding path handles.
    cases = [
        # Plain aligned Cube matmul (K == C).
        {
            "name": "Standard MatMul",
            "equation": "ij, jk -> ik",
            "shape0": (256, 256),
            "shape1": (256, 256),
        },
        # Same in float16 (b16 TTRANS + fp16 matmul with fp32 accumulation).
        {
            "name": "Standard MatMul (fp16)",
            "equation": "ij, jk -> ik",
            "shape0": (512, 512),
            "shape1": (512, 512),
            "dtype": torch.float16,
        },
        # Transpose-heavy: input0 "ji" needs a real (non-identity) 2D transpose of
        # a large operand, driving the blocked multi-core TTRANS path.
        {
            "name": "Transpose-Heavy",
            "equation": "ji, jk -> ik",
            "shape0": (1024, 512),
            "shape1": (1024, 256),
        },
        # Degenerate contraction (C == 1 -> K == 16): a pure outer product.
        {
            "name": "Outer Product",
            "equation": "i, j -> ij",
            "shape0": (2048,),
            "shape1": (1536,),
        },
        # Full reduction to a scalar.
        {
            "name": "Dot Product",
            "equation": "i, i -> ",
            "shape0": (49152,),
            "shape1": (49152,),
        },
        # Batched aligned matmul (I > 1, K == C).
        {
            "name": "Batch MatMul",
            "equation": "bij, bjk -> bik",
            "shape0": (32, 64, 80),
            "shape1": (32, 80, 64),
        },
        # Multi-index contraction collapsing two axes.
        {
            "name": "Tensor Contraction",
            "equation": "ijk, jkl -> il",
            "shape0": (32, 64, 80),
            "shape1": (64, 80, 32),
        },
        # Custom multi-letter layout with a batched (inplace) output axis.
        {
            "name": "Custom Layout",
            "equation": "abc, cd -> abd",
            "shape0": (32, 64, 128),
            "shape1": (128, 256),
        },
        # Non-16-aligned contraction (C = 250 -> K = 256), single batch: exercises
        # the K-padded transpose destinations.
        {
            "name": "Unaligned Contract",
            "equation": "ij, jk -> ik",
            "shape0": (256, 250),
            "shape1": (250, 256),
        },
        # Batched non-16-aligned contraction (C = 100 -> K = 112, I > 1): exercises
        # the per-batch K-row repack of ws1 (batched_pad_copy_inline).
        {
            "name": "Batch Unaligned",
            "equation": "bij, bjk -> bik",
            "shape0": (16, 64, 100),
            "shape1": (16, 100, 64),
        },
    ]

    warmup = 10
    runs = 100

    results = []

    print("=" * 100)
    print(f"{'EINSUM PRE-COMPILED C++ VS PYTORCH NPU BENCHMARK':^100}")
    print("=" * 100)
    print(f"Warmup iterations: {warmup} | Benchmark runs: {runs}")
    print("-" * 100)
    print(f"{'Test Case':<24} | {'Dtype':<7} | {'Build (s)':<9} | {'Custom (ms)':<13} | {'PyTorch (ms)':<13} | {'Speedup':<8}")
    print("-" * 100)

    for case in cases:
        name = case["name"]
        eq = case["equation"]
        s0, s1 = case["shape0"], case["shape1"]
        dtype = case.get("dtype", torch.float32)
        dtype_str = "float16" if dtype == torch.float16 else "float32"

        inp0 = torch.rand(s0, dtype=dtype).npu()
        inp1 = torch.rand(s1, dtype=dtype).npu()

        # 1. Benchmark Builder/Compilation (one-time build overhead)
        t_build_start = time.perf_counter()
        try:
            builder = EinsumBuilder(eq, [s0, s1], dtype, device="npu")
            runner = builder.build()
        except ValueError as e:
            print(f"{name:<24} | {dtype_str:<7} | {'Skipped':<9} | {'N/A':<13} | {'N/A':<13} | {'N/A':<8}")
            continue
        t_build = time.perf_counter() - t_build_start

        # Warm up execution
        for _ in range(warmup):
            _ = runner(inp0, inp1)
            _ = torch.einsum(eq, inp0, inp1)
        torch.npu.synchronize()

        # 2. Benchmark Custom JIT C++ Execution
        t_custom_total = 0.0
        for _ in range(runs):
            t_start = time.perf_counter()
            _ = runner(inp0, inp1)
            torch.npu.synchronize()
            t_custom_total += (time.perf_counter() - t_start)
        avg_custom_ms = (t_custom_total / runs) * 1000.0

        # 3. Benchmark Native PyTorch NPU Execution
        t_torch_total = 0.0
        for _ in range(runs):
            t_start = time.perf_counter()
            _ = torch.einsum(eq, inp0, inp1)
            torch.npu.synchronize()
            t_torch_total += (time.perf_counter() - t_start)
        avg_torch_ms = (t_torch_total / runs) * 1000.0

        # Release ctypes lock and clean up build artifacts
        runner = None
        builder.cleanup()

        # Compute speedup
        speedup = avg_torch_ms / avg_custom_ms if avg_custom_ms > 0 else 0.0
        print(f"{name:<24} | {dtype_str:<7} | {t_build:<9.4f} | {avg_custom_ms:<13.4f} | {avg_torch_ms:<13.4f} | {speedup:<7.2f}x")

        results.append({
            "name": name,
            "custom_ms": avg_custom_ms,
            "torch_ms": avg_torch_ms,
            "speedup": speedup
        })

    print("=" * 100)
    
    # Plotting the results
    plot_results(results)


def plot_results(results):
    names = [r["name"] for r in results]
    custom_times = [r["custom_ms"] for r in results]
    torch_times = [r["torch_ms"] for r in results]
    
    x = range(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(max(10, 1.3 * len(names)), 6))
    
    # Use logarithmic scale to handle high dynamic range of execution times
    ax.set_yscale('log')
    
    # Plot bars
    rects1 = ax.bar([i - width/2 for i in x], custom_times, width, label='Custom Pre-compiled C++', color='#1f77b4')
    rects2 = ax.bar([i + width/2 for i in x], torch_times, width, label='Native PyTorch NPU', color='#ff7f0e')
    
    # Add labels and titles
    ax.set_ylabel('Execution Time (ms) - Log Scale')
    ax.set_title('Einsum Execution Time Comparison: Custom pre-compiled C++ vs PyTorch NPU')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    # Annotate speedup labels above the bars
    for i, r in enumerate(results):
        speedup_text = f"{r['speedup']:.2f}x"
        # Place label above the taller bar
        y_pos = max(r["custom_ms"], r["torch_ms"]) * 1.2
        ax.text(i, y_pos, speedup_text, ha='center', va='bottom', fontsize=9, fontweight='bold', color='purple')
        
    fig.tight_layout()
    
    # Save the output file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved benchmark comparison plot to: {output_path}")


if __name__ == "__main__":
    run_benchmark()
