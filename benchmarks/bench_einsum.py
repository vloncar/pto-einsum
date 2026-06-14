import time
import os
import torch
from pto_einsum import EinsumBuilder

# Use headless matplotlib backend to prevent display errors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_benchmark():
    # Define test cases for larger tensors
    cases = [
        {
            "name": "Standard Matrix Mult",
            "equation": "ij, jk -> ik",
            "shape0": (256, 256),
            "shape1": (256, 256),
        },
        {
            "name": "Outer Product",
            "equation": "i, j -> ij",
            "shape0": (2048,),
            "shape1": (1536,),
        },
        {
            "name": "Dot Product",
            "equation": "i, i -> ",
            "shape0": (49152,),
            "shape1": (49152,),
        },
        {
            "name": "Batch Matrix Mult",
            "equation": "bij, bjk -> bik",
            "shape0": (32, 64, 80),
            "shape1": (32, 80, 64),
        },
        {
            "name": "Tensor Contraction",
            "equation": "ijk, jkl -> il",
            "shape0": (32, 64, 80),
            "shape1": (64, 80, 32),
        },
    ]

    dtype = torch.float32
    warmup = 10
    runs = 100

    results = []

    print("=" * 95)
    print(f"{'EINSUM PRE-COMPILED C++ VS PYTORCH NPU BENCHMARK':^95}")
    print("=" * 95)
    print(f"Warmup iterations: {warmup} | Benchmark runs: {runs} | Precision: {dtype}")
    print("-" * 95)
    print(f"{'Test Case':<32} | {'Build (s)':<10} | {'Custom C++ (ms)':<15} | {'PyTorch NPU (ms)':<15} | {'Speedup':<10}")
    print("-" * 95)

    for case in cases:
        name = case["name"]
        eq = case["equation"]
        s0, s1 = case["shape0"], case["shape1"]

        inp0 = torch.rand(s0, dtype=dtype).npu()
        inp1 = torch.rand(s1, dtype=dtype).npu()

        # 1. Benchmark Builder/Compilation (one-time build overhead)
        t_build_start = time.perf_counter()
        try:
            builder = EinsumBuilder(eq, [s0, s1], dtype, device="npu")
            runner = builder.build()
        except ValueError as e:
            print(f"{name:<32} | {'Skipped':<10} | {'N/A':<15} | {'N/A':<15} | {'N/A':<10}")
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
        print(f"{name:<32} | {t_build:<10.4f} | {avg_custom_ms:<15.4f} | {avg_torch_ms:<15.4f} | {speedup:<10.2f}x")

        results.append({
            "name": name,
            "custom_ms": avg_custom_ms,
            "torch_ms": avg_torch_ms,
            "speedup": speedup
        })

    print("=" * 95)
    
    # Plotting the results
    plot_results(results)


def plot_results(results):
    names = [r["name"] for r in results]
    custom_times = [r["custom_ms"] for r in results]
    torch_times = [r["torch_ms"] for r in results]
    
    x = range(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use logarithmic scale to handle high dynamic range of execution times
    ax.set_yscale('log')
    
    # Plot bars
    rects1 = ax.bar([i - width/2 for i in x], custom_times, width, label='Custom Pre-compiled C++', color='#1f77b4')
    rects2 = ax.bar([i + width/2 for i in x], torch_times, width, label='Native PyTorch NPU', color='#ff7f0e')
    
    # Add labels and titles
    ax.set_ylabel('Execution Time (ms) - Log Scale')
    ax.set_title('Einsum Execution Time Comparison: Custom pre-compiled C++ vs PyTorch NPU')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
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
