"""Plan 0 — quantify the transpose<->matmul overlap ceiling for the fused einsum.

The fused kernel runs bulk-synchronously: Phase A (Vec input transposes) -> barrier
-> Phase B (Cube matmul) -> barrier -> Phase C (Vec output transpose). At any instant
one engine (all AIV or all AIC) is idle. This script measures how big that bubble is
for real (linear-)attention contractions, so we can decide whether a producer/consumer
overlap is worth building (Plan 1/2) before writing any of it.

Method: the kernel is gated by EINSUM_PHASE_STOP (see pto_einsum.h). The cross-core
barriers always run; only the phase *compute* bodies are capped. So building the same
equation at stop 0/1/2/3 and host-timing each isolates every phase by subtraction:

    t(0) = F                      (launch + barriers, no compute)
    t(1) = F + T_A                (+ input transposes)
    t(2) = F + T_A + T_B          (+ Cube matmul)
    t(3) = F + T_A + T_B + T_C    (+ output transpose) == production kernel

    T_A = t(1)-t(0)   T_B = t(2)-t(1)   T_C = t(3)-t(2)

The overlap is between the Vec work (A+C) and the Cube work (B). With perfect
pipelining the runtime -> max(T_A+T_C, T_B) instead of the serial T_A+T_B+T_C, so the
*upper bound* on the saving is

    ceiling = min(T_A+T_C, T_B) / (T_A+T_B+T_C)

This is an idealized ceiling (infinite tiles, no startup/drain, perfect rate match).
A coarse-grained real pipeline gets a fraction of it. If the ceiling is small here,
the overlap is not worth the cross-core race surface.

Run (header edit -> clear cache first):
    export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
    rm -rf build
    python benchmarks/profile_phase_overlap.py            # default config set
    python benchmarks/profile_phase_overlap.py --dtype fp16
"""

import argparse
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401  (registers the 'npu' device)
from pto_einsum import EinsumBuilder

WARMUP = 10
INNER = 50      # kernel launches per timed sample (amortizes host launch/sync overhead)
SAMPLES = 25    # timed samples; we take the median

# (label, equation, [shape0, shape1], note) — real attention contractions plus a
# plain-matmul control. b,s,h batch/free axes; d/n/s contracted.
def configs():
    return [
        # Softmax attention QK^T scores: N-D input transposes on BOTH operands and a
        # non-identity output -> all three phases are real. The richest overlap case.
        ("attn_scores  S=128",  "bshd,bthd->bsht", [(4, 128, 8, 64), (4, 128, 8, 64)]),
        ("attn_scores  S=512",  "bshd,bthd->bsht", [(4, 512, 8, 64), (4, 512, 8, 64)]),
        ("attn_scores  S=1024", "bshd,bthd->bsht", [(2, 1024, 8, 64), (2, 1024, 8, 64)]),
        # Attention context A·V: contract over key length t, N-D transposes + non-id out.
        ("attn_context S=512",  "bsht,bthd->bshd", [(4, 512, 8, 512), (4, 512, 8, 64)]),
        # Linear-attention QK^T: one input transpose (input1) + matmul, identity output
        # (no Phase C). The Vec side here is a single operand transpose.
        ("linattn_qk  T=2048",  "tn,sn->ts",       [(2048, 64), (2048, 64)]),
        ("linattn_qk  T=512",   "tn,sn->ts",       [(512, 64), (512, 64)]),
        # Linear-attention A·V: plain skip-both matmul (identity perms, K==C, identity
        # out) -> Phase A & C ~ 0. Pure-Cube control: ceiling should read ~0%.
        ("linattn_av  T=2048",  "ts,sp->tp",       [(2048, 2048), (2048, 128)]),
    ]


def time_variant(eq, shapes, dtype, stop):
    """Build the equation with EINSUM_PHASE_STOP=stop and return median ms/launch."""
    os.environ["EINSUM_EXTRA_DEFINES"] = f"-DEINSUM_PHASE_STOP={stop}"
    builder = EinsumBuilder(eq, shapes, dtype, device="npu")
    runner = builder.build()
    try:
        a = torch.rand(*shapes[0], dtype=dtype, device="npu")
        b = torch.rand(*shapes[1], dtype=dtype, device="npu")
        for _ in range(WARMUP):
            runner(a, b)
        torch.npu.synchronize()

        samples = []
        for _ in range(SAMPLES):
            t0 = time.perf_counter()
            for _ in range(INNER):
                runner(a, b)
            torch.npu.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0 / INNER)
        return statistics.median(samples)
    finally:
        builder.cleanup()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    args = ap.parse_args()
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    print("=" * 104)
    print(f"{'PHASE-OVERLAP PROFILE (Plan 0) -- fused einsum, NPU ' + args.dtype:^104}")
    print("=" * 104)
    print(f"Inner={INNER}  Samples={SAMPLES}  (median ms per kernel launch)")
    print("-" * 104)
    print(f"{'config':<22}{'equation':<18}{'F':>8}{'T_A':>8}{'T_B':>8}{'T_C':>8}"
          f"{'total':>9}{'Vec=A+C':>9}{'Cube=B':>8}{'ceiling':>9}")
    print("-" * 104)

    for label, eq, shapes in configs():
        try:
            t = {s: time_variant(eq, shapes, dtype, s) for s in (0, 1, 2, 3)}
        except Exception as exc:
            print(f"{label:<22}{eq:<18}  FAILED: {str(exc).splitlines()[0][:54]}")
            continue

        F = t[0]
        T_A = max(t[1] - t[0], 0.0)
        T_B = max(t[2] - t[1], 0.0)
        T_C = max(t[3] - t[2], 0.0)
        total = T_A + T_B + T_C
        vec = T_A + T_C
        cube = T_B
        ceiling = (min(vec, cube) / total * 100.0) if total > 1e-9 else 0.0

        print(f"{label:<22}{eq:<18}{F:>8.3f}{T_A:>8.3f}{T_B:>8.3f}{T_C:>8.3f}"
              f"{total:>9.3f}{vec:>9.3f}{cube:>8.3f}{ceiling:>8.1f}%")

    print("-" * 104)
    print("ceiling = min(Vec, Cube)/total -- idealized upper bound on overlap saving.")
    print("Pure-Cube controls (plain matmul) should read ~0%: nothing to overlap.")


if __name__ == "__main__":
    main()
