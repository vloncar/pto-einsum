"""4-way GDN benchmark: torch reference vs torch-einsum vs pto-einsum vs megagdn-pto.

For every (N_seq, L_seg) workload it times each of the six GDN stages
(cumsum, kkt, solve_tril, wy_fast, chunk_h, chunk_o) for all four implementations.
The rendered chart is a *grouped stacked bar chart* restricted to the four
contraction stages (kkt, wy_fast, chunk_h, chunk_o) for the three NPU
implementations (torch.einsum, pto-einsum, megagdn) — the stages stack within a
bar, the implementations form a group, and each (N_seq, L_seg) workload is one
group. The torch-glue stages (cumsum, and the dominant solve_tril) are timed and
recorded in the JSON but deliberately not plotted: a full-pipeline chart is
swamped by solve_tril and says nothing about the einsum contraction backend, which
is what this benchmark exists to compare.

  1. torch reference  — RefGDN, naive per-head Python loops, fp32          (ref_gdn.py)
  2. torch einsum     — EinsumGDN, head-vectorised torch.einsum, fp32      (einsum_gdn.py)
  3. pto-einsum       — PtoGDN, same glue but contractions via pto_einsum, fp32 (pto_gdn.py)
  4. megagdn-pto      — MegaGDN, hand-written fused PTO kernels, staged, fp16 (mega_gdn.py)

Implementations 1-3 share identical glue and run in fp32; megagdn runs its native
fp16. Per-stage inputs are a single fp32 reference forward (same tensors fed to all
torch-family stages); megagdn stages run on its own fp16 buffers after one warmup
pipeline. Correctness of every implementation's full pipeline is checked against the
fp32 reference (Frobenius relative error).

Run:
    export PTO_LIB_PATH=/home/vloncar/work/einsum_workspace/pto-isa
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0
    python benchmarks/complex/gdn/bench_gdn_4way.py
    python benchmarks/complex/gdn/bench_gdn_4way.py --configs 1x512,2x512,1x1024,2x1024 --H 32
"""
import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_npu  # noqa: F401

from utils import generate_random_inputs
from ref_gdn import RefGDN
from einsum_gdn import EinsumGDN
from pto_gdn import PtoGDN
from mega_gdn import MegaGDN

C = 128
D = 128
STAGES = ["cumsum", "kkt", "solve_tril", "wy_fast", "chunk_h", "chunk_o"]
IMPLS = ["torch_ref", "torch_einsum", "pto_einsum", "megagdn"]
IMPL_LABELS = {
    "torch_ref": "torch ref",
    "torch_einsum": "torch einsum",
    "pto_einsum": "pto-einsum",
    "megagdn": "megagdn-pto",
}


# ---------------------------------------------------------------------------
# Timing helper — device time via NPU events, with an L2 flush between samples.
# ---------------------------------------------------------------------------
_FLUSH = None  # single reused L2-flush buffer (allocating one per call thrashes the allocator)


def _flush_buf():
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="npu")
    return _FLUSH


def bench_npu(fn, warmup=3, iters=10):
    flush = _flush_buf()
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        flush.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.npu.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2]  # median ms


def frob_rel(actual, expected):
    a, e = actual.double(), expected.double()
    denom = e.pow(2).sum()
    if denom == 0:
        return 0.0
    return (((a - e).pow(2).sum()) / denom).sqrt().item()


# ---------------------------------------------------------------------------
# Per-config measurement
# ---------------------------------------------------------------------------
def measure_config(n_seq, l_seg, H, Hg, einsum_gdn, pto_gdn, ref_iters):
    dev = "npu:0"
    T = n_seq * l_seg
    scale = D ** -0.5
    cu_t = torch.arange(0, T + 1, l_seg, dtype=torch.int32, device=dev)
    cu_list = cu_t.tolist()

    q, k, v, beta, g_in = (x.to(dev) for x in
                           generate_random_inputs(T, H, Hg, D, dtype=torch.float32))

    # fp32 reference (golden) on NPU + captured intermediates for per-stage timing.
    ref = RefGDN(torch.float32)
    ref.device = dev
    g_sum = ref.cumsum(g_in, C, cu_list)
    A = ref.kkt(k, beta, g_sum, C, cu_list)
    A_inv = ref.solve_tril(A, C, cu_list)
    w, u = ref.wy_fast(k, v, beta, A_inv, g_sum, C, cu_list)
    h_states, v_new, _ = ref.chunk_h(k, w, u, g_sum, C, cu_list)
    golden = ref.chunk_o(q, k, v_new, h_states, g_sum, C, cu_list) * scale
    torch.npu.synchronize()

    # Stage closures for the torch-family implementations (shared inputs).
    def torch_family_closures(impl):
        return {
            "cumsum":     lambda: impl.cumsum(g_in, C, cu_list),
            "kkt":        lambda: impl.kkt(k, beta, g_sum, C, cu_list),
            "solve_tril": lambda: impl.solve_tril(A, C, cu_list),
            "wy_fast":    lambda: impl.wy_fast(k, v, beta, A_inv, g_sum, C, cu_list),
            "chunk_h":    lambda: impl.chunk_h(k, w, u, g_sum, C, cu_list),
            "chunk_o":    lambda: impl.chunk_o(q, k, v_new, h_states, g_sum, C, cu_list),
        }

    times = {im: {} for im in IMPLS}
    errs = {}

    # 1-3: torch-family
    for name, impl, iters in [
        ("torch_ref", ref, ref_iters),
        ("torch_einsum", einsum_gdn, 10),
        ("pto_einsum", pto_gdn, 10),
    ]:
        cl = torch_family_closures(impl)
        for st in STAGES:
            times[name][st] = bench_npu(cl[st], warmup=2 if name == "torch_ref" else 3,
                                        iters=iters)
        out = impl.run_full_pipeline(q, k, v, g_in, beta, cu_list, H, Hg, scale, C=C)
        errs[name] = frob_rel(out, golden)
        torch.npu.synchronize()

    # 4: megagdn staged (fp16 buffers). Two independent fresh pipelines guard against
    # any residual nondeterminism; report the consistent error and warn if they disagree.
    e_reps = []
    for _ in range(2):
        m = MegaGDN(q, k, v, g_in, beta, cu_t, H, Hg, scale)
        e_reps.append(frob_rel(m.run_full_pipeline(), golden))
    errs["megagdn"] = min(e_reps)
    if max(e_reps) > 2 * min(e_reps) + 1e-3:
        print(f"  [warn] megagdn nondeterministic this config: frob_rel reps={e_reps}")
    mega = MegaGDN(q, k, v, g_in, beta, cu_t, H, Hg, scale)
    mega.run_full_pipeline()  # warmup populates buffers for per-stage timing
    for st in STAGES:
        times["megagdn"][st] = bench_npu(getattr(mega, st), warmup=3, iters=15)

    return times, errs, T


# ---------------------------------------------------------------------------
# Plot — grouped stacked bars
# ---------------------------------------------------------------------------
def plot(results, configs, impls, stages, title, out_png):
    fig, ax = plt.subplots(figsize=(max(9, 2.6 * len(configs) + 2), 6.2))
    cmap = plt.get_cmap("tab20")
    stage_colors = {st: cmap(i / max(1, len(STAGES))) for i, st in enumerate(STAGES)}

    n_imp = len(impls)
    group_w = 0.84
    bar_w = group_w / n_imp
    x = np.arange(len(configs))
    ymax = max(sum(results[c][im][s] for s in stages) for c in configs for im in impls)

    seen = set()
    for gi, cfg in enumerate(configs):
        for ii, im in enumerate(impls):
            bx = x[gi] - group_w / 2 + bar_w * (ii + 0.5)
            bottom = 0.0
            for st in stages:
                h = results[cfg][im][st]
                ax.bar(bx, h, bar_w * 0.92, bottom=bottom,
                       color=stage_colors[st], edgecolor="white", linewidth=0.3,
                       label=st if st not in seen else None)
                seen.add(st)
                bottom += h
            ax.text(bx, bottom + 0.01 * ymax, f"{bottom:.2f}", ha="center", va="bottom",
                    fontsize=6.5, rotation=90)
            ax.text(bx, -0.015 * ymax, IMPL_LABELS[im], ha="center", va="top",
                    fontsize=7.0, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}·L={l}\nT={n*l}" for (n, l) in configs])
    ax.tick_params(axis="x", length=0, pad=64)
    ax.set_ylabel("stage latency (ms, median)")
    ax.set_title(title)
    ax.legend(title="stage", ncol=2, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, ymax * 1.16)
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


def parse_configs(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        n, l = tok.lower().split("x")
        out.append((int(n), int(l)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--configs", default="1x512,2x512,1x1024,2x1024",
                    help="Comma list of NxL (N_seq x L_seg), e.g. 1x512,2x1024.")
    ap.add_argument("--H", type=int, default=32, help="Value head count.")
    ap.add_argument("--hg", type=int, default=16, help="Key head count Hg.")
    ap.add_argument("--ref-iters", type=int, default=3,
                    help="Timed iters for the (slow) torch reference.")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.npu.set_device(args.device)
    configs = parse_configs(args.configs)
    H, Hg = args.H, args.hg
    assert H % Hg == 0

    einsum_gdn = EinsumGDN(torch.float32); einsum_gdn.device = args.device
    pto_gdn = PtoGDN(torch.float32); pto_gdn.device = args.device

    results, all_errs = {}, {}
    print(f"Workloads: {configs}   H={H} Hg={Hg} D={D} C={C}")
    print("=" * 78)
    for (n, l) in configs:
        torch.npu.synchronize()
        torch.npu.empty_cache()
        t0 = time.perf_counter()
        times, errs, T = measure_config(n, l, H, Hg, einsum_gdn, pto_gdn, args.ref_iters)
        results[(n, l)] = times
        all_errs[(n, l)] = errs
        print(f"\nN={n} L={l} (T={T})   [{time.perf_counter()-t0:.1f}s]")
        hdr = "stage".ljust(12) + "".join(IMPL_LABELS[i].rjust(14) for i in IMPLS)
        print(hdr)
        for st in STAGES:
            row = st.ljust(12) + "".join(f"{times[i][st]:14.3f}" for i in IMPLS)
            print(row)
        tot = "TOTAL".ljust(12) + "".join(
            f"{sum(times[i].values()):14.3f}" for i in IMPLS)
        print(tot)
        print("frob_rel".ljust(12) + "".join(f"{errs[i]:14.2e}" for i in IMPLS))

    pto_gdn.cleanup()

    out_dir = Path(args.out_dir)
    json_path = out_dir / "gdn_4way_results.json"
    json_path.write_text(json.dumps({
        "H": H, "Hg": Hg, "D": D, "C": C,
        "configs": [list(c) for c in configs],
        "stages": STAGES, "impls": IMPLS,
        "times_ms": {f"{n}x{l}": results[(n, l)] for (n, l) in configs},
        "frob_rel": {f"{n}x{l}": all_errs[(n, l)] for (n, l) in configs},
    }, indent=2))
    print(f"\n[saved {json_path}]")

    sub = f"(H={H}, Hg={Hg}, D={D}, C={C})"
    # Only the contraction stages (where the einsum backend actually matters) for the
    # three NPU implementations — torch.einsum vs pto-einsum vs megagdn. The torch-glue
    # stages (cumsum, and especially the dominant solve_tril) are dropped: a full-pipeline
    # plot is swamped by solve_tril and tells you nothing about the contraction backend.
    contraction = ["kkt", "wy_fast", "chunk_h", "chunk_o"]
    comp = plot(results, configs, ["torch_einsum", "pto_einsum", "megagdn"], contraction,
                f"GDN contraction stages only (kkt+wy_fast+chunk_h+chunk_o)  {sub}",
                out_dir / "gdn_4way_contractions.png")
    print(f"[saved {comp}]")


if __name__ == "__main__":
    main()
