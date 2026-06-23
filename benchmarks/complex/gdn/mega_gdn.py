"""megagdn-pto staged adapter for the 4-way GDN benchmark.

Wraps the hand-written PTO chunk-GDN kernels (``megagdn_pto.kernel_libs`` +
``fast_inverse``) behind the same six-stage interface the torch / einsum / pto-einsum
implementations expose, so every stage is timed individually and the full pipeline
can be checked against the reference. This is the *staged* path (one launch per
stage) — directly comparable stage-by-stage. The fully fused single-launch
``run_mega_kernel`` is reported separately as an end-to-end lower bound.

All kernels are fp16 (gates fp32), C=128. Inputs follow megagdn layout:
  q,k [1,T,Hg,D] fp16 ; v [1,T,H,D] fp16 ; beta [1,T,H] fp16 ; g_in [1,T,H] fp32.
"""
import os
import sys

import torch

# megagdn-pto is a sibling repo of pto-einsum and is not pip-installed; add its root
# to the path (override with MEGAGDN_PTO_PATH).
_MEGAGDN_ROOT = os.getenv(
    "MEGAGDN_PTO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../megagdn-pto")),
)
if _MEGAGDN_ROOT not in sys.path:
    sys.path.insert(0, _MEGAGDN_ROOT)

from megagdn_pto.compile import BLOCK_DIM
from megagdn_pto.kernel_libs import (
    transpose_gates, transpose_beta, total_chunks,
    run_chunk_cumsum, run_scaled_dot_kkt, run_wy_fast, run_chunk_h, run_chunk_o,
)
from megagdn_pto.fast_inverse import load_tri_inverse, solve_tril

D = 128
C = 128


class MegaGDN:
    """Staged megagdn-pto pipeline. One instance per (T, H, Hg, cu_seqlens) shape."""

    STAGES = ("cumsum", "kkt", "solve_tril", "wy_fast", "chunk_h", "chunk_o")

    def __init__(self, q, k, v, g_in, beta, cu_seqlens, H, Hg, scale):
        self.dev = q.device
        self.H, self.Hg, self.scale = H, Hg, scale
        self.T = q.shape[1]
        self.cu = cu_seqlens.to(torch.int32) if cu_seqlens is not None else \
            torch.tensor([0, self.T], dtype=torch.int32, device=self.dev)
        self.N_seq = int(self.cu.numel()) - 1
        self.tc = total_chunks(self.N_seq, self.T, C, self.cu)
        self.tri_inv = load_tri_inverse()

        # Inputs (fp16 activations, fp32 gates) — same values as the reference.
        self.q = q.to(torch.float16)
        self.k = k.to(torch.float16)
        self.v = v.to(torch.float16)
        self.g_in = g_in.to(torch.float32)
        self.beta = beta.to(torch.float16)

        # Persistent buffers / intermediates (allocated once; reused every stage).
        # All intermediates are zero-initialised: a stage that only writes the valid
        # (non-padding) region of a buffer must not expose freed-memory garbage to the
        # next stage, otherwise results go nondeterministic across runs/configs.
        T, H = self.T, self.H
        self.g_sum = torch.zeros(1, T, H, device=self.dev, dtype=torch.float32)
        self.g_t = torch.zeros(H, T, device=self.dev, dtype=torch.float32)
        self.beta_t = torch.zeros(H, T, device=self.dev, dtype=torch.float16)
        self.A = torch.zeros(1, T, H, C, device=self.dev, dtype=torch.float16)
        self.A_inv = torch.zeros(1, T, H, C, device=self.dev, dtype=torch.float16)
        self.w = torch.zeros(1, T, H, D, device=self.dev, dtype=torch.float16)
        self.u = torch.zeros(1, T, H, D, device=self.dev, dtype=torch.float16)
        self.s = torch.zeros(self.tc * H, D, D, device=self.dev, dtype=torch.float16)
        self.v_new = torch.zeros(1, T, H, D, device=self.dev, dtype=torch.float16)
        self.fs = torch.zeros(self.N_seq * H, D, D, device=self.dev, dtype=torch.float16)
        self.o = torch.zeros(1, T, H, D, device=self.dev, dtype=torch.float16)
        self.msk_l = torch.tril(torch.ones(C, C, device=self.dev), diagonal=-1).float()
        self.msk_f = torch.tril(torch.ones(C, C, device=self.dev), diagonal=0).float()

    # --- individual stages (each a single kernel launch, in-place into buffers) ---
    def cumsum(self):
        run_chunk_cumsum(self.g_in, self.g_sum, chunk_size=C,
                         cu_seqlens=self.cu, batch_size_override=self.N_seq)
        # Gate transposes feed kkt/wy/h/o; recompute here so cumsum output is consistent.
        self.g_t = transpose_gates(self.g_sum)
        self.beta_t = transpose_beta(self.beta)

    def kkt(self):
        run_scaled_dot_kkt(self.k, self.beta, self.g_sum, self.msk_l, self.A,
                           g_t=self.g_t, beta_t=self.beta_t,
                           chunk_size=C, cu_seqlens=self.cu,
                           batch_size_override=self.N_seq, key_heads=self.Hg)

    def solve_tril(self):
        self.A_inv = solve_tril(self.A, self.cu, C, self.H, self.tri_inv)

    def wy_fast(self):
        run_wy_fast(self.k, self.v, self.beta, self.g_sum, self.A_inv, self.w, self.u,
                    g_t=self.g_t, beta_t=self.beta_t, chunk_size=C,
                    cu_seqlens=self.cu, batch_size_override=self.N_seq, key_heads=self.Hg)

    def chunk_h(self):
        run_chunk_h(self.k, self.w, self.u, self.g_sum, self.s, self.v_new, self.fs,
                    g_t=self.g_t, chunk_size=C, cu_seqlens=self.cu,
                    batch_size_override=self.N_seq, key_heads=self.Hg)

    def chunk_o(self):
        run_chunk_o(self.q, self.k, self.v_new, self.s, self.g_sum, self.msk_f, self.o,
                    g_t=self.g_t, chunk_size=C, cu_seqlens=self.cu,
                    batch_size_override=self.N_seq, key_heads=self.Hg)

    def run_full_pipeline(self):
        """Run all six stages in order and return the scaled output [1,T,H,D].

        Synchronizes between stages, mirroring the canonical staged usage
        (``bench_gdn_kernels.run_staged``); some stages have host-side launch logic
        (e.g. solve_tril tile counting) that expects the previous stage materialized.
        """
        for st in self.STAGES:
            getattr(self, st)()
            torch.npu.synchronize()
        return self.o.float() * self.scale
