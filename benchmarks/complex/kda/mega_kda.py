"""megagdn-pto staged adapter for the 4-way KDA benchmark.

Wraps the hand-written PTO chunk-KDA kernels (``megagdn_pto.kda_kernel_libs`` +
``fast_inverse``) behind the same six-stage interface the torch / einsum /
pto-einsum implementations expose, so every stage is timed individually and the
full pipeline can be checked against the reference.  This is the *staged* path
(one launch per stage), directly comparable stage-by-stage.

All kernels are fp16, C=128.  KDA gates are per-dimension ``[1, T, HV, K]`` (vs
GDN's scalar ``[1, T, H]``), and q/k are GQA-expanded to HV heads before the
kernels (the KDA runners expect already-expanded operands).  Query scale is
applied to q up front (mirrors test_kda_e2e.py) so the fp16 magnitudes match the
validated e2e numerics.
"""
import os
import sys

import torch

# megagdn-pto is a sibling repo of pto-einsum and is not pip-installed; add its
# root to the path (override with MEGAGDN_PTO_PATH).
_MEGAGDN_ROOT = os.getenv(
    "MEGAGDN_PTO_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../megagdn-pto")),
)
if _MEGAGDN_ROOT not in sys.path:
    sys.path.insert(0, _MEGAGDN_ROOT)

from megagdn_pto.kernel_libs import total_chunks
from megagdn_pto.fast_inverse import load_tri_inverse, solve_tril
from megagdn_pto.kda_kernel_libs import (
    run_gate_cumsum_kda,
    run_kkt_kda,
    run_wy_kda,
    run_chunk_h_kda,
    run_chunk_o_kda,
)

D = 128
C = 128


class MegaKDA:
    """Staged megagdn-pto KDA pipeline. One instance per (T, H, HV, cu) shape."""

    STAGES = ("cumsum", "kkt", "solve_tril", "wy", "chunk_h", "chunk_o")

    def __init__(self, q, k, v, g_in, beta, cu_seqlens, H, HV, scale):
        self.dev = q.device
        self.H, self.HV, self.scale = H, HV, scale
        self.T = q.shape[1]
        G = HV // H
        self.cu = cu_seqlens.to(torch.int32) if cu_seqlens is not None else \
            torch.tensor([0, self.T], dtype=torch.int32, device=self.dev)
        self.N_seq = int(self.cu.numel()) - 1
        self.tc = total_chunks(self.N_seq, self.T, C, self.cu)
        self.tri_inv = load_tri_inverse()
        self.stream = torch.npu.current_stream()._as_parameter_

        # Inputs (fp16).  q/k GQA-expanded to HV; q pre-scaled (matches e2e).
        self.q = (q.to(torch.float16).repeat_interleave(G, dim=2) * scale).contiguous()
        self.k = k.to(torch.float16).repeat_interleave(G, dim=2).contiguous()
        self.v = v.to(torch.float16)
        self.g_in = g_in.to(torch.float16)          # [1, T, HV, K]
        self.beta = beta.to(torch.float16)          # [1, T, HV]

        # Persistent buffers / intermediates (allocated once, zero-initialised so a
        # stage that writes only the valid region never exposes freed-memory garbage
        # to the next stage — otherwise results go nondeterministic across runs).
        T, HV_ = self.T, self.HV
        self.g_sum = torch.zeros(1, T, HV_, D, device=self.dev, dtype=torch.float16)
        self.L = torch.zeros(1, T, HV_, C, device=self.dev, dtype=torch.float16)
        self.A_inv = torch.zeros(1, T, HV_, C, device=self.dev, dtype=torch.float16)
        self.u = torch.zeros(1, T, HV_, D, device=self.dev, dtype=torch.float16)
        self.w = torch.zeros(1, T, HV_, D, device=self.dev, dtype=torch.float16)
        self.s = torch.zeros(self.tc, HV_, D, D, device=self.dev, dtype=torch.float16)
        self.v_corr = torch.zeros(1, T, HV_, D, device=self.dev, dtype=torch.float16)
        self.o = torch.zeros(1, T, HV_, D, device=self.dev, dtype=torch.float16)

    # --- individual stages (each a single kernel launch into the buffers) ---
    def cumsum(self):
        run_gate_cumsum_kda(self.g_in, self.g_sum, stream=self.stream, chunk_size=C,
                            cu_seqlens=self.cu, batch_size_override=self.N_seq)

    def kkt(self):
        run_kkt_kda(self.k, self.g_sum, self.beta, self.L, stream=self.stream,
                    chunk_size=C, cu_seqlens=self.cu, batch_size_override=self.N_seq)

    def solve_tril(self):
        self.A_inv = solve_tril(self.L, self.cu, C, self.HV, self.tri_inv)

    def wy(self):
        run_wy_kda(self.k, self.v, self.g_sum, self.beta, self.A_inv, self.u, self.w,
                   stream=self.stream, chunk_size=C, cu_seqlens=self.cu,
                   batch_size_override=self.N_seq)

    def chunk_h(self):
        run_chunk_h_kda(self.k, self.w, self.u, self.g_sum, self.s, self.v_corr,
                        stream=self.stream, chunk_size=C, cu_seqlens=self.cu,
                        batch_size_override=self.N_seq)

    def chunk_o(self):
        run_chunk_o_kda(self.q, self.k, self.v_corr, self.s, self.g_sum, self.o,
                        stream=self.stream, chunk_size=C, cu_seqlens=self.cu,
                        batch_size_override=self.N_seq)

    def run_full_pipeline(self):
        """Run all six stages in order and return the output [1, T, HV, V].

        q is already scaled up front, so no extra scale is applied here (mirrors
        test_kda_e2e.py).  Synchronizes between stages — some have host-side launch
        logic (solve_tril tile counting) that expects the previous stage realized.
        """
        for st in self.STAGES:
            getattr(self, st)()
            torch.npu.synchronize()
        return self.o.float()
