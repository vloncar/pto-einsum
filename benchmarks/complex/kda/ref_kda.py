"""Device-aware fp32 reference for the KDA (Kimi Delta Attention) pipeline.

Mirrors ``ref_gdn.RefGDN``: naive per-head Python loops, executed in
``self.dtype`` on ``self.device``.  Defaults to CPU (the double-precision
correctness oracle the bench cross-checks against); set ``self.device`` to an
NPU to produce the fp32 NPU golden plus the captured per-stage intermediates the
4-way benchmark feeds to every torch-family stage.

KDA is GDN with a *per-dimension* gate.  Where GDN carries one scalar decay per
(token, head) and applies it as an outer-product factor ``exp(g_i - g_j)``, KDA
carries a K-vector decay ``g`` of shape ``[B, T, HV, K]`` and bakes it straight
into the contraction operands (``k*exp(g)`` against ``k*exp(-g)``).  The four
contraction stages therefore use the *same* einsum equations as GDN — only the
gating differs — so the same pto-einsum direct-read fast paths fire.

Pipeline (matches tests/ref_kda.py in megagdn-pto, the e2e ground truth):
  cumsum -> kkt (L) -> solve_tril ((I+L)^{-1}) -> wy (u, w)
         -> chunk_h (state snapshots + v_corr) -> chunk_o (output)
"""
import os

import numpy as np
import torch

from utils import seq_ranges, total_chunks


class RefKDA:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.device = os.getenv("EINSUM_TEST_DEVICE", "cpu")

    # --- Stage 1: within-chunk per-dimension cumulative sum of g -------------
    def cumsum(self, g: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        """g: [B, T, HV, K] -> within-chunk cumulative sum along T."""
        gf = g.to(device=self.device, dtype=self.dtype)
        out = torch.zeros_like(gf)
        B, T, HV, K = gf.shape
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                out[:, s:e] = gf[:, s:e].cumsum(dim=1)
        return out

    # --- Stage 2: L matrix  L[r,c] = beta[r] * (k_r*exp(g_r))·(k_c*exp(-g_c)) -
    def kkt(
        self,
        k: torch.Tensor,
        beta: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ) -> torch.Tensor:
        """k: [B,T,H,K]  beta: [B,T,HV]  g_cumsum: [B,T,HV,K] -> L: [B,T,HV,cs]
        strictly lower triangular per chunk."""
        B, T, H, Kd = k.shape
        HV = beta.shape[2]
        grp = HV // H
        out = torch.zeros(B, T, HV, cs, dtype=self.dtype, device=self.device)
        kf = k.to(device=self.device, dtype=self.dtype)
        bf = beta.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                v = e - s
                ar = torch.arange(v, device=self.device)
                mask = (ar[:, None] > ar[None, :]).to(self.dtype)
                for h in range(HV):
                    hg = h // grp
                    kc = kf[0, s:e, hg, :]            # [v, K]
                    gc = gf[0, s:e, h, :]             # [v, K]
                    bc = bf[0, s:e, h]                # [v]
                    a_op = kc * torch.exp(gc)
                    b_op = kc * torch.exp(-gc)
                    blk = (a_op @ b_op.T) * bc[:, None]
                    out[0, s:e, h, :v] = blk * mask
        return out

    # --- Stage 3: (I + L)^{-1} per chunk (shared with GDN) -------------------
    def solve_tril(self, A: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        B, T, H, _ = A.shape
        out = torch.zeros(B, T, H, cs, dtype=self.dtype, device=self.device)
        Af = A.to(device=self.device, dtype=self.dtype)
        if self.device != "cpu":
            for bos, eos in seq_ranges(T, cu_seqlens):
                for j in range(0, eos - bos, cs):
                    s, e = bos + j, min(bos + j + cs, eos)
                    v = e - s
                    Ac = Af[0, s:e, :, :v].permute(1, 0, 2)  # [H, v, v]
                    eye = torch.eye(v, dtype=self.dtype, device=self.device).expand(H, v, v)
                    invM = torch.linalg.solve_triangular(
                        eye + Ac, eye, upper=False, unitriangular=True)
                    out[0, s:e, :, :v] = invM.permute(1, 0, 2)
            return out
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                v = e - s
                for h in range(H):
                    Ac = Af[0, s:e, h, :v]
                    M = np.linalg.inv((np.identity(v) + Ac.numpy()).astype(np.double))
                    out[0, s:e, h, :v] = torch.from_numpy(M).to(self.dtype)
        return out

    # --- Stage 4: u = A_inv @ (beta*v),  w = A_inv @ (beta*exp(g)*k) ---------
    def wy(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A_inv: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ):
        B, T, H, Kd = k.shape
        HV = v.shape[2]
        Vd = v.shape[-1]
        grp = HV // H
        u = torch.zeros(B, T, HV, Vd, dtype=self.dtype, device=self.device)
        w = torch.zeros(B, T, HV, Kd, dtype=self.dtype, device=self.device)
        kf = k.to(device=self.device, dtype=self.dtype)
        vf = v.to(device=self.device, dtype=self.dtype)
        bf = beta.to(device=self.device, dtype=self.dtype)
        Af = A_inv.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                v_len = e - s
                for h in range(HV):
                    hg = h // grp
                    A_invc = Af[0, s:e, h, :v_len]      # [v, v]
                    kc = kf[0, s:e, hg, :]
                    gc = gf[0, s:e, h, :]
                    vc = vf[0, s:e, h, :]
                    bc = bf[0, s:e, h, None]            # [v, 1]
                    u[0, s:e, h, :] = A_invc @ (vc * bc)
                    w[0, s:e, h, :] = A_invc @ (kc * torch.exp(gc) * bc)
        return u, w

    # --- Stage 5: sequential state pass (snapshots + v_corr) ----------------
    def chunk_h(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ):
        B, T, H, Kd = k.shape
        HV = w.shape[2]
        Vd = u.shape[-1]
        grp = HV // H
        kf = k.to(device=self.device, dtype=self.dtype)
        wf = w.to(device=self.device, dtype=self.dtype)
        uf = u.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)
        ranges = seq_ranges(T, cu_seqlens)
        tc = total_chunks(T, cs, cu_seqlens)
        s_snap = torch.zeros(tc, HV, Kd, Vd, dtype=self.dtype, device=self.device)
        v_corr = torch.zeros_like(uf)
        ci_base = 0
        for bos, eos in ranges:
            nc = (eos - bos + cs - 1) // cs
            for h in range(HV):
                hg = h // grp
                S = torch.zeros(Kd, Vd, dtype=self.dtype, device=self.device)
                for ci in range(nc):
                    s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                    gc = gf[0, s:e, h, :]              # [c, K]
                    g_total = gc[e - s - 1]            # [K]
                    kc = kf[0, s:e, hg, :]
                    s_snap[ci_base + ci, h] = S.clone()
                    vcorr = uf[0, s:e, h, :] - wf[0, s:e, h, :] @ S    # [c, V]
                    v_corr[0, s:e, h, :] = vcorr
                    k_rest = kc * torch.exp(g_total[None, :] - gc)     # [c, K]
                    S = torch.exp(g_total)[:, None] * S + k_rest.T @ vcorr
            ci_base += nc
        return s_snap, v_corr

    # --- Stage 6: output  o = (q*exp(g)) @ S + tril(q_eff @ k_eff^T) @ v_corr -
    def chunk_o(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v_corr: torch.Tensor,
        s_snap: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ) -> torch.Tensor:
        B, T, H, Kd = q.shape
        HV = v_corr.shape[2]
        Vd = v_corr.shape[-1]
        grp = HV // H
        qf = q.to(device=self.device, dtype=self.dtype)
        kf = k.to(device=self.device, dtype=self.dtype)
        vf = v_corr.to(device=self.device, dtype=self.dtype)
        sf = s_snap.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)
        o = torch.zeros(B, T, HV, Vd, dtype=self.dtype, device=self.device)
        ci_base = 0
        for bos, eos in seq_ranges(T, cu_seqlens):
            nc = (eos - bos + cs - 1) // cs
            for h in range(HV):
                hg = h // grp
                for ci in range(nc):
                    s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                    vlen = e - s
                    gc = gf[0, s:e, h, :]
                    q_eff = qf[0, s:e, hg, :] * torch.exp(gc)     # [c, K]
                    k_eff = kf[0, s:e, hg, :] * torch.exp(-gc)    # [c, K]
                    S = sf[ci_base + ci, h]                       # [K, V]
                    ar = torch.arange(vlen, device=self.device)
                    causal = (ar[:, None] >= ar[None, :]).to(self.dtype)
                    Aqk = (q_eff @ k_eff.T) * causal
                    o[0, s:e, h, :] = q_eff @ S + Aqk @ vf[0, s:e, h, :]
            ci_base += nc
        return o

    def run_full_pipeline(
        self, q, k, v, g_in, beta, cu_seqlens_list, H, HV, scale=1.0, C=128
    ):
        cu = cu_seqlens_list
        g_sum = self.cumsum(g_in, C, cu)
        L = self.kkt(k, beta, g_sum, C, cu)
        A_inv = self.solve_tril(L, C, cu)
        u, w = self.wy(k, v, beta, A_inv, g_sum, C, cu)
        s_snap, v_corr = self.chunk_h(k, w, u, g_sum, C, cu)
        o = self.chunk_o(q, k, v_corr, s_snap, g_sum, C, cu)
        return o * scale
