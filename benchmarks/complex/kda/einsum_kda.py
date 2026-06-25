"""KDA pipeline expressed with two-operand einsum contractions.

Direct analogue of ``einsum_gdn.EinsumGDN``: identical chunk-loop / fold glue,
the only swappable piece is ``self._einsum`` (overridden by ``PtoKDA`` to route
contractions through pto-einsum).  KDA differs from GDN purely in the gate: it is
a per-dimension log-decay ``g`` of shape ``[B, T, HV, K]`` baked into the
contraction operands, rather than GDN's scalar-per-head outer-product factor.
The four contraction equations are therefore identical to GDN's:

  kkt      : bihd, bjhd -> bihj     (NT  direct read)
  wy       : bihj, bjhd -> bihd     (NN-strided direct read)
  chunk_h  : bvhd, bhde -> bvhe  +  bvhd, bvhe -> bhde  (TN direct read)
  chunk_o  : bvhd, bhde -> bvhe ,  bihd, bjhd -> bihj ,  bihj, bjhd -> bihd

so the same pto-einsum direct-read fast paths fire as in the GDN benchmark.
"""
import os

import torch

from utils import seq_ranges, total_chunks


class EinsumKDA:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.device = os.getenv("EINSUM_TEST_DEVICE", "npu")
        # Max chunks folded into one batched einsum (see _batch_groups). Folding
        # ALL nc chunks at once materialises every chunk's O(C^2*HV) intermediates
        # simultaneously -> peak transient grows ~linearly in T and OOMs at large
        # T. Capping bounds peak memory while collapsing nc launches into
        # ceil(nc/cap). cap >= nc => single group, bit-exact / perf-identical.
        self.batch_cap = int(os.getenv("KDA_BATCH_CAP", "64"))

    def _batch_groups(self, Bn: int):
        cap = self.batch_cap
        step = Bn if not cap or cap <= 0 else min(Bn, cap)
        for g0 in range(0, Bn, step):
            yield g0, min(g0 + step, Bn)

    def _einsum(self, eq: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Two-operand contraction backend.  ``PtoKDA`` overrides this."""
        return torch.einsum(eq, a, b)

    @staticmethod
    def _regular_nc(T: int, cs: int, cu_seqlens=None):
        """Number of full chunks if every sequence length is a multiple of cs,
        else None (ragged layouts fall back to the per-chunk loop)."""
        for bos, eos in seq_ranges(T, cu_seqlens):
            if (eos - bos) % cs != 0:
                return None
        return T // cs

    # --- Stage 1: per-dimension within-chunk cumulative sum -----------------
    def cumsum(self, g: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        gf = g.to(device=self.device, dtype=self.dtype)
        out = torch.zeros_like(gf)
        T = gf.shape[1]
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                out[:, s:e] = gf[:, s:e].cumsum(dim=1)
        return out

    # --- Stage 2: L = tril(beta * (k*exp(g)) @ (k*exp(-g))^T) ----------------
    def kkt(
        self,
        k: torch.Tensor,
        beta: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ) -> torch.Tensor:
        B, T, H, Dd = k.shape
        HV = beta.shape[2]
        grp = HV // H
        kf = k.to(device=self.device, dtype=self.dtype)
        bf = beta.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)

        nc = self._regular_nc(T, cs, cu_seqlens)
        if nc is not None:
            Bn = B * nc
            kfb = kf.reshape(Bn, cs, H, Dd)
            gfb = gf.reshape(Bn, cs, HV, Dd)
            bfb = bf.reshape(Bn, cs, HV)
            ar = torch.arange(cs, device=self.device)
            mask = (ar[:, None] > ar[None, :])
            out = torch.empty(Bn, cs, HV, cs, dtype=self.dtype, device=self.device)
            for g0, g1 in self._batch_groups(Bn):
                kc_exp = kfb[g0:g1].repeat_interleave(grp, dim=2)
                gc = gfb[g0:g1]
                bc = bfb[g0:g1]
                a_op = kc_exp * torch.exp(gc)
                b_op = kc_exp * torch.exp(-gc)
                L = self._einsum("bihd, bjhd -> bihj", a_op, b_op)
                blk = L * bc.unsqueeze(-1)
                out[g0:g1] = blk * mask[None, :, None, :]
            return out.reshape(B, T, HV, cs)

        out = torch.zeros(B, T, HV, cs, dtype=self.dtype, device=self.device)
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                v = e - s
                kc_exp = kf[:, s:e, :, :].repeat_interleave(grp, dim=2)
                gc = gf[:, s:e, :, :]
                bc = bf[:, s:e, :]
                a_op = kc_exp * torch.exp(gc)
                b_op = kc_exp * torch.exp(-gc)
                L = self._einsum("bihd, bjhd -> bihj", a_op, b_op)
                blk = L * bc.unsqueeze(-1)
                ar = torch.arange(v, device=self.device)
                mask = (ar[:, None] > ar[None, :])
                out[:, s:e, :, :v] = blk * mask[None, :, None, :]
        return out

    # --- Stage 3: (I + L)^{-1} per chunk (shared with GDN) ------------------
    def solve_tril(self, A: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        B, T, H, _ = A.shape
        out = torch.zeros(B, T, H, cs, dtype=self.dtype, device=self.device)
        Af = A.to(device=self.device, dtype=self.dtype)
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                v = e - s
                Ac = Af[:, s:e, :, :v].permute(0, 2, 1, 3)
                I = torch.eye(v, dtype=self.dtype, device=self.device).view(1, 1, v, v)
                M = I + Ac
                if self.device == "cpu":
                    invM = torch.linalg.inv(M.double()).to(self.dtype)
                else:
                    invM = torch.linalg.solve_triangular(
                        M, I.expand_as(M), upper=False, unitriangular=True)
                out[:, s:e, :, :v] = invM.permute(0, 2, 1, 3)
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
        kf = k.to(device=self.device, dtype=self.dtype)
        vf = v.to(device=self.device, dtype=self.dtype)
        bf = beta.to(device=self.device, dtype=self.dtype)
        Af = A_inv.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)

        nc = self._regular_nc(T, cs, cu_seqlens)
        if nc is not None:
            # Both outputs contract the same A_inv against different RHS (kb, vb),
            # so stack the RHS along the feature axis and issue ONE batched einsum.
            Bn = B * nc
            Afb = Af.reshape(Bn, cs, HV, cs)
            bfb = bf.reshape(Bn, cs, HV)
            gfb = gf.reshape(Bn, cs, HV, Kd)
            vfb = vf.reshape(Bn, cs, HV, Vd)
            kfb = kf.reshape(Bn, cs, H, Kd)
            w = torch.empty(Bn, cs, HV, Kd, dtype=self.dtype, device=self.device)
            u = torch.empty(Bn, cs, HV, Vd, dtype=self.dtype, device=self.device)
            for g0, g1 in self._batch_groups(Bn):
                bfold = bfb[g0:g1].unsqueeze(-1)
                vb = vfb[g0:g1] * bfold
                kc_exp = kfb[g0:g1].repeat_interleave(grp, dim=2)
                kb = kc_exp * torch.exp(gfb[g0:g1]) * bfold
                kv = torch.cat([kb, vb], dim=-1)
                wu = self._einsum("bihj, bjhd -> bihd", Afb[g0:g1], kv)
                w[g0:g1] = wu[..., :Kd]
                u[g0:g1] = wu[..., Kd:]
            return u.reshape(B, T, HV, Vd), w.reshape(B, T, HV, Kd)

        w = torch.zeros(B, T, HV, Kd, dtype=self.dtype, device=self.device)
        u = torch.zeros(B, T, HV, Vd, dtype=self.dtype, device=self.device)
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                valid = e - s
                Ab = Af[:, s:e, :, :valid]
                gc = gf[:, s:e, :, :]
                bfold = bf[:, s:e, :].unsqueeze(-1)
                vb = vf[:, s:e, :, :] * bfold
                kc_exp = kf[:, s:e, :, :].repeat_interleave(grp, dim=2)
                kb = kc_exp * torch.exp(gc) * bfold
                u[:, s:e, :, :] = self._einsum("bihj, bjhd -> bihd", Ab, vb)
                w[:, s:e, :, :] = self._einsum("bihj, bjhd -> bihd", Ab, kb)
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
            S = torch.zeros(B, HV, Kd, Vd, dtype=self.dtype, device=self.device)
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                gc = gf[:, s:e, :, :]                  # [B, c, HV, K]
                g_total = gc[:, -1:, :, :]             # [B, 1, HV, K]
                s_snap[ci_base + ci, :] = S[0].clone()
                wc = wf[:, s:e, :, :]
                uc = uf[:, s:e, :, :]
                ws = self._einsum("bvhd, bhde -> bvhe", wc, S)
                vcorr = uc - ws
                v_corr[:, s:e, :, :] = vcorr
                kc_exp = kf[:, s:e, :, :].repeat_interleave(grp, dim=2)
                k_rest = kc_exp * torch.exp(g_total - gc)    # [B, c, HV, K]
                kv = self._einsum("bvhd, bvhe -> bhde", k_rest, vcorr)   # [B, HV, K, V]
                scale_S = torch.exp(g_total).squeeze(1).unsqueeze(-1)    # [B, HV, K, 1]
                S = scale_S * S + kv
            ci_base += nc
        return s_snap, v_corr

    # --- Stage 6: o = (q*exp(g)) @ S + tril(q_eff @ k_eff^T) @ v_corr --------
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
        hf = s_snap.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)
        ranges = seq_ranges(T, cu_seqlens)

        nc = self._regular_nc(T, cs, cu_seqlens)
        if nc is not None:
            Bn = B * nc
            qfb = qf.reshape(Bn, cs, H, Kd)
            kfb = kf.reshape(Bn, cs, H, Kd)
            vfb = vf.reshape(Bn, cs, HV, Vd)
            gfb = gf.reshape(Bn, cs, HV, Kd)
            hfb = hf.reshape(Bn, HV, Kd, Vd)
            ar = torch.arange(cs, device=self.device)
            causal = (ar[:, None] >= ar[None, :]).to(self.dtype)
            causal_exp = causal.unsqueeze(0).unsqueeze(2)        # [1, c, 1, c]
            o = torch.empty(Bn, cs, HV, Vd, dtype=self.dtype, device=self.device)
            for g0, g1 in self._batch_groups(Bn):
                gc = gfb[g0:g1]
                q_eff = qfb[g0:g1].repeat_interleave(grp, dim=2) * torch.exp(gc)
                k_eff = kfb[g0:g1].repeat_interleave(grp, dim=2) * torch.exp(-gc)
                inter = self._einsum("bvhd, bhde -> bvhe", q_eff, hfb[g0:g1])
                qk = self._einsum("bihd, bjhd -> bihj", q_eff, k_eff)
                intra = self._einsum("bihj, bjhd -> bihd", qk * causal_exp, vfb[g0:g1])
                o[g0:g1] = inter + intra
            return o.reshape(B, T, HV, Vd)

        o = torch.zeros(B, T, HV, Vd, dtype=self.dtype, device=self.device)
        ci_base = 0
        for bos, eos in ranges:
            nc = (eos - bos + cs - 1) // cs
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                vlen = e - s
                gc = gf[:, s:e, :, :]
                q_eff = qf[:, s:e, :, :].repeat_interleave(grp, dim=2) * torch.exp(gc)
                k_eff = kf[:, s:e, :, :].repeat_interleave(grp, dim=2) * torch.exp(-gc)
                h_chunk = hf[ci_base + ci].unsqueeze(0)
                inter = self._einsum("bvhd, bhde -> bvhe", q_eff, h_chunk)
                qk = self._einsum("bihd, bjhd -> bihj", q_eff, k_eff)
                ar = torch.arange(vlen, device=self.device)
                causal = (ar[:, None] >= ar[None, :]).to(self.dtype)
                causal_exp = causal.unsqueeze(0).unsqueeze(2)
                intra = self._einsum("bihj, bjhd -> bihd", qk * causal_exp, vf[:, s:e, :, :])
                o[:, s:e, :, :] = inter + intra
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
