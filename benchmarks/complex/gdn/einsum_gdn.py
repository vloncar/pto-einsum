import os
import torch
from utils import seq_ranges, total_chunks

def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))

class EinsumGDN:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.device = os.getenv("EINSUM_TEST_DEVICE", "npu")
        # Max chunks folded into one batched einsum (see _batch_groups). Folding ALL
        # nc chunks at once materialises every chunk's O(C^2*H) intermediates
        # simultaneously -> peak transient grows ~linearly in T and OOMs at large T.
        # Capping the group bounds peak memory to ~cap chunks' worth while still
        # collapsing nc launches into ceil(nc/cap). cap >= nc => single group =
        # bit-exact, perf-identical to the unblocked fold. 0/None => unbounded.
        self.batch_cap = int(os.getenv("GDN_BATCH_CAP", "64"))

    def _batch_groups(self, Bn: int):
        """Yield [g0, g1) slices over the folded batch axis (B*nc), at most
        batch_cap rows each. A single (0, Bn) group when batch_cap covers Bn."""
        cap = self.batch_cap
        step = Bn if not cap or cap <= 0 else min(Bn, cap)
        for g0 in range(0, Bn, step):
            yield g0, min(g0 + step, Bn)

    def _einsum(self, eq: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Two-operand contraction backend. Subclasses (e.g. the pto-einsum GDN)
        override this to swap torch.einsum for a different einsum implementation
        while reusing the exact same pipeline glue."""
        return torch.einsum(eq, a, b)

    @staticmethod
    def _regular_nc(T: int, cs: int, cu_seqlens=None):
        """Number of full chunks if every sequence length is a multiple of the
        chunk size cs, else None.

        When regular, T partitions into uniform cs-blocks that never cross a
        sequence boundary, so the host chunk loop in the chunk-independent stages
        (kkt, wy_fast, chunk_o) can be folded into the leading batch axis: each
        stage then issues ONE batched einsum (b = B*nc) instead of one per chunk,
        collapsing nc launches into a single launch in the pto-einsum backend.
        Ragged layouts (partial last chunk, mixed seqlens) return None and fall
        back to the per-chunk loop, which stays bit-for-bit unchanged."""
        for bos, eos in seq_ranges(T, cu_seqlens):
            if (eos - bos) % cs != 0:
                return None
        return T // cs

    def cumsum(self, g: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        B, T, H = g.shape
        gf = g.to(device=self.device, dtype=self.dtype)
        out = torch.zeros_like(gf)
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                out[:, s:e, :] = gf[:, s:e, :].cumsum(dim=1)
        return out

    def kkt(
        self,
        k: torch.Tensor,
        beta: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ) -> torch.Tensor:
        B, T, Hg, Dd = k.shape
        H = beta.shape[2]
        grp = H // Hg
        kf = k.to(device=self.device, dtype=self.dtype)
        bf = beta.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)

        nc = self._regular_nc(T, cs, cu_seqlens)
        if nc is not None:
            # Fold the chunk loop into the leading batch axis: one batched einsum
            # (b = B*nc) replaces nc per-chunk calls. Same per-chunk math, same
            # [b, i, h, j] equation — only the batch is larger. Processed in groups
            # of <= batch_cap chunks so peak transient stays bounded at large T; the
            # repeat_interleave/glue run per-group on the slice, never the whole fold.
            Bn = B * nc
            kfb = kf.reshape(Bn, cs, Hg, Dd)
            gfb = gf.reshape(Bn, cs, H)
            bfb = bf.reshape(Bn, cs, H)
            mask = torch.arange(cs, device=self.device)[:, None] > torch.arange(cs, device=self.device)[None, :]
            out = torch.empty(Bn, cs, H, cs, dtype=self.dtype, device=self.device)
            for g0, g1 in self._batch_groups(Bn):
                kc_exp = kfb[g0:g1].repeat_interleave(grp, dim=2)
                gc = gfb[g0:g1]
                bc = bfb[g0:g1]

                qk = self._einsum("bihd, bjhd -> bihj", kc_exp, kc_exp)
                diff_g = gc.unsqueeze(-1) - gc.permute(0, 2, 1).unsqueeze(1)
                exp_g = _safe_exp(diff_g)
                blk = qk * exp_g * bc.unsqueeze(-1)
                out[g0:g1] = blk * mask[None, :, None, :]
            return out.reshape(B, T, H, cs)

        out = torch.zeros(B, T, H, cs, dtype=self.dtype, device=self.device)
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                v = e - s
                kc = kf[:, s:e, :, :]
                gc = gf[:, s:e, :]
                bc = bf[:, s:e, :]

                kc_exp = kc.repeat_interleave(grp, dim=2)
                
                # Emit the contraction directly in [b, i, h, j] order (head h second,
                # key-row j innermost) instead of [b, i, j, h]+permute. With j (=free1)
                # innermost the einsum output is fusible, so the Cube stores each tile
                # straight to `out` and drops the Phase C transpose; it also folds away
                # the trailing .permute(0,1,3,2) the [b,i,j,h] formulation needed. The
                # gate / beta / causal-mask factors are built in the same [b,i,h,j] order.
                qk = self._einsum("bihd, bjhd -> bihj", kc_exp, kc_exp)
                diff_g = gc.unsqueeze(-1) - gc.permute(0, 2, 1).unsqueeze(1)
                exp_g = _safe_exp(diff_g)

                blk = qk * exp_g * bc.unsqueeze(-1)
                mask = torch.arange(v, device=self.device)[:, None] > torch.arange(v, device=self.device)[None, :]

                out[:, s:e, :, :v] = blk * mask[None, :, None, :]
        return out

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
                    # CPU reference path: LAPACK double-precision inverse (test parity).
                    invM = torch.linalg.inv(M.double()).to(self.dtype)
                else:
                    # NPU: linalg.inv falls back to host; M is unit lower-triangular, so a
                    # native batched triangular solve against I gives the same inverse.
                    invM = torch.linalg.solve_triangular(
                        M, I.expand_as(M), upper=False, unitriangular=True
                    )

                out[:, s:e, :, :v] = invM.permute(0, 2, 1, 3)
        return out

    def wy_fast(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        A_inv: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ):
        B, T, Hg, Kd = k.shape
        H = v.shape[2]
        grp = H // Hg

        kf = k.to(device=self.device, dtype=self.dtype)
        vf = v.to(device=self.device, dtype=self.dtype)
        bf = beta.to(device=self.device, dtype=self.dtype)
        Af = A_inv.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)

        nc = self._regular_nc(T, cs, cu_seqlens)
        if nc is not None:
            # Fold the chunk loop into the leading batch axis. Both outputs (w, u)
            # contract the same LHS Ab against different RHS (kb, vb), so stack the
            # RHS along the feature axis and issue ONE batched einsum (b = B*nc,
            # free d = Kd+Dv) instead of 2*nc per-chunk calls — fewer launches and
            # a single larger matmul for the Cube to tile. Grouped by batch_cap so
            # peak transient stays bounded at large T (one stacked matmul per group).
            Bn = B * nc
            Dv = v.shape[-1]
            Afb = Af.reshape(Bn, cs, H, cs)                        # [B*nc, i, H, j]
            bfb = bf.reshape(Bn, cs, H)
            gfb = gf.reshape(Bn, cs, H)
            vfb = vf.reshape(Bn, cs, H, Dv)
            kfb = kf.reshape(Bn, cs, Hg, Kd)
            w = torch.empty(Bn, cs, H, Kd, dtype=self.dtype, device=self.device)
            u = torch.empty(Bn, cs, H, Dv, dtype=self.dtype, device=self.device)
            for g0, g1 in self._batch_groups(Bn):
                bfold = bfb[g0:g1].unsqueeze(-1)
                vb = vfb[g0:g1] * bfold                            # [g, j, H, D]
                kc_exp = kfb[g0:g1].repeat_interleave(grp, dim=2)
                kb = kc_exp * bfold * torch.exp(gfb[g0:g1]).unsqueeze(-1)  # [g, j, H, D]

                kv = torch.cat([kb, vb], dim=-1)                   # [g, j, H, Kd+Dv]
                wu = self._einsum("bihj, bjhd -> bihd", Afb[g0:g1], kv)    # [g, i, H, Kd+Dv]
                w[g0:g1] = wu[..., :Kd]
                u[g0:g1] = wu[..., Kd:]
            return w.reshape(B, T, H, Kd), u.reshape(B, T, H, Dv)

        w = torch.zeros(B, T, H, Kd, dtype=self.dtype, device=self.device)
        u = torch.zeros(B, T, H, v.shape[-1], dtype=self.dtype, device=self.device)
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                valid = e - s

                # Keep every operand in its natural [B, chunk, H, D] layout and let the
                # einsum index labels carry the transpose: i = query row, j = key row
                # (contracted), h = head, d = feature. This drops the four explicit
                # .permute(0,2,1,3) round-trips the [B,H,i,j] formulation needed.
                Ab = Af[:, s:e, :, :valid]                       # [B, i, H, j]
                gc = gf[:, s:e, :]
                vb = vf[:, s:e, :, :] * bf[:, s:e, :].unsqueeze(-1)        # [B, j, H, D]

                kc = kf[:, s:e, :, :]
                kc_exp = kc.repeat_interleave(grp, dim=2)
                kb = kc_exp * bf[:, s:e, :].unsqueeze(-1) * torch.exp(gc).unsqueeze(-1)  # [B, j, H, D]

                u[:, s:e, :, :] = self._einsum("bihj, bjhd -> bihd", Ab, vb)
                w[:, s:e, :, :] = self._einsum("bihj, bjhd -> bihd", Ab, kb)

        return w, u

    def chunk_h(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ):
        B, T, Hg, Dd = k.shape
        H = w.shape[2]
        grp = H // Hg
        
        kf = k.to(device=self.device, dtype=self.dtype)
        wf = w.to(device=self.device, dtype=self.dtype)
        uf = u.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)
        
        ranges = seq_ranges(T, cu_seqlens)
        tc = total_chunks(T, cs, cu_seqlens)
        
        h_out = torch.zeros(tc, H, Dd, Dd, dtype=self.dtype, device=self.device)
        v_new = torch.zeros_like(uf)
        final = torch.zeros(len(ranges), H, Dd, Dd, dtype=self.dtype, device=self.device)
        
        ci_base = 0
        for si, (bos, eos) in enumerate(ranges):
            nc = (eos - bos + cs - 1) // cs
            S = torch.zeros(B, H, Dd, Dd, dtype=self.dtype, device=self.device)
            
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                gc = gf[:, s:e, :]
                gl = gc[:, -1:, :]
                
                h_out[ci_base + ci, :] = S[0].clone()
                
                wc = wf[:, s:e, :, :]
                uc = uf[:, s:e, :, :]
                
                ws = self._einsum("bvhd, bhde -> bvhe", wc, S)
                vc = uc - ws
                v_new[:, s:e, :, :] = vc
                
                kc = kf[:, s:e, :, :]
                kc_exp = kc.repeat_interleave(grp, dim=2)
                
                exp_diff = torch.exp(gl - gc).unsqueeze(-1)
                vc_scaled = vc * exp_diff
                
                kv = self._einsum("bvhd, bvhe -> bhde", kc_exp, vc_scaled)
                S = torch.exp(gl).permute(0, 2, 1).unsqueeze(-1) * S + kv
                
            final[si, :] = S[0]
            ci_base += nc
            
        return h_out, v_new, final

    def chunk_o(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v_new: torch.Tensor,
        h_states: torch.Tensor,
        g_cumsum: torch.Tensor,
        cs: int,
        cu_seqlens=None,
    ) -> torch.Tensor:
        B, T, Hg, Dd = q.shape
        H = v_new.shape[2]
        grp = H // Hg
        
        qf = q.to(device=self.device, dtype=self.dtype)
        kf = k.to(device=self.device, dtype=self.dtype)
        vf = v_new.to(device=self.device, dtype=self.dtype)
        hf = h_states.to(device=self.device, dtype=self.dtype)
        gf = g_cumsum.to(device=self.device, dtype=self.dtype)
        
        ranges = seq_ranges(T, cu_seqlens)

        nc = self._regular_nc(T, cs, cu_seqlens)
        if nc is not None:
            # Fold the chunk loop into the leading batch axis: three batched
            # einsums (b = B*nc) replace 3*nc per-chunk calls. The per-chunk state
            # h_states[ci] is gathered by reshaping its global-chunk axis to batch.
            # Grouped by batch_cap so peak transient stays bounded at large T.
            Bn = B * nc
            qfb = qf.reshape(Bn, cs, Hg, Dd)
            kfb = kf.reshape(Bn, cs, Hg, Dd)
            vfb = vf.reshape(Bn, cs, H, Dd)
            gfb = gf.reshape(Bn, cs, H)
            hfb = hf.reshape(Bn, H, Dd, Dd)
            causal = torch.arange(cs, device=self.device)[:, None] >= torch.arange(cs, device=self.device)[None, :]
            causal_exp = causal.unsqueeze(0).unsqueeze(2).to(self.dtype)
            o = torch.empty(Bn, cs, H, Dd, dtype=self.dtype, device=self.device)
            for g0, g1 in self._batch_groups(Bn):
                qc_exp = qfb[g0:g1].repeat_interleave(grp, dim=2)
                kc_exp = kfb[g0:g1].repeat_interleave(grp, dim=2)
                vc = vfb[g0:g1]
                gc = gfb[g0:g1]

                qh = self._einsum("bvhd, bhde -> bvhe", qc_exp, hfb[g0:g1])
                inter = qh * torch.exp(gc).unsqueeze(-1)

                qk = self._einsum("bihd, bjhd -> bihj", qc_exp, kc_exp)
                diff_g = gc.unsqueeze(2) - gc.unsqueeze(1)
                diff_g = diff_g.permute(0, 1, 3, 2)
                gate = torch.exp(torch.minimum(diff_g, torch.zeros_like(diff_g)))
                qk_gated = qk * gate * causal_exp

                intra = self._einsum("bihj, bjhd -> bihd", qk_gated, vc)
                o[g0:g1] = inter + intra
            return o.reshape(B, T, H, Dd)

        o = torch.zeros(B, T, H, Dd, dtype=self.dtype, device=self.device)
        ci_base = 0
        for bos, eos in ranges:
            nc = (eos - bos + cs - 1) // cs
            for ci in range(nc):
                s, e = bos + ci * cs, min(bos + (ci + 1) * cs, eos)
                vlen = e - s
                qc = qf[:, s:e, :, :]
                kc = kf[:, s:e, :, :]
                vc = vf[:, s:e, :, :]
                gc = gf[:, s:e, :]
                
                qc_exp = qc.repeat_interleave(grp, dim=2)
                kc_exp = kc.repeat_interleave(grp, dim=2)
                
                h_chunk = hf[ci_base + ci].unsqueeze(0)
                
                qh = self._einsum("bvhd, bhde -> bvhe", qc_exp, h_chunk)
                inter = qh * torch.exp(gc).unsqueeze(-1)
                
                qk = self._einsum("bihd, bjhd -> bihj", qc_exp, kc_exp)
                
                causal = torch.arange(vlen, device=self.device)[:, None] >= torch.arange(vlen, device=self.device)[None, :]
                
                diff_g = gc.unsqueeze(2) - gc.unsqueeze(1)
                diff_g = diff_g.permute(0, 1, 3, 2)
                
                gate = torch.exp(torch.minimum(diff_g, torch.zeros_like(diff_g)))
                causal_exp = causal.unsqueeze(0).unsqueeze(2)
                
                qk_gated = qk * gate * causal_exp.to(self.dtype)
                
                intra = self._einsum("bihj, bjhd -> bihd", qk_gated, vc)
                o[:, s:e, :, :] = inter + intra
                
            ci_base += nc
        return o

    def run_full_pipeline(
        self, q, k, v, g_in, beta, cu_seqlens_list, H, Hg, scale=1.0, C=128
    ):
        cu = cu_seqlens_list
        g_sum = self.cumsum(g_in, C, cu)
        A = self.kkt(k, beta, g_sum, C, cu)
        A_inv = self.solve_tril(A, C, cu)
        w, u = self.wy_fast(k, v, beta, A_inv, g_sum, C, cu)
        h_states, v_new, _ = self.chunk_h(k, w, u, g_sum, C, cu)
        o = self.chunk_o(q, k, v_new, h_states, g_sum, C, cu)
        return o * scale
