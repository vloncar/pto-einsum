import os
import torch
from utils import seq_ranges, total_chunks

def _safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0, torch.exp(x), torch.zeros_like(x))

class EinsumGDN:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
        self.device = os.getenv("EINSUM_TEST_DEVICE", "npu")

    def _einsum(self, eq: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Two-operand contraction backend. Subclasses (e.g. the pto-einsum GDN)
        override this to swap torch.einsum for a different einsum implementation
        while reusing the exact same pipeline glue."""
        return torch.einsum(eq, a, b)

    def cumsum(self, g: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        B, T, H = g.shape
        gf = g.to(self.dtype).to(self.device)
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
        out = torch.zeros(B, T, H, cs, dtype=self.dtype, device=self.device)
        kf = k.to(self.dtype).to(self.device)
        bf = beta.to(self.dtype).to(self.device)
        gf = g_cumsum.to(self.dtype).to(self.device)
        
        for bos, eos in seq_ranges(T, cu_seqlens):
            for j in range(0, eos - bos, cs):
                s, e = bos + j, min(bos + j + cs, eos)
                v = e - s
                kc = kf[:, s:e, :, :]
                gc = gf[:, s:e, :]
                bc = bf[:, s:e, :]
                
                kc_exp = kc.repeat_interleave(grp, dim=2)
                
                qk = self._einsum("bihd, bjhd -> bijh", kc_exp, kc_exp)
                diff_g = gc.unsqueeze(2) - gc.unsqueeze(1)
                exp_g = _safe_exp(diff_g)
                
                blk = qk * exp_g * bc.unsqueeze(2)
                mask = torch.arange(v, device=self.device)[:, None] > torch.arange(v, device=self.device)[None, :]
                
                masked_blk = blk * mask.unsqueeze(0).unsqueeze(-1)
                out[:, s:e, :, :v] = masked_blk.permute(0, 1, 3, 2)
        return out

    def solve_tril(self, A: torch.Tensor, cs: int, cu_seqlens=None) -> torch.Tensor:
        B, T, H, _ = A.shape
        out = torch.zeros(B, T, H, cs, dtype=self.dtype, device=self.device)
        Af = A.to(self.dtype).to(self.device)
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
        w = torch.zeros(B, T, H, Kd, dtype=self.dtype, device=self.device)
        u = torch.zeros(B, T, H, v.shape[-1], dtype=self.dtype, device=self.device)
        
        kf = k.to(self.dtype).to(self.device)
        vf = v.to(self.dtype).to(self.device)
        bf = beta.to(self.dtype).to(self.device)
        Af = A_inv.to(self.dtype).to(self.device)
        gf = g_cumsum.to(self.dtype).to(self.device)
        
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
        
        kf = k.to(self.dtype).to(self.device)
        wf = w.to(self.dtype).to(self.device)
        uf = u.to(self.dtype).to(self.device)
        gf = g_cumsum.to(self.dtype).to(self.device)
        
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
        
        qf = q.to(self.dtype).to(self.device)
        kf = k.to(self.dtype).to(self.device)
        vf = v_new.to(self.dtype).to(self.device)
        hf = h_states.to(self.dtype).to(self.device)
        gf = g_cumsum.to(self.dtype).to(self.device)
        
        o = torch.zeros(B, T, H, Dd, dtype=self.dtype, device=self.device)
        ranges = seq_ranges(T, cu_seqlens)
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
        _, v_new, _ = self.chunk_h(k, w, u, g_sum, C, cu)
        h_states, v_new, _ = self.chunk_h(k, w, u, g_sum, C, cu)
        o = self.chunk_o(q, k, v_new, h_states, g_sum, C, cu)
        return o * scale
