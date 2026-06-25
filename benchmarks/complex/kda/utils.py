import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class NumericalAccuracy:
    # Elem-wise relative tolerance.
    rtol: float = 5e-3
    # Elem-wise absolute tolerance.
    atol: float = 1.5 * 1e-4
    # Frobenius norm-wise relative tolerance.
    ftol: float = 1e-3

    def stats_ok(
        self, actual: torch.Tensor, expected: torch.Tensor, chunk_size: int = 1
    ) -> bool:
        adjusted_rtol = min(0.5, self.rtol * chunk_size)

        actual_fp64 = actual.double()
        expected_fp64 = expected.double()

        diff = (actual_fp64 - expected_fp64).abs()
        expected_norm = torch.sum(expected_fp64**2)
        if expected_norm == 0:
            frob_rel_error = torch.tensor(0.0)
        else:
            frob_rel_error = torch.sqrt(torch.sum(diff**2) / expected_norm)

        rel_err_bound = self.atol + adjusted_rtol * expected_fp64.abs()

        all_ok = True
        if (diff > rel_err_bound).any():
            print(
                f"ERROR: max error larger than the bound: {(diff).max().item()}. "
                f"ATOL={self.atol} RTOL={adjusted_rtol}"
            )
            all_ok = False
        if frob_rel_error > self.ftol:
            print(
                f"ERROR: large frobenius relative error: {frob_rel_error}. FTOL={self.ftol}"
            )
            all_ok = False
        return all_ok


def generate_random_inputs(T, H, HV, K, dtype=torch.float32, device="cpu"):
    """KDA inputs.  Unlike GDN (scalar per-head gate), KDA's gate ``g`` is a
    per-dimension log-decay of shape ``[1, T, HV, K]`` and ``beta`` is a
    post-sigmoid scalar in ``(0, 1)``.

      q, k: ``[1, T, H,  K]``  L2-normalised  (H  = query/key heads)
      v:    ``[1, T, HV, K]``                 (HV = value/gate heads, HV % H == 0)
      beta: ``[1, T, HV]``     sigmoid(randn) in (0, 1)
      g:    ``[1, T, HV, K]``  small negative per-dim log-decay

    g is kept in ``(-0.05, 0)`` per element so the within-chunk cumulative sum
    stays bounded (|g_cs| < ~7 for C=128): the megagdn fp16 kernels stage
    ``exp(g_cs)`` and ``exp(-g_cs)`` separately for the Cube GEMMs, and larger
    magnitudes overflow fp16 (~65504) -> inf -> NaN.  Mirrors test_kda_e2e.py.
    """
    q = F.normalize(torch.randn(1, T, H, K, dtype=dtype, device=device), dim=-1, p=2)
    k = F.normalize(torch.randn(1, T, H, K, dtype=dtype, device=device), dim=-1, p=2)
    v = torch.randn(1, T, HV, K, dtype=dtype, device=device)
    beta = torch.sigmoid(torch.randn(1, T, HV, dtype=dtype, device=device))
    g = -torch.rand(1, T, HV, K, dtype=torch.float32, device=device) * 0.05
    return q, k, v, beta, g.to(dtype)


def seq_ranges(T: int, cu_seqlens=None) -> list[tuple[int, int]]:
    if cu_seqlens is None:
        return [(0, T)]
    cu = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
    return [(cu[i], cu[i + 1]) for i in range(len(cu) - 1)]


def total_chunks(T: int, cs: int, cu_seqlens=None) -> int:
    ranges = seq_ranges(T, cu_seqlens)
    return sum((eos - bos + cs - 1) // cs for bos, eos in ranges)
