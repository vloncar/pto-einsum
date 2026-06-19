"""GDN pipeline backed by the JIT-compiled pto-einsum kernels.

This reuses the *entire* ``EinsumGDN`` algorithm — chunk loops, cumsum, gating,
masking, the triangular solve — and swaps only the two-operand contractions
(``self._einsum``) from ``torch.einsum`` to a real ``pto_einsum`` kernel. So the
4-way benchmark isolates exactly what the pto-einsum library contributes on the
contraction-heavy stages (kkt, wy_fast, chunk_h, chunk_o), with identical glue.

Each distinct (equation, input-shapes, dtype) builds one persistent runner the
first time it is seen and reuses it on every later chunk (the chunk loop hits the
same shapes when sequence lengths are multiples of the chunk size), so steady-state
timing reflects launch + compute, not JIT compilation.
"""
import torch
from einsum_gdn import EinsumGDN
from pto_einsum import EinsumBuilder


class PtoGDN(EinsumGDN):
    def __init__(self, dtype: torch.dtype):
        super().__init__(dtype)
        self._runners: dict = {}
        self._builders: list = []

    def _einsum(self, eq: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # pto-einsum kernels consume contiguous NPU operands; chunk slices are views.
        a = a.contiguous()
        b = b.contiguous()
        key = (eq, tuple(a.shape), tuple(b.shape), a.dtype)
        runner = self._runners.get(key)
        if runner is None:
            builder = EinsumBuilder(eq, [tuple(a.shape), tuple(b.shape)], a.dtype, device="npu")
            runner = builder.build()
            self._runners[key] = runner
            self._builders.append(builder)
        return runner(a, b)

    def cleanup(self):
        for b in self._builders:
            try:
                b.cleanup()
            except Exception:
                pass
        self._builders.clear()
        self._runners.clear()
