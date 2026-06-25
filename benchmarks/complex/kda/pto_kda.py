"""KDA pipeline backed by the JIT-compiled pto-einsum kernels.

Reuses the *entire* ``EinsumKDA`` algorithm — chunk loops, cumsum, gating,
masking, the triangular solve — and swaps only the two-operand contractions
(``self._einsum``) from ``torch.einsum`` to a real ``pto_einsum`` kernel.  The
4-way benchmark thus isolates exactly what pto-einsum contributes on the
contraction stages (kkt, wy, chunk_h, chunk_o) with identical glue.

Each distinct (equation, input-shapes, dtype) builds one persistent runner the
first time it is seen and reuses it on every later chunk, so steady-state timing
reflects launch + compute, not JIT compilation.
"""
import torch

from einsum_kda import EinsumKDA
from pto_einsum import EinsumBuilder


class PtoKDA(EinsumKDA):
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
