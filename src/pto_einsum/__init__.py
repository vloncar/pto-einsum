"""pto_einsum: a JIT-compiled einsum for Ascend NPUs (a2a3) via the PTO-ISA.

Public API:
    einsum(equation, a, b, device="npu")  -> result tensor
    EinsumBuilder(equation, shapes, dtype) -> reusable compiled runner

Example:
    >>> from pto_einsum import einsum
    >>> c = einsum("ij, jk -> ik", a, b)            # a, b are NPU tensors
"""

from .builder import einsum, EinsumBuilder, INCLUDE_DIR

__all__ = ["einsum", "EinsumBuilder", "INCLUDE_DIR"]
__version__ = "0.1.0"
