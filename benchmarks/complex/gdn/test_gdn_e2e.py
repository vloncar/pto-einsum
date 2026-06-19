import pytest
import torch
import os
from typing import List, Optional

from utils import NumericalAccuracy, generate_random_inputs
from ref_gdn import RefGDN
from einsum_gdn import EinsumGDN

ACCURACY = NumericalAccuracy()
C = 128
D = 128
HG = 16

TEST_SHAPES = [
    (256, None),
    (512, None),
    ([0, 256, 512], 512),
    ([0, 128, 384, 768], 768),
    ([0, 128, 256, 512], 512),
]

HEADS = [16, 32]

@pytest.fixture
def gdn_instances():
    ref = RefGDN(torch.double)
    einsum = EinsumGDN(torch.double)
    return ref, einsum

@pytest.mark.parametrize("T_or_cu, T_total", TEST_SHAPES, ids=lambda x: str(x))
@pytest.mark.parametrize("H", HEADS)
def test_e2e_pipeline(gdn_instances, T_or_cu, T_total, H):
    ref_gdn, einsum_gdn = gdn_instances
    
    cu_list = T_or_cu if isinstance(T_or_cu, list) else None
    T = T_total if cu_list else T_or_cu
    scale = D ** -0.5
    
    q, k, v, beta, g_in = generate_random_inputs(T, H, HG, D, dtype=torch.float32)
    
    ref_out = ref_gdn.run_full_pipeline(q, k, v, g_in, beta, cu_list, H, HG, scale, C=C)
    einsum_out = einsum_gdn.run_full_pipeline(q, k, v, g_in, beta, cu_list, H, HG, scale, C=C)
    
    assert ACCURACY.stats_ok(einsum_out, ref_out)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
