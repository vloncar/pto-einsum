import pytest
import torch
import os
from dataclasses import dataclass
from typing import List, Optional

from utils import NumericalAccuracy, generate_random_inputs
from ref_gdn import RefGDN
from einsum_gdn import EinsumGDN

ACCURACY = NumericalAccuracy()
C = 128
D = 128

@dataclass
class GDNTestCase:
    label: str
    cu_seqlens_list: Optional[List[int]]
    T: int
    dtype: torch.dtype = torch.double

def _cu_from_seqlens(seqlens: List[int]) -> List[int]:
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    return cu

def get_test_cases():
    cases = []
    for T in [128, 256, 385]:
        cases.append(GDNTestCase(f"fixed T={T}", None, T))
    for seqlens in [
        [256, 256],
        [128, 128, 128],
        [1, 63, 64, 128]
    ]:
        cu = _cu_from_seqlens(seqlens)
        cases.append(GDNTestCase(f"varlen {seqlens}", cu, cu[-1]))
    return cases

TEST_CASES = get_test_cases()
HEADS = [16, 32]
HG = 16

@pytest.fixture
def gdn_instances():
    ref = RefGDN(torch.double)
    einsum = EinsumGDN(torch.double)
    return ref, einsum

@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda tc: tc.label)
@pytest.mark.parametrize("H", HEADS)
def test_cumsum(gdn_instances, tc, H):
    ref_gdn, einsum_gdn = gdn_instances
    T = tc.T
    _, _, _, _, g_in = generate_random_inputs(T, H, HG, D, dtype=torch.float32)
    
    ref_out = ref_gdn.cumsum(g_in, C, tc.cu_seqlens_list)
    einsum_out = einsum_gdn.cumsum(g_in, C, tc.cu_seqlens_list)
    
    assert ACCURACY.stats_ok(einsum_out, ref_out)

@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda tc: tc.label)
@pytest.mark.parametrize("H", HEADS)
def test_kkt(gdn_instances, tc, H):
    ref_gdn, einsum_gdn = gdn_instances
    T = tc.T
    _, k, _, beta, g_in = generate_random_inputs(T, H, HG, D)
    
    g_sum = ref_gdn.cumsum(g_in, C, tc.cu_seqlens_list)
    
    ref_out = ref_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)
    einsum_out = einsum_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)
    
    assert ACCURACY.stats_ok(einsum_out, ref_out, chunk_size=C)

@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda tc: tc.label)
@pytest.mark.parametrize("H", HEADS)
def test_solve_tril(gdn_instances, tc, H):
    ref_gdn, einsum_gdn = gdn_instances
    T = tc.T
    _, k, _, beta, g_in = generate_random_inputs(T, H, HG, D)
    g_sum = ref_gdn.cumsum(g_in, C, tc.cu_seqlens_list)
    A = ref_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)
    
    ref_out = ref_gdn.solve_tril(A, C, tc.cu_seqlens_list)
    einsum_out = einsum_gdn.solve_tril(A, C, tc.cu_seqlens_list)
    
    assert ACCURACY.stats_ok(einsum_out, ref_out, chunk_size=C)

@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda tc: tc.label)
@pytest.mark.parametrize("H", HEADS)
def test_wy_fast(gdn_instances, tc, H):
    ref_gdn, einsum_gdn = gdn_instances
    T = tc.T
    _, k, v, beta, g_in = generate_random_inputs(T, H, HG, D)
    g_sum = ref_gdn.cumsum(g_in, C, tc.cu_seqlens_list)
    A = ref_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)
    A_inv = ref_gdn.solve_tril(A, C, tc.cu_seqlens_list)
    
    ref_w, ref_u = ref_gdn.wy_fast(k, v, beta, A_inv, g_sum, C, tc.cu_seqlens_list)
    einsum_w, einsum_u = einsum_gdn.wy_fast(k, v, beta, A_inv, g_sum, C, tc.cu_seqlens_list)
    
    assert ACCURACY.stats_ok(einsum_w, ref_w, chunk_size=C)
    assert ACCURACY.stats_ok(einsum_u, ref_u, chunk_size=C)

@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda tc: tc.label)
@pytest.mark.parametrize("H", HEADS)
def test_chunk_h(gdn_instances, tc, H):
    ref_gdn, einsum_gdn = gdn_instances
    T = tc.T
    _, k, v, beta, g_in = generate_random_inputs(T, H, HG, D)
    g_sum = ref_gdn.cumsum(g_in, C, tc.cu_seqlens_list)
    A = ref_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)
    A_inv = ref_gdn.solve_tril(A, C, tc.cu_seqlens_list)
    w, u = ref_gdn.wy_fast(k, v, beta, A_inv, g_sum, C, tc.cu_seqlens_list)
    
    ref_h, ref_v, _ = ref_gdn.chunk_h(k, w, u, g_sum, C, tc.cu_seqlens_list)
    einsum_h, einsum_v, _ = einsum_gdn.chunk_h(k, w, u, g_sum, C, tc.cu_seqlens_list)
    
    assert ACCURACY.stats_ok(einsum_h, ref_h, chunk_size=C)
    assert ACCURACY.stats_ok(einsum_v, ref_v, chunk_size=C)

@pytest.mark.parametrize("tc", TEST_CASES, ids=lambda tc: tc.label)
@pytest.mark.parametrize("H", HEADS)
def test_chunk_o(gdn_instances, tc, H):
    ref_gdn, einsum_gdn = gdn_instances
    T = tc.T
    q, k, v, beta, g_in = generate_random_inputs(T, H, HG, D)
    g_sum = ref_gdn.cumsum(g_in, C, tc.cu_seqlens_list)
    A = ref_gdn.kkt(k, beta, g_sum, C, tc.cu_seqlens_list)
    A_inv = ref_gdn.solve_tril(A, C, tc.cu_seqlens_list)
    w, u = ref_gdn.wy_fast(k, v, beta, A_inv, g_sum, C, tc.cu_seqlens_list)
    h_states, v_new, _ = ref_gdn.chunk_h(k, w, u, g_sum, C, tc.cu_seqlens_list)
    
    ref_o = ref_gdn.chunk_o(q, k, v_new, h_states, g_sum, C, tc.cu_seqlens_list)
    einsum_o = einsum_gdn.chunk_o(q, k, v_new, h_states, g_sum, C, tc.cu_seqlens_list)
    
    assert ACCURACY.stats_ok(einsum_o, ref_o, chunk_size=C)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
