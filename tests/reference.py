"""Pure-Python einsum reference (transpose + batched matmul).

These are the golden building blocks the standalone-kernel component test
(test_matmul.py) validates the NPU transpose / Cube matmul against. This module is
a helper, not a pytest suite -- run it directly (`python tests/reference.py`) for a
quick CPU self-check that the reference itself matches torch.einsum.
"""
import numpy as np
import torch


def transfer_idx(index, to_shape, perm_strides):
    idx = 0
    for i in range(len(to_shape) - 1, -1, -1):
        idx += (index % to_shape[i]) * perm_strides[i]
        index //= to_shape[i]
    return idx

def transpose(data_flat, to_shape, perm_strides):
    N = int(np.prod(to_shape))
    res = np.zeros(N, dtype=data_flat.dtype)
    for i in range(N):
        idx = transfer_idx(i, to_shape, perm_strides)
        res[i] = data_flat[idx]
    return res

def batched_matmul(tpose_i0, tpose_i1, I, L0, L1, C):
    res = np.zeros(I * L0 * L1, dtype=np.float16)
    tpose_i0 = tpose_i0.flatten()
    tpose_i1 = tpose_i1.flatten()
    for i in range(I):
        for l0 in range(L0):
            for l1 in range(L1):
                acc = 0.0
                for c in range(C):
                    a = tpose_i0[(i * L0 + l0) * C + c]
                    b = tpose_i1[(i * L1 + l1) * C + c]
                    acc += a * b
                res[(i * L0 + l0) * L1 + l1] = acc
    return res

def _self_check():
    # Test case: ij, jk -> ik
    # shapes: (32, 64), (64, 128)
    # output: (32, 128)
    L0, L1, C, I = 32, 128, 64, 1
    
    inp0 = np.random.rand(32, 64).astype(np.float32)
    inp1 = np.random.rand(64, 128).astype(np.float32)

    # 1. Transpose Input 0
    # From config: from_shape=(32, 64), to_shape=(32, 64), perm=(0, 1), perm_strides=(64, 1)
    to_shape0 = (32, 64)
    perm_strides0 = (64, 1)
    tpose0 = transpose(inp0.flatten(), to_shape0, perm_strides0)

    # 2. Transpose Input 1
    # From config: from_shape=(64, 128), to_shape=(128, 64), perm=(1, 0), perm_strides=(1, 128)
    to_shape1 = (128, 64)
    perm_strides1 = (1, 128)
    tpose1 = transpose(inp1.flatten(), to_shape1, perm_strides1)

    # 3. Batched Matmul
    res_flat = batched_matmul(tpose0, tpose1, I, L0, L1, C)

    # 4. Transpose Output
    # From config: from_shape=(32, 128), to_shape=(32, 128), perm=(0, 1), perm_strides=(128, 1)
    to_shape_out = (32, 128)
    perm_strides_out = (128, 1)
    final_res = transpose(res_flat, to_shape_out, perm_strides_out)
    final_res = final_res.reshape(32, 128)

    # Compare with torch
    torch_res = torch.einsum('ij, jk -> ik', torch.tensor(inp0), torch.tensor(inp1)).numpy()

    print("Max difference:", np.max(np.abs(final_res - torch_res)))
    assert np.allclose(final_res, torch_res, rtol=1e-3, atol=1e-5)
    print("Test passed successfully!")

if __name__ == '__main__':
    _self_check()
