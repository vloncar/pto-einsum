import numpy as np
import torch
import torch_npu
import ctypes
import sys
import os

TEST_DEVICE = os.getenv('EINSUM_TEST_DEVICE', 'npu')

# test_python_einsum is a sibling helper module in this tests/ directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pto_einsum import EinsumBuilder
from test_python_einsum import transpose as py_transpose, batched_matmul as py_batched_matmul

def test_npu_split_einsum():
    if TEST_DEVICE == 'cpu':
        print("Skipping split NPU test on CPU.")
        return

    torch.manual_seed(42)
    np.random.seed(42)

    equation = 'ij, jk -> ik'
    shape0 = (32, 64)
    shape1 = (64, 128)
    dtype = torch.float32

    L0, L1, C, I = 32, 128, 64, 1

    inp0 = torch.rand(shape0, dtype=dtype)
    inp1 = torch.rand(shape1, dtype=dtype)

    builder = EinsumBuilder(equation, [shape0, shape1], dtype)
    builder.parse_equation()
    builder.generate_code()
    print("Compiling split NPU kernels...")
    builder.compile()
    builder.load_library()
    print("Compilation and loading successful.")

    # 1. Transpose Input 0
    to_shape0 = (32, 64)
    perm_strides0 = (64, 1)
    py_tpose0 = py_transpose(inp0.numpy().flatten(), to_shape0, perm_strides0)

    npu_inp0 = inp0.npu()
    npu_tpose0 = torch.zeros(to_shape0, dtype=dtype, device="npu")

    # 2. Transpose Input 1 (with new order: contract + invariant1 = identity perm)
    to_shape1 = (64, 128)
    perm_strides1 = (128, 1)
    py_tpose1 = py_transpose(inp1.numpy().flatten(), to_shape1, perm_strides1)
    
    npu_inp1 = inp1.npu()
    npu_tpose1 = torch.zeros(to_shape1, dtype=dtype, device="npu")

    # Sync before taking stream pointer
    torch.npu.synchronize()
    stream_ptr = torch.npu.current_stream()._as_parameter_

    builder.lib.run_transpose_inp0(
        ctypes.cast(npu_inp0.data_ptr(), ctypes.POINTER(builder.c_type)),
        ctypes.cast(npu_tpose0.data_ptr(), ctypes.POINTER(builder.c_type)),
        stream_ptr
    )
    
    builder.lib.run_transpose_inp1(
        ctypes.cast(npu_inp1.data_ptr(), ctypes.POINTER(builder.c_type)),
        ctypes.cast(npu_tpose1.data_ptr(), ctypes.POINTER(builder.c_type)),
        stream_ptr
    )
    
    # Sync after custom kernels
    torch.npu.synchronize()

    diff0 = np.max(np.abs(py_tpose0.flatten() - npu_tpose0.cpu().numpy().flatten()))
    print(f"Transpose Inp0 diff: {diff0}")
    assert diff0 < 1e-4

    diff1 = np.max(np.abs(py_tpose1.flatten() - npu_tpose1.cpu().numpy().flatten()))
    print(f"Transpose Inp1 diff: {diff1}")
    assert diff1 < 1e-4

    # 3. Batched Matmul
    # py_batched_matmul expects B as (L1, C), so transpose py_tpose1 for the Python reference
    py_tpose1_for_cpu = py_tpose1.reshape(64, 128).T.flatten()  # (C,L1) -> (L1,C)
    py_res_flat = py_batched_matmul(py_tpose0, py_tpose1_for_cpu, I, L0, L1, C)

    # npu_tpose1 is already (C, L1) which is what the kernel expects
    npu_ws_res = torch.zeros(I * L0 * L1, dtype=torch.float32, device="npu")

    torch.npu.synchronize()
    stream_ptr = torch.npu.current_stream()._as_parameter_
    
    builder.lib.run_batched_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    builder.lib.run_batched_matmul(
        ctypes.cast(npu_tpose0.data_ptr(), ctypes.POINTER(builder.c_type)),
        ctypes.cast(npu_tpose1.data_ptr(), ctypes.POINTER(builder.c_type)),
        ctypes.cast(npu_ws_res.data_ptr(), ctypes.c_void_p),
        stream_ptr
    )
    
    torch.npu.synchronize()

    py_res_flat_np = py_res_flat
    npu_ws_res_np = npu_ws_res.cpu().numpy().flatten()
    print("Python matmul sample:", py_res_flat_np[:10])
    print("NPU matmul sample:   ", npu_ws_res_np[:10])
    diff2 = np.max(np.abs(py_res_flat_np - npu_ws_res_np))
    print(f"Batched Matmul diff: {diff2}")
    assert diff2 < 1e-2

    # 4. Final Output Transpose
    to_shape_out = (32, 128)
    perm_strides_out = (128, 1)
    py_final_res = py_transpose(py_res_flat, to_shape_out, perm_strides_out).reshape(to_shape_out)

    npu_out = torch.zeros(to_shape_out, dtype=torch.float32, device="npu")

    torch.npu.synchronize()
    stream_ptr = torch.npu.current_stream()._as_parameter_

    builder.lib.run_transpose_out(
        ctypes.cast(npu_ws_res.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(npu_out.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        stream_ptr
    )

    torch.npu.synchronize()

    py_final_res_np = py_final_res.flatten()
    npu_out_np = npu_out.cpu().numpy().flatten()
    print("Python final sample:", py_final_res_np[:10])
    print("NPU final sample:   ", npu_out_np[:10])
    diff3 = np.max(np.abs(py_final_res_np - npu_out_np))
    print(f"Final Transpose diff: {diff3}")
    assert diff3 < 1e-2
    
    print("All NPU split components verify perfectly against Python baseline!")

if __name__ == "__main__":
    test_npu_split_einsum()
