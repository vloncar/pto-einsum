import numpy as np
import torch
import torch_npu
import ctypes
import os
import shutil

TEST_DEVICE = os.getenv('EINSUM_TEST_DEVICE', 'npu')

def get_transpose_config(shape, perm):
    new_shape = tuple(shape[i] for i in perm)
    strides = np.cumprod((shape[1:] + (1,))[::-1])[::-1]
    perm_strides = tuple(int(strides[i]) for i in perm)
    return new_shape, perm_strides

def generate_transpose_cpp(config_name, shape, perm, perm_strides, new_shape, dtype_str):
    dims = len(shape)
    N = int(np.prod(shape))
    
    config_code = f"""struct {config_name} {{
    static constexpr unsigned dims = {dims};
    static constexpr unsigned N = {N};
    static constexpr unsigned from_shape[{dims}] = {{{", ".join(map(str, shape))}}};
    static constexpr unsigned to_shape[{dims}] = {{{", ".join(map(str, new_shape))}}};
    static constexpr unsigned perm[{dims}] = {{{", ".join(map(str, perm))}}};
    static constexpr unsigned perm_strides[{dims}] = {{{", ".join(map(str, perm_strides))}}};
}};
"""
    
    cpp_code = f"""#include "pto_einsum.h"

{config_code}

extern "C" {{
    void run_transpose(const {dtype_str}* input, {dtype_str}* output, void* stream) {{
        pto_einsum::transpose<{dtype_str}, {config_name}>(input, output, stream);
    }}
}}
"""
    return cpp_code

def compile_and_load(cpp_code, build_dir, npu_arch, script_dir):
    import hashlib
    import subprocess
    
    code_hash = hashlib.md5(cpp_code.encode('utf-8')).hexdigest()
    target_dir = os.path.join(build_dir, f"tpose_{code_hash}")
    os.makedirs(target_dir, exist_ok=True)
    
    cpp_filename = os.path.join(target_dir, "tpose.cpp")
    so_filename = os.path.join(target_dir, "tpose.so")
    
    with open(cpp_filename, 'w') as f:
        f.write(cpp_code)
        
    from pto_einsum import INCLUDE_DIR

    ascend_path = os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    pto_lib_path = os.environ.get("PTO_LIB_PATH")
    if not pto_lib_path:
        raise RuntimeError(
            "PTO_LIB_PATH is not set. Point it at the pto-isa install root "
            "(containing 'include/pto/...'). Example: export PTO_LIB_PATH=/path/to/pto-isa"
        )

    compile_cmd = [
        "bisheng", "-O3", "-shared", "-fPIC", "-std=c++17", "-xcce", f"--npu-arch={npu_arch}",
        "-I", INCLUDE_DIR,
        "-I", f"{ascend_path}/include",
        "-I", f"{pto_lib_path}/include",
        "-L", f"{ascend_path}/lib64",
        "-lascendcl",
        cpp_filename, "-o", so_filename
    ]
    
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        shutil.rmtree(target_dir)
        raise RuntimeError(f"Compilation failed:\n{result.stderr}")
        
    lib = ctypes.CDLL(so_filename)
    return lib, target_dir

def test_transpose_case(shape, perm, dtype):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, "build_tpose")
    os.makedirs(build_dir, exist_ok=True)
    
    npu_arch = os.environ.get("NPU_ARCH", "dav-2201").strip()
    
    if dtype == torch.float32:
        dtype_str = 'float'
        c_type = ctypes.c_float
    elif dtype == torch.float16:
        dtype_str = 'half'
        c_type = ctypes.c_uint16
    else:
        raise TypeError("Unsupported dtype")

    new_shape, perm_strides = get_transpose_config(shape, perm)
    
    cpp_code = generate_transpose_cpp("tpose_config", shape, perm, perm_strides, new_shape, dtype_str)
    
    lib, target_dir = compile_and_load(cpp_code, build_dir, npu_arch, script_dir)
    
    # Set up ctypes signature
    lib.run_transpose.argtypes = [
        ctypes.POINTER(c_type),
        ctypes.POINTER(c_type),
        ctypes.c_void_p
    ]
    lib.run_transpose.restype = None
    
    # Generate random inputs
    inp = torch.rand(shape, dtype=dtype)
    expected = inp.permute(perm).contiguous()
    
    npu_inp = inp.npu()
    npu_out = torch.zeros(new_shape, dtype=dtype, device="npu")
    
    torch.npu.synchronize()
    stream_ptr = torch.npu.current_stream()._as_parameter_
    
    inp_ptr = ctypes.cast(npu_inp.data_ptr(), ctypes.POINTER(c_type))
    out_ptr = ctypes.cast(npu_out.data_ptr(), ctypes.POINTER(c_type))
    
    lib.run_transpose(inp_ptr, out_ptr, stream_ptr)
    
    torch.npu.synchronize()
    
    res_cpu = npu_out.cpu()
    
    # Cleanup compiled library files
    del lib
    shutil.rmtree(target_dir)
    
    # Compare
    diff = torch.max(torch.abs(expected - res_cpu)).item()
    print(f"Shape: {shape} | Perm: {perm} | Dtype: {dtype} -> Diff: {diff}")
    assert diff < 1e-4, f"Mismatch found! Shape: {shape}, Perm: {perm}, Diff: {diff}"

def run_tests():
    if TEST_DEVICE == 'cpu':
        print("Skipping transpose NPU test on CPU.")
        return
        
    # Setup seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_cases = [
        # 1. Identity Permutations (tests the optimized copy path)
        ((128,), (0,), torch.float32),
        ((256,), (0,), torch.float16),
        ((32, 64), (0, 1), torch.float32),
        ((16, 32), (0, 1), torch.float16),
        ((8, 16, 32), (0, 1, 2), torch.float32),
        ((4, 8, 16, 32), (0, 1, 2, 3), torch.float32),
        
        # Non-multiples of 16 for identity copies
        ((123,), (0,), torch.float32),
        ((17, 35), (0, 1), torch.float32),
        ((3, 7, 13), (0, 1, 2), torch.float32),
        
        # 2. 2D Non-Identity Permutations (tests 2D transpose mapping)
        ((32, 64), (1, 0), torch.float32),
        ((64, 32), (1, 0), torch.float32),
        ((16, 16), (1, 0), torch.float32),
        ((32, 64), (1, 0), torch.float16),
        
        # Non-multiples of 16 for 2D transpose (must have total size matching 32-byte alignment)
        ((10, 20), (1, 0), torch.float32),
        ((8, 12), (1, 0), torch.float32),
        ((10, 24), (1, 0), torch.float16),

        # 3. Large 2D transposes that exceed the UB budget (tests the blocked
        #    TTRANS path; requires 16-aligned dims). 256x256 fp32 ~ 768 KB > UB.
        ((256, 256), (1, 0), torch.float32),
        ((256, 128), (1, 0), torch.float32),
        ((128, 256), (1, 0), torch.float16),
        ((512, 64), (1, 0), torch.float32),
    ]
    
    print("=========================================")
    print("       RUNNING NPU TRANSPOSE TESTS       ")
    print("=========================================")
    
    success_count = 0
    for shape, perm, dtype in test_cases:
        try:
            test_transpose_case(shape, perm, dtype)
            success_count += 1
        except Exception as e:
            print(f"FAILED: Shape: {shape} | Perm: {perm} | Dtype: {dtype}. Error: {e}")
            
    print("=========================================")
    print(f"Passed: {success_count}/{len(test_cases)} tests.")
    print("=========================================")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, "build_tpose")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        
    if success_count != len(test_cases):
        exit(1)

if __name__ == "__main__":
    run_tests()
