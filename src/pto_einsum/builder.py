import os
import shutil
import hashlib
from math import prod
import numpy as np
from typing import TypedDict
import ctypes
import subprocess
import torch
import torch_npu
from . import templates

# Directory holding the bundled C++ kernel headers (pto_einsum.h, cpu_einsum.h).
# Used as the compiler include path and exposed for out-of-package builds.
INCLUDE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "include")

class EinsumRecipe(TypedDict):
    direct_sum_axis: tuple[tuple[int, ...], tuple[int, ...]]
    in_transpose_idxs: tuple[tuple[int, ...], tuple[int, ...]]
    L0: int
    L1: int
    I: int
    C: int
    out_interpret_shape: tuple[int, ...]
    out_transpose_idxs: tuple[int, ...]

class EinsumBuilder:
    def __init__(self, equation: str, input_shapes: list[tuple[int, ...]], dtype: torch.dtype, device: str = "npu"):
        self.equation = equation
        self.input_shapes = input_shapes
        self.dtype = dtype
        self.device = device
        
        # Set up directory paths. Generated sources and compiled .so files go under
        # the build base (override with EINSUM_BUILD_DIR; defaults to ./build in the
        # current working directory so it works for installed packages too).
        self.include_dir = INCLUDE_DIR
        self.build_base = os.getenv("EINSUM_BUILD_DIR")
        if not self.build_base:
            self.build_base = os.path.join(os.getcwd(), "build")
        else:
            self.build_base = os.path.abspath(self.build_base)
            
        # Placeholders for intermediate results
        self.recipe = None
        self.cpp_code = None
        self.cpp_filename = None
        self.so_filename = None
        self.lib = None
        self.c_type = None
        self.code_hash = None
        self.target_dir = None
        self._workspace = None  # persistent NPU workspace (lazily allocated in run)
        self._splitk = False    # split-K output needs zero-init (set in build())

    def _validate_einsum_expr(self, fn: str, shape0: tuple[int, ...], shape1: tuple[int, ...]):
        inp, out = map(str.strip, fn.split('->'))
        in0, in1 = map(str.strip, inp.split(','))
        alphabets = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        s_alphabets = set(alphabets)

        if not (s_alphabets >= set(in0.replace('...', '') + in1.replace('...', '') + out.replace('...', ''))):
            raise ValueError(f"einsum string {fn} is invalid: subscripts should be in [a-zA-Z] and '...' only")

        in0 = in0.replace('...', '0')
        in1 = in1.replace('...', '0')
        out = out.replace('...', '0')
        ax_in0, ax_in1, ax_out = list(in0), list(in1), list(out)
        sax_in0, sax_in1, sax_out = set(ax_in0), set(ax_in1), set(ax_out)
        free_indices = ''.join(sorted(s_alphabets - sax_in0 - sax_in1 - sax_out))

        if len(sax_in0) != len(ax_in0):
            for a in in0:
                if in0.count(a) == 1:
                    continue
                a = a if a != '0' else '...'
                raise ValueError(f"einsum string {fn} is invalid: input0 subscripts includes '{a}' multiple times")
        if len(sax_in1) != len(ax_in1):
            for a in in1:
                if in1.count(a) == 1:
                    continue
                a = a if a != '0' else '...'
                raise ValueError(f"einsum string {fn} is invalid: input1 subscripts includes '{a}' multiple times")
        if len(sax_out) != len(ax_out):
            for a in out:
                if out.count(a) == 1:
                    continue
                a = a if a != '0' else '...'
                raise ValueError(f"einsum string {fn} is invalid: output subscripts includes '{a}' multiple times")

        if '0' in sax_in0 or '0' in sax_in1 or '0' in sax_out:
            if '0' not in sax_out:
                raise ValueError(f'einsum string {fn} is invalid: output does not allow broadcasting, but inputs do')
            if '0' not in sax_in0 and '0' not in sax_in1:
                raise ValueError(f'einsum string {fn} is invalid: output allows broadcasting, but inputs do not')

        if remaining := sax_out - sax_in0 - sax_in1:
            raise ValueError(f'einsum string {fn} is invalid: output subscripts {remaining} not found in inputs')

        _common_in = sax_in0 & sax_in1

        if '0' in sax_in0 and '0' in sax_in1:
            n_boardcast0 = len(shape0) - len(sax_in0) + 1
            n_boardcast1 = len(shape1) - len(sax_in1) + 1
            assert n_boardcast0 == n_boardcast1, f"'...' expands to {n_boardcast0} and {n_boardcast1}-axis in input0 and input1."
            in0 = in0.replace('0', free_indices[:n_boardcast0])
            in1 = in1.replace('0', free_indices[:n_broadcast1])
            out = out.replace('0', free_indices[:n_boardcast0])
            ax_in0, ax_in1, ax_out = list(in0), list(in1), list(out)
            _common_in = set(ax_in0) & set(ax_in1)

        else:
            if '0' in sax_in0:
                if len(sax_in0) - 1 > len(shape0):
                    raise ValueError(f'Input0 requires at least {len(sax_in0) - 1} dimensions, but only {len(shape0)} given')
                n_broadcast = len(shape0) - len(sax_in0) + 1
                in0 = in0.replace('0', free_indices[:n_broadcast])
                out = out.replace('0', free_indices[:n_broadcast])
                ax_in0 = list(in0)
                ax_out = list(out)
            else:
                if len(sax_in0) != len(shape0):
                    raise ValueError(f'Input0 requires {len(sax_in0)} dimensions, but {len(shape0)} is given')

            if '0' in sax_in1:
                if len(sax_in1) - 1 > len(shape1):
                    raise ValueError(f'Input1 requires at least {len(sax_in1) - 1} dimensions, but only {len(shape1)} given')
                n_broadcast = len(shape1) - len(sax_in1) + 1
                in1 = in1.replace('0', free_indices[:n_broadcast])
                out = out.replace('0', free_indices[:n_broadcast])
                ax_in1 = list(in1)
                ax_out = list(out)
            else:
                if len(sax_in1) != len(shape1):
                    raise ValueError(f'Input1 requires {len(sax_in1)} dimensions, but {len(shape1)} is given')

        for a in _common_in:
            ax_0 = ax_in0.index(a)
            ax_1 = ax_in1.index(a)
            if shape0[ax_0] != shape1[ax_1]:
                raise ValueError(
                    f"Input dimension size mismatches for common subscript '{a}': {shape0[ax_0]} and {shape1[ax_1]}"
                )

        out_shape = tuple(shape0[ax_in0.index(a)] if a in ax_in0 else shape1[ax_in1.index(a)] for a in ax_out)
        return f'{in0},{in1}->{out}', out_shape

    def parse_recipe(self, fn: str, input_shape0: tuple[int, ...], input_shape1: tuple[int, ...]) -> EinsumRecipe:
        fn, _ = self._validate_einsum_expr(fn, input_shape0, input_shape1)

        _in, _out = fn.split('->')
        _in0, _in1 = _in.split(',')

        in0, in1, out = list(_in0), list(_in1), list(_out)
        s_in0, s_in1, s_out = set(in0), set(in1), set(out)
        _common = s_in0 & s_in1
        _contract = _common - s_out
        _inplace = _common & s_out
        contract = sorted(_contract, key=lambda x: in1.index(x))
        inplace = sorted(_inplace, key=lambda x: in1.index(x))
        invariant0 = sorted((s_out - _common) & s_in0, key=lambda x: in0.index(x))
        invariant1 = sorted((s_out - _common) & s_in1, key=lambda x: in1.index(x))
        direct_sum0 = s_in0 - s_out - _common
        direct_sum1 = s_in1 - s_out - _common
        direct_sum_axis = (
            tuple(sorted(in0.index(x) for x in direct_sum0)),
            tuple(sorted(in1.index(x) for x in direct_sum1)),
        )

        contract_idxs = tuple(map(in0.index, contract)), tuple(map(in1.index, contract))
        inplace_idxs = tuple(map(in0.index, inplace)), tuple(map(in1.index, inplace))
        invariant_idxs = tuple(map(in0.index, invariant0)), tuple(map(in1.index, invariant1))

        inplace_shape = tuple(input_shape0[i] for i in inplace_idxs[0])
        inplace_size = prod(inplace_shape)
        contract_size = prod(input_shape0[i] for i in contract_idxs[0])
        invariant_shape0 = tuple(input_shape0[i] for i in invariant_idxs[0])
        invariant_shape1 = tuple(input_shape1[i] for i in invariant_idxs[1])
        invariant_size0, invariant_size1 = prod(invariant_shape0), prod(invariant_shape1)

        transpose_idx0 = inplace_idxs[0] + invariant_idxs[0] + contract_idxs[0]
        #transpose_idx1 = inplace_idxs[1] + invariant_idxs[1] + contract_idxs[1]
        transpose_idx1 = inplace_idxs[1] + contract_idxs[1] + invariant_idxs[1]

        out_shape_pretranspose = inplace_shape + invariant_shape0 + invariant_shape1
        _out_transpose_idx = np.argsort(tuple(map(out.index, inplace + invariant0 + invariant1)))
        out_transpose_idx = tuple(int(i) for i in _out_transpose_idx)

        # Note: the Cube kernel pads L0/L1/C up to fractal granularity internally
        # (see batched_matmul_kernel_standalone), so no multiple-of-16 restriction on
        # the logical dims is required here — arbitrary shapes (including degenerate
        # L0/L1/C == 1) are supported.

        return EinsumRecipe(
            direct_sum_axis=direct_sum_axis,
            in_transpose_idxs=(transpose_idx0, transpose_idx1),
            out_interpret_shape=out_shape_pretranspose,
            out_transpose_idxs=out_transpose_idx,
            L0=invariant_size0,
            L1=invariant_size1,
            I=inplace_size,
            C=contract_size,
        )

    def transpose_config_gen(self, name: str, shape: tuple[int, ...], perm: tuple[int, ...]):
        new_shape = tuple(shape[i] for i in perm)
        strides = np.cumprod((shape[1:] + (1,))[::-1])[::-1]
        perm_strides = tuple(int(strides[i]) for i in perm)
        return dict(
            dims=len(shape),
            N=prod(shape),
            from_shape=', '.join(str(x) for x in shape),
            perm=', '.join(str(x) for x in perm),
            perm_strides=', '.join(str(x) for x in perm_strides),
            to_shape=', '.join(str(x) for x in new_shape),
            config_name=name,
        )

    def _tile_k(self, tile: int, L0: int, L1: int, C: int) -> int:
        # Pick the K-tile depth. tile_m/tile_n stay at the base `tile`; tile_k is
        # grown as large as the on-chip buffers allow for *thin* problems (small
        # padded M/N), which cuts the serial K-step count on the one core that owns
        # the (few) output tiles. For square/large M,N the L0A/L0B caps below pin it
        # back to `tile`, so this never changes the well-tuned matmul path.
        #
        # Caps (data_T operands live in L0A/L0B = 64 KB each; double-buffered Mat
        # tiles share the 512 KB CBUF, matching the kernel's static_assert):
        #   L0A : Mt*Kt*ds <= 64 KB,  L0B : Nt*Kt*ds <= 64 KB
        #   CBUF: 2*(Mt+Nt)*Kt*ds <= 512 KB
        # The result must divide the padded K so the kernel's K-tiling assert holds.
        ds = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        pad16 = lambda x: (x + 15) // 16 * 16
        Mpad, Npad, Kpad = pad16(L0), pad16(L1), pad16(C)
        Mt, Nt = min(tile, Mpad), min(tile, Npad)
        cap = min(65536 // (Mt * ds), 65536 // (Nt * ds),
                  (512 * 1024) // (2 * (Mt + Nt) * ds), Kpad)
        cap -= cap % 16  # keep fractal-aligned
        # Largest divisor of Kpad that fits the cap (16 always divides Kpad).
        for d in range(cap, 15, -16):
            if Kpad % d == 0:
                return d
        return 16

    def _splitk_eligible(self) -> bool:
        # Mirror einsum_fused_kernel's SPLITK_ELIGIBLE: the kernel splits a tile's
        # K-contraction across otherwise-idle cores, which atomic-add their slices
        # into the output, so the host must pre-zero it. Enabled for identity-output
        # (matmul writes res directly), splittable K (nK >= 2), and a tile grid small
        # enough to leave cores idle. Must stay in lockstep with the kernel constant.
        L0, L1, C, I = self.recipe['L0'], self.recipe['L1'], self.recipe['C'], self.recipe['I']
        perm = tuple(self.recipe['out_transpose_idxs'])
        out_identity = perm == tuple(range(len(perm)))
        pad16 = lambda x: (x + 15) // 16 * 16
        Mpad, Npad, Kpad = pad16(L0), pad16(L1), pad16(C)
        tile = int(os.getenv("EINSUM_TILE_SIZE", "128"))
        Mt, Nt = min(tile, Mpad), min(tile, Npad)
        Kt = self._tile_k(tile, L0, L1, C)
        nK = Kpad // Kt
        total_tiles = I * (Mpad // Mt) * (Npad // Nt)
        return out_identity and nK >= 2 and total_tiles < 16

    def einsum_config_gen(self, tpose_inp0_name: str, tpose_inp1_name: str, tpose_out_name: str):
        # Cube matmul tile sizes. A single base size (env EINSUM_TILE_SIZE, default
        # 128) drives tile_m/tile_n; tile_k is tuned per-problem (see _tile_k) to
        # speed up thin/large-K contractions like the dot product. The kernel clamps
        # each tile to its padded dim.
        tile = int(os.getenv("EINSUM_TILE_SIZE", "128"))
        L0, L1, C = self.recipe['L0'], self.recipe['L1'], self.recipe['C']
        return dict(
            tpose_inp0_name=tpose_inp0_name,
            tpose_inp1_name=tpose_inp1_name,
            tpose_out_name=tpose_out_name,
            n_free0=L0,
            n_free1=L1,
            n_contract=C,
            n_inplace=self.recipe['I'],
            tile_m=tile,
            tile_n=tile,
            tile_k=self._tile_k(tile, L0, L1, C),
        )

    def parse_equation(self):
        if len(self.input_shapes) != 2:
            raise ValueError("Only exactly two input shapes are supported.")
        shape0, shape1 = self.input_shapes
        self.recipe = self.parse_recipe(self.equation, shape0, shape1)

    def generate_code(self):
        # Determine C++ and ctypes type mappings
        if self.dtype == torch.float32:
            data_t = 'float'
            self.c_type = ctypes.c_float
        elif self.dtype == torch.float16:
            if self.device == "cpu":
                raise TypeError("float16 is not supported on CPU")
            data_t = 'half'
            self.c_type = ctypes.c_uint16
        else:
            raise TypeError(f"Unsupported torch dtype: {self.dtype}")

        shape0, shape1 = self.input_shapes
        tpose_inp0_name = 'config_tpose_inp0'
        tpose_inp1_name = 'config_tpose_inp1'
        tpose_out_name = 'config_tpose_out'
        
        tpose_inp0_dict = self.transpose_config_gen(tpose_inp0_name, shape0, self.recipe['in_transpose_idxs'][0])
        tpose_inp1_dict = self.transpose_config_gen(tpose_inp1_name, shape1, self.recipe['in_transpose_idxs'][1])
        tpose_out_dict = self.transpose_config_gen(tpose_out_name, self.recipe['out_interpret_shape'], self.recipe['out_transpose_idxs'])

        tpose_inp0_code = templates.transpose_config_template.format(**tpose_inp0_dict)
        tpose_inp1_code = templates.transpose_config_template.format(**tpose_inp1_dict)
        tpose_out_code = templates.transpose_config_template.format(**tpose_out_dict)

        einsum_dict = self.einsum_config_gen(
            tpose_inp0_name=tpose_inp0_name,
            tpose_inp1_name=tpose_inp1_name,
            tpose_out_name=tpose_out_name,
        )
        einsum_code = templates.einsum_config_template.format(**einsum_dict)

        if self.device == "cpu":
            self.cpp_code = """#include "cpu_einsum.h"
#include <stdio.h>

{tpose_inp0_code}
{tpose_inp1_code}
{tpose_out_code}
{einsum_code}

extern "C" {{
    void run_einsum(const {data_t}* input0, const {data_t}* input1, float* output, void* stream) {{
        cpu::einsum<{data_t}, config_einsum>(input0, input1, output);
    }}
}}
""".format(
                tpose_inp0_code=tpose_inp0_code,
                tpose_inp1_code=tpose_inp1_code,
                tpose_out_code=tpose_out_code,
                einsum_code=einsum_code,
                data_t=data_t,
                tpose_inp0_name=tpose_inp0_name,
                tpose_inp1_name=tpose_inp1_name,
                tpose_out_name=tpose_out_name
            )
        else:
            self.cpp_code = templates.shared_lib_template.format(
                tpose_inp0_code=tpose_inp0_code,
                tpose_inp1_code=tpose_inp1_code,
                tpose_out_code=tpose_out_code,
                einsum_code=einsum_code,
                data_t=data_t,
                tpose_inp0_name=tpose_inp0_name,
                tpose_inp1_name=tpose_inp1_name,
                tpose_out_name=tpose_out_name,
            )

        self.code_hash = hashlib.md5(self.cpp_code.encode('utf-8')).hexdigest()
        self.target_dir = os.path.join(self.build_base, self.code_hash)

    def compile(self):
        os.makedirs(self.target_dir, exist_ok=True)

        self.cpp_filename = os.path.join(self.target_dir, f"einsum_{self.code_hash}.cpp")
        self.so_filename = os.path.join(self.target_dir, f"einsum_{self.code_hash}.so")

        if not os.path.exists(self.so_filename):
            with open(self.cpp_filename, 'w') as f:
                f.write(self.cpp_code)
            
            if self.device == "cpu":
                compile_cmd = [
                    "g++", "-O3", "-shared", "-fPIC", "-std=c++17",
                    "-I", self.include_dir,
                    self.cpp_filename, "-o", self.so_filename
                ]
            else:
                ascend_path = os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
                pto_lib_path = os.environ.get("PTO_LIB_PATH")
                if not pto_lib_path:
                    raise RuntimeError(
                        "PTO_LIB_PATH is not set. It must point to the pto-isa install root "
                        "(the directory containing 'include/pto/...') so the NPU kernels can be "
                        "compiled. Example: export PTO_LIB_PATH=/path/to/pto-isa"
                    )

                npu_arch = os.environ.get("NPU_ARCH", "dav-2201").strip()

                # Compile the shared library
                compile_cmd = [
                    "bisheng", "-O3", "-shared", "-fPIC", "-std=c++17", "-xcce", f"--npu-arch={npu_arch}",
                    "-I", self.include_dir,
                    "-I", f"{ascend_path}/include",
                    "-I", f"{pto_lib_path}/include",
                    "-L", f"{ascend_path}/lib64",
                    "-lascendcl", "-lruntime",
                    self.cpp_filename, "-o", self.so_filename
                ]
            
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Compilation failed:\n{result.stderr}")

    def __del__(self):
        # Safety net for callers that drop the builder without cleanup() (e.g. the
        # one-shot einsum()): free the device workspace so it does not leak.
        try:
            if getattr(self, "_workspace", None) is not None and getattr(self, "lib", None) is not None:
                self.lib.run_einsum_teardown(self._workspace)
                self._workspace = None
        except Exception:
            pass

    def cleanup(self):
        # Free the persistent workspace (while the library is still loaded).
        if self._workspace is not None and self.lib is not None:
            self.lib.run_einsum_teardown(self._workspace)
            self._workspace = None

        # Unload/clear CDLL reference to release system locks
        self.lib = None

        # Delete generated build directory and contents
        if self.target_dir and os.path.exists(self.target_dir):
            shutil.rmtree(self.target_dir)

    def load_library(self):
        self.lib = ctypes.CDLL(self.so_filename)
        
        transpose_sig = [
            ctypes.POINTER(self.c_type),  # input
            ctypes.POINTER(self.c_type),  # output
            ctypes.c_void_p               # stream
        ]
        
        if self.device == "npu":
            self.lib.run_transpose_inp0.argtypes = transpose_sig
            self.lib.run_transpose_inp0.restype = None
            
            self.lib.run_transpose_inp1.argtypes = transpose_sig
            self.lib.run_transpose_inp1.restype = None
            
            self.lib.run_transpose_out.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_void_p
            ]
            self.lib.run_transpose_out.restype = None
            
            self.lib.run_batched_matmul.argtypes = [
                ctypes.POINTER(self.c_type),  # ws0
                ctypes.POINTER(self.c_type),  # ws1
                ctypes.POINTER(ctypes.c_float),  # ws_res
                ctypes.c_void_p               # stream
            ]
            self.lib.run_batched_matmul.restype = None

        self.lib.run_einsum.argtypes = [
            ctypes.POINTER(self.c_type),  # input0
            ctypes.POINTER(self.c_type),  # input1
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_void_p               # stream
        ]
        self.lib.run_einsum.restype = None

        # Persistent-workspace path (NPU only): setup once, exec per call, free at cleanup.
        if self.device != "cpu":
            self.lib.run_einsum_setup.argtypes = [ctypes.c_void_p]  # stream
            self.lib.run_einsum_setup.restype = ctypes.c_void_p     # workspace
            self.lib.run_einsum_exec.argtypes = [
                ctypes.POINTER(self.c_type),     # input0
                ctypes.POINTER(self.c_type),     # input1
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_void_p,                 # workspace
                ctypes.c_void_p                  # stream
            ]
            self.lib.run_einsum_exec.restype = None
            self.lib.run_einsum_teardown.argtypes = [ctypes.c_void_p]  # workspace
            self.lib.run_einsum_teardown.restype = None

    def build(self):
        self.parse_equation()
        self.generate_code()
        self.compile()
        self.load_library()

        # Split-K configs have the cores atomic-add into the output, so it must start
        # zeroed (see _splitk_eligible). self.recipe is populated by generate_code().
        self._splitk = self._splitk_eligible()

        # Return a Callable wrapper that validates inputs
        def run(*operands):
            if len(operands) != 2:
                raise ValueError("Expected exactly two operands.")
            inp0, inp1 = operands
            
            # Validation checks
            if inp0.shape != self.input_shapes[0] or inp1.shape != self.input_shapes[1]:
                raise ValueError(
                    f"Input shapes {inp0.shape} and {inp1.shape} do not match built shapes {self.input_shapes}."
                )
            if inp0.dtype != self.dtype or inp1.dtype != self.dtype:
                raise TypeError(
                    f"Input datatypes {inp0.dtype} and {inp1.dtype} do not match built datatype {self.dtype}."
                )

            # Ensure inputs are contiguous so data_ptr() refers to a flat contiguous C-order array!
            if not inp0.is_contiguous():
                inp0 = inp0.contiguous()
            if not inp1.is_contiguous():
                inp1 = inp1.contiguous()

            if self.device == "cpu":
                output_tensor_npu = torch.zeros(self.recipe['out_interpret_shape'], dtype=torch.float32, device="cpu")
                stream_ptr = None
            elif self._splitk:
                # Split-K cores atomic-add their K-slices into the output, so it must
                # start zeroed. torch.zeros runs on the same stream before the kernel,
                # so ordering holds without an explicit sync.
                output_tensor_npu = torch.zeros(self.recipe['out_interpret_shape'], dtype=torch.float32, device="npu")
                stream_ptr = torch.npu.current_stream()._as_parameter_
            else:
                # The kernel writes every output element, so skip zero-init.
                output_tensor_npu = torch.empty(self.recipe['out_interpret_shape'], dtype=torch.float32, device="npu")
                stream_ptr = torch.npu.current_stream()._as_parameter_

            in0_ptr = ctypes.cast(inp0.data_ptr(), ctypes.POINTER(self.c_type))
            in1_ptr = ctypes.cast(inp1.data_ptr(), ctypes.POINTER(self.c_type))
            out_ptr = ctypes.cast(output_tensor_npu.data_ptr(), ctypes.POINTER(ctypes.c_float))

            if self.device == "cpu":
                self.lib.run_einsum(in0_ptr, in1_ptr, out_ptr, stream_ptr)
            else:
                # Allocate (and zero the contraction pad) once, then reuse across
                # calls; exec just launches on the stream (no per-call alloc/sync).
                if self._workspace is None:
                    self._workspace = self.lib.run_einsum_setup(stream_ptr)
                self.lib.run_einsum_exec(in0_ptr, in1_ptr, out_ptr, self._workspace, stream_ptr)
            return output_tensor_npu

        return run


def einsum(equation: str, *operands, device: str = "npu"):
    if len(operands) != 2:
        raise ValueError("Only exactly two operands are supported.")
    
    inp0, inp1 = operands
    if inp0.dtype != inp1.dtype:
        raise ValueError("Operands must have the same datatype.")
        
    shapes = [inp0.shape, inp1.shape]
    dtype = inp0.dtype

    # Build the pre-compiled ctypes execution callable
    runner = EinsumBuilder(equation, shapes, dtype, device=device).build()
    
    # Run and return tensor
    return runner(inp0, inp1)
