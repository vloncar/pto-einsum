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

# Max A elements streamed per Vector block in the broadcast kernel (matches
# BCAST_TILECAP in pto_einsum.h); a ColExpand op whose innermost broadcast run exceeds
# this falls through to the matmul path (would need in-kernel column blocking).
BCAST_TILECAP = 4096
# The scalar-mode broadcast operand is loaded whole into UB once; keep it bounded.
_BCAST_SCALAR_SIZE_CAP = 8192

# Neutral broadcast fields for the non-broadcast recipe return paths (elementwise, matmul).
_NO_BCAST = dict(
    broadcast=False, bc_mode=0, bc_Cc=1, bc_Rr=1, bc_Inner=1, bc_Outer=1,
    bc_sizeB=1, bc_full_is_in0=True, bc_outer_dims=(), bc_outer_bstride=(),
)

class EinsumRecipe(TypedDict):
    in_transpose_idxs: tuple[tuple[int, ...], tuple[int, ...]]
    L0: int
    L1: int
    I: int
    C: int
    out_interpret_shape: tuple[int, ...]
    out_transpose_idxs: tuple[int, ...]
    # The true (post-transpose) output shape, i.e. out_interpret_shape permuted by
    # out_transpose_idxs. The kernel always writes `res` in this final order; only
    # for an identity output perm does it coincide with out_interpret_shape.
    out_shape: tuple[int, ...]
    # Pure elementwise (Hadamard) multiply: no contracted index, identical index
    # order on both inputs and the output (e.g. `TS,TS->TS`). Routed to the
    # elementwise kernel instead of the transpose->matmul->transpose pipeline; the
    # matmul fields above are then unused. n_elem is the total element count.
    elementwise: bool
    n_elem: int
    # Pure broadcast / scaling: empty contraction, one operand carries every output axis
    # (the "full" operand, in output order) and the other a strict subset, broadcast over
    # the axes it lacks (e.g. `bsd,d->bsd`, `bshd,sd->bshd`, `hqk,h->hqk`). Routed to the
    # Vector broadcast-multiply kernel (broadcast_mul). The matmul fields above are unused.
    # bc_mode: 0 = ColExpand (B varies along inner cols), 1 = scalar (B constant per group).
    broadcast: bool
    bc_mode: int
    bc_Cc: int           # ColExpand column-vector length (mode 0)
    bc_Rr: int           # ColExpand broadcast-row count (mode 0)
    bc_Inner: int        # per-group contiguous A region (Rr*Cc for mode 0, inner block mode 1)
    bc_Outer: int        # number of output groups
    bc_sizeB: int        # element count of the broadcast operand
    bc_full_is_in0: bool # which raw operand is the full (contiguous) one
    bc_outer_dims: tuple      # OUTER axis sizes, outermost first
    bc_outer_bstride: tuple   # broadcast-operand element stride per OUTER axis

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
        s_alphabets = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

        # Only named axes [a-zA-Z] are accepted; broadcasting ('...') is not supported.
        if not (s_alphabets >= set(in0 + in1 + out)):
            raise ValueError(f"einsum string {fn} is invalid: subscripts should be in [a-zA-Z]")

        ax_in0, ax_in1, ax_out = list(in0), list(in1), list(out)
        sax_in0, sax_in1, sax_out = set(ax_in0), set(ax_in1), set(ax_out)

        for label, subs in (("input0", in0), ("input1", in1), ("output", out)):
            for a in subs:
                if subs.count(a) > 1:
                    raise ValueError(f"einsum string {fn} is invalid: {label} subscripts includes '{a}' multiple times")

        if remaining := sax_out - sax_in0 - sax_in1:
            raise ValueError(f'einsum string {fn} is invalid: output subscripts {remaining} not found in inputs')

        if len(sax_in0) != len(shape0):
            raise ValueError(f'Input0 requires {len(sax_in0)} dimensions, but {len(shape0)} is given')
        if len(sax_in1) != len(shape1):
            raise ValueError(f'Input1 requires {len(sax_in1)} dimensions, but {len(shape1)} is given')

        _common_in = sax_in0 & sax_in1
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
        fn, out_shape = self._validate_einsum_expr(fn, input_shape0, input_shape1)

        _in, _out = fn.split('->')
        _in0, _in1 = _in.split(',')

        in0, in1, out = list(_in0), list(_in1), list(_out)

        # Pure elementwise (Hadamard) multiply: identical index order on both inputs
        # and the output => no contracted index, no transpose, no broadcast, so
        # res[i] = in0[i] * in1[i] over the flat buffers. Route to the elementwise
        # kernel rather than emitting a batch of degenerate 1x1x1 matmuls. The
        # matmul-oriented recipe fields are filled with neutral values (unused). Any N
        # is handled directly: the kernel's per-block dynamic valid extent covers the
        # short tail, so there is no minimum-size fallback.
        if in0 == in1 == out:
            return EinsumRecipe(
                in_transpose_idxs=((), ()),
                out_interpret_shape=out_shape,
                out_transpose_idxs=tuple(range(len(out_shape))),
                out_shape=out_shape,
                L0=1, L1=1, I=prod(out_shape), C=1,
                elementwise=True,
                n_elem=prod(out_shape),
                **_NO_BCAST,
            )

        s_in0, s_in1, s_out = set(in0), set(in1), set(out)
        _common = s_in0 & s_in1
        _contract = _common - s_out
        _inplace = _common & s_out

        # Single-operand reduction axes: an index that appears in exactly one operand
        # and not in the output. torch.einsum sums these out, but the transpose->matmul
        # pipeline has no reduce stage and the axis fits none of the contract/invariant/
        # inplace buckets below, so it would be silently dropped (wrong result). Reject
        # it with a clean error until an in-pipeline Vector reduction lands (roadmap:
        # "Single-operand reduction axes", IMPLEMENTATION.md).
        _solo = (s_in0 - s_in1 - s_out) | (s_in1 - s_in0 - s_out)
        if _solo:
            raise ValueError(
                f"einsum '{fn}': index {sorted(_solo)} appears in only one operand and "
                "is reduced away; single-operand reduction is not supported.")

        # Pure broadcast / scaling: no contracted index and exactly one operand carries
        # every output axis in output order; the other is a strict subset (broadcast). Route
        # to the Vector broadcast kernel instead of a batch of degenerate 1xC matmuls (which
        # would also trip the batched-partial-tile guard). The both-subset case (outer
        # product, e.g. `bi,bj->bij`) is a genuine rank-1 matmul and stays on the Cube path.
        if not _contract:
            bc = self._try_broadcast(in0, in1, out, input_shape0, input_shape1, out_shape)
            if bc is not None:
                return bc
        contract = sorted(_contract, key=lambda x: in1.index(x))
        inplace = sorted(_inplace, key=lambda x: in1.index(x))
        invariant0 = sorted((s_out - _common) & s_in0, key=lambda x: in0.index(x))
        invariant1 = sorted((s_out - _common) & s_in1, key=lambda x: in1.index(x))

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
        transpose_idx1 = inplace_idxs[1] + contract_idxs[1] + invariant_idxs[1]

        out_shape_pretranspose = inplace_shape + invariant_shape0 + invariant_shape1
        _out_transpose_idx = np.argsort(tuple(map(out.index, inplace + invariant0 + invariant1)))
        out_transpose_idx = tuple(int(i) for i in _out_transpose_idx)

        # Note: the Cube kernel pads L0/L1/C up to fractal granularity internally
        # (see batched_matmul_kernel_standalone), so no multiple-of-16 restriction on
        # the logical dims is required here — arbitrary shapes (including degenerate
        # L0/L1/C == 1) are supported.

        return EinsumRecipe(
            in_transpose_idxs=(transpose_idx0, transpose_idx1),
            out_interpret_shape=out_shape_pretranspose,
            out_transpose_idxs=out_transpose_idx,
            out_shape=out_shape,
            L0=invariant_size0,
            L1=invariant_size1,
            I=inplace_size,
            C=contract_size,
            elementwise=False,
            n_elem=prod(out_shape),
            **_NO_BCAST,
        )

    def _try_broadcast(self, in0, in1, out, shape0, shape1, out_shape):
        """Classify a no-contraction einsum as a broadcast/scaling op, or return None to
        fall through to the matmul path. Requires the full operand to be in output order and
        the broadcast operand's axes to appear in output order (no transpose needed)."""
        # The broadcast kernel is NPU-only; on CPU these fall through to the matmul path,
        # which the generic CPU einsum reference evaluates correctly.
        if self.device == "cpu":
            return None
        s_in0, s_in1, s_out = set(in0), set(in1), set(out)
        # Exactly one operand carries every output axis (the "full" one); the other a
        # strict subset. (Both-full = transpose-elementwise; both-subset = outer product.)
        full_is_in0 = (s_in0 == s_out)
        full_is_in1 = (s_in1 == s_out)
        if full_is_in0 == full_is_in1:
            return None
        full, bc = (in0, in1) if full_is_in0 else (in1, in0)
        # The full operand must already be in output order (else it needs a transpose).
        if full != out:
            return None
        n = len(out)
        sizes = list(out_shape)
        s_bc = set(bc)
        bc_positions = [out.index(ax) for ax in bc]
        if bc_positions != sorted(bc_positions):  # broadcast axes out of output order
            return None

        # Broadcast-operand element stride per output axis (0 where B is absent), built from
        # the innermost axis outward (B is contiguous in output order).
        bstride = [0] * n
        running = 1
        for p in range(n - 1, -1, -1):
            if out[p] in s_bc:
                bstride[p] = running
                running *= sizes[p]
        size_b = prod(sizes[p] for p in bc_positions) if bc_positions else 1

        present = [out[p] in s_bc for p in range(n)]
        if present[n - 1]:
            mode = 0  # ColExpand: B varies along the innermost contiguous run
            p = n - 1
            Cc = 1
            while p >= 0 and present[p]:
                if Cc * sizes[p] > BCAST_TILECAP:
                    return None  # innermost B-present run exceeds a UB tile; needs C-blocking
                Cc *= sizes[p]; p -= 1
            Rr = 1
            while p >= 0 and not present[p]:
                Rr *= sizes[p]; p -= 1
            inner = Rr * Cc
        else:
            mode = 1  # scalar: B is constant across the inner block
            if size_b > _BCAST_SCALAR_SIZE_CAP:
                return None  # broadcast operand loaded whole into UB; keep it small
            p = n - 1
            inner = 1
            while p >= 0 and not present[p]:
                inner *= sizes[p]; p -= 1
            Cc, Rr = 1, 1
        outer_axes = list(range(0, p + 1))      # output positions above the inner block
        outer_dims = tuple(sizes[q] for q in outer_axes)
        outer_bstride = tuple(bstride[q] for q in outer_axes)
        outer = prod(outer_dims) if outer_dims else 1

        return EinsumRecipe(
            in_transpose_idxs=((), ()),
            out_interpret_shape=out_shape,
            out_transpose_idxs=tuple(range(n)),
            out_shape=out_shape,
            L0=1, L1=1, I=prod(out_shape), C=1,
            elementwise=False,
            n_elem=prod(out_shape),
            broadcast=True,
            bc_mode=mode,
            bc_Cc=Cc, bc_Rr=Rr, bc_Inner=inner, bc_Outer=outer,
            bc_sizeB=size_b, bc_full_is_in0=full_is_in0,
            bc_outer_dims=outer_dims, bc_outer_bstride=outer_bstride,
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
        # Match MatmulGeom's full-tile grid: a partial free dim is padded up to a whole
        # tile (Ma/Na), so the tile count ceils over the padded dim.
        ceil = lambda a, b: -(-a // b)
        total_tiles = I * ceil(Mpad, Mt) * ceil(Npad, Nt)
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

        # Partial output tiles (a free dim not a multiple of the matmul tile) need the
        # operands padded up to a whole tile, which currently only has a single-batch
        # layout — so reject batched (I>1) partial configs with a clean error rather
        # than a kernel static_assert. The contraction dim never gates (tile_k divides
        # padded K); only the free dims L0/L1 matter.
        L0, L1, I = self.recipe['L0'], self.recipe['L1'], self.recipe['I']
        if I > 1 and not self.recipe['elementwise'] and not self.recipe['broadcast']:
            pad16 = lambda x: (x + 15) // 16 * 16
            tile = int(os.getenv("EINSUM_TILE_SIZE", "128"))
            Mpad, Npad = pad16(L0), pad16(L1)
            Mt, Nt = min(tile, Mpad), min(tile, Npad)
            if Mpad % Mt != 0 or Npad % Nt != 0:
                raise ValueError(
                    f"batched einsum (I={I}) requires each output free dim to be a "
                    f"multiple of the matmul tile ({tile}) once padded to 16; got "
                    f"L0={L0} (Mpad={Mpad}), L1={L1} (Npad={Npad}). Partial tiles are "
                    f"currently single-batch (I==1) only.")

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

        # Elementwise (Hadamard) path: a single config + a kernel call, no transpose
        # or matmul configs. Reuses the run_einsum* entry-point names so the dispatch
        # in run()/load_library is unchanged.
        if self.recipe['elementwise']:
            config_code = templates.elementwise_config_template.format(N=self.recipe['n_elem'])
            tmpl = templates.cpu_lib_elementwise_template if self.device == "cpu" \
                else templates.shared_lib_elementwise_template
            self.cpp_code = tmpl.format(config_code=config_code, data_t=data_t)
            self.code_hash = hashlib.md5(self.cpp_code.encode('utf-8')).hexdigest()
            self.target_dir = os.path.join(self.build_base, self.code_hash)
            return

        # Broadcast / scaling path: a single config + a Vector kernel call (NPU only).
        if self.recipe['broadcast']:
            r = self.recipe
            outer_dims = r['bc_outer_dims'] or (1,)        # pad to >=1 element (rank 0 -> dummy)
            outer_bstride = r['bc_outer_bstride'] or (0,)
            config_code = templates.broadcast_config_template.format(
                mode=r['bc_mode'], N=r['n_elem'], Cc=r['bc_Cc'], Rr=r['bc_Rr'],
                Inner=r['bc_Inner'], Outer=r['bc_Outer'], sizeB=r['bc_sizeB'],
                outer_rank=len(r['bc_outer_dims']), OR=max(1, len(r['bc_outer_dims'])),
                outer_dims=', '.join(map(str, outer_dims)),
                outer_bstride=', '.join(map(str, outer_bstride)),
            )
            full, bcast = ('input0', 'input1') if r['bc_full_is_in0'] else ('input1', 'input0')
            self.cpp_code = templates.shared_lib_broadcast_template.format(
                config_code=config_code, data_t=data_t, full=full, bcast=bcast)
            self.code_hash = hashlib.md5(self.cpp_code.encode('utf-8')).hexdigest()
            self.target_dir = os.path.join(self.build_base, self.code_hash)
            return

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

        tmpl = templates.cpu_lib_template if self.device == "cpu" else templates.shared_lib_template
        self.cpp_code = tmpl.format(
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
        
        # The elementwise and broadcast paths expose only the run_einsum* entry points (no
        # transpose / batched_matmul symbols exist in their .so), so skip their setup.
        if self.device == "npu" and not self.recipe['elementwise'] and not self.recipe['broadcast']:
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
        # The elementwise and broadcast paths have no matmul, so they are never split-K.
        self._splitk = False if (self.recipe['elementwise'] or self.recipe['broadcast']) \
            else self._splitk_eligible()

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
                output_tensor_npu = torch.zeros(self.recipe['out_shape'], dtype=torch.float32, device="cpu")
                stream_ptr = None
            elif self._splitk:
                # Split-K cores atomic-add their K-slices into the output, so it must
                # start zeroed. torch.zeros runs on the same stream before the kernel,
                # so ordering holds without an explicit sync.
                output_tensor_npu = torch.zeros(self.recipe['out_shape'], dtype=torch.float32, device="npu")
                stream_ptr = torch.npu.current_stream()._as_parameter_
            else:
                # The kernel writes every output element, so skip zero-init.
                output_tensor_npu = torch.empty(self.recipe['out_shape'], dtype=torch.float32, device="npu")
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
