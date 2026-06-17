# pto-einsum

A JIT-compiled [`einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
for Huawei Ascend NPUs (the **a2a3** SoC, e.g. Ascend 910B4), built on the
**PTO-ISA** kernel programming paradigm.

Given an einsum equation and two operands, the package parses the equation into a
transpose → batched-matmul → transpose plan, generates the matching C++ kernel
configuration, compiles it for the NPU with the CANN `bisheng` compiler, and
dispatches it on the active PyTorch NPU stream. Compiled kernels are cached on
disk and the resulting callable can be reused across calls.

```python
import torch, torch_npu
from pto_einsum import einsum

a = torch.rand(256, 256).npu()
b = torch.rand(256, 256).npu()
c = einsum("ij, jk -> ik", a, b)        # runs on the NPU, returns an NPU tensor
```

## Status

- Targets the **a2a3** architecture only.
- Exactly **two** operands per equation.
- Dtypes: `float32` and `float16` (NPU); `float32` on the CPU reference path.
- **Tiling**: the Cube matmul and the 2D transpose are tiled so problems no longer
  need to fit entirely on-chip. The matmul tile size is configurable
  (`EINSUM_TILE_SIZE`, default 128); the (fractal-padded) problem dims must divide
  evenly by the tile size (arbitrary tails are not yet supported). The matmul
  K-loop is double-buffered.
- **2D transpose** uses the `TTRANS` hardware tile op (`vnchwconv`) rather than a
  scalar element copy — a large speedup on transposing equations. It is general
  for `float32`/`float16`; large *unaligned* 2D transposes are not yet supported.
  The blocked (large-tensor) transpose is distributed across all Vector cores.
- **Single fused kernel.** The transpose → matmul → transpose pipeline runs in one
  mix-kernel launch (Cube + Vector cores cooperating, full cross-core barriers via
  FFTS) instead of four separate launches — removing launch overhead and host
  round-trips. Contraction padding for non-16-aligned `K` is written directly by
  the transposes into a K-padded workspace (no mid-pipeline host copy/sync),
  covering `K==C` and `K!=C` for any batch count.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the kernel-level details.

## Requirements

This package compiles and runs real NPU kernels, so it needs an Ascend
environment — it cannot run on a plain CPU host:

- Python ≥ 3.10
- Huawei **CANN** toolkit (developed against 9.0.0) providing the `bisheng`
  compiler and `libascendcl`
- An Ascend NPU + driver (a2a3 / 910B4)
- The **pto-isa** header library (separate repository)
- `torch`, `torch_npu` (matching the CANN version), `numpy`

Python dependencies are pinned in [requirements.txt](requirements.txt). Note that
`torch_npu` is only installable on a host with the matching CANN toolkit.

### Environment variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `PTO_LIB_PATH` | **yes** (NPU) | — | Root of the pto-isa install (contains `include/pto/...`) |
| `ASCEND_HOME_PATH` | no | `/usr/local/Ascend/ascend-toolkit/latest` | CANN toolkit root |
| `NPU_ARCH` | no | `dav-2201` | `bisheng` target architecture |
| `EINSUM_TILE_SIZE` | no | `128` | Cube matmul tile size (multiple of 16) |
| `EINSUM_BUILD_DIR` | no | `./build` | Where generated sources / compiled `.so` are cached |
| `EINSUM_TEST_DEVICE` | no | `npu` | Device used by the test suite (`npu` or `cpu`) |

## Installation

From the project root, in your Ascend Python environment:

```bash
pip install -e .            # editable install (setuptools >= 64)
```

On older toolchains (e.g. the setuptools shipped inside some CANN containers)
use the legacy path:

```bash
python setup.py develop --no-deps
```

Then make the pto-isa headers discoverable:

```bash
export PTO_LIB_PATH=/path/to/pto-isa
```

## Usage

One-shot call:

```python
from pto_einsum import einsum
c = einsum("bij, bjk -> bik", a, b)     # a, b are NPU tensors
```

Reuse a compiled kernel across many inputs of the same shape/dtype:

```python
from pto_einsum import EinsumBuilder

runner = EinsumBuilder("ij, jk -> ik", [(256, 256), (256, 256)], torch.float32).build()
for a, b in batches:
    c = runner(a, b)
```

The CPU reference implementation (used for validation) is selected with
`device="cpu"`:

```python
c = einsum("ij, jk -> ik", a_cpu, b_cpu, device="cpu")
```

## Project layout

```
pto-einsum/
├── src/pto_einsum/
│   ├── __init__.py        # public API: einsum, EinsumBuilder
│   ├── builder.py         # equation parsing, codegen, JIT compile, dispatch
│   ├── templates.py       # C++ config-struct templates
│   └── include/
│       ├── pto_einsum.h   # NPU kernels (transpose + tiled Cube matmul)
│       └── cpu_einsum.h   # CPU reference kernels
├── tests/                 # pytest suites (einsum, split, transpose, python ref)
├── benchmarks/            # bench_einsum.py, debug_einsum.py
├── IMPLEMENTATION.md      # kernel design notes
├── pyproject.toml / setup.cfg / setup.py
└── requirements.txt
```

## Running the tests

```bash
export PTO_LIB_PATH=/path/to/pto-isa
pytest tests/test_einsum.py tests/test_split_npu.py
python tests/test_transpose_npu.py
```

Verify across tile sizes, e.g. `EINSUM_TILE_SIZE=32 pytest tests/test_einsum.py`.

## Benchmark

```bash
export PTO_LIB_PATH=/path/to/pto-isa
python benchmarks/bench_einsum.py             # all benchmarks
python benchmarks/bench_einsum.py matmul      # a single benchmark
```

Compares the custom kernels against `torch.einsum` on the NPU. Each benchmark is
an einsum pattern swept over a range of tensor sizes (for scaling behaviour).
With no argument all benchmarks run; pass a benchmark name to run just one
(`matmul`, `matmul-fp16`, `transpose`, `outer`, `dot-product`, `batch-matmul`,
`contraction`, `custom-layout`, `unaligned`, `batch-unaligned`). Plots are
written to `benchmarks/bench_results/`: one image per benchmark plus a
`summary.png` showing each pattern's mean speedup with its min/max range.

> **Note:** the JIT build cache keys only on the generated `.cpp` (config), not on
> the bundled headers. If you edit `src/pto_einsum/include/*.h`, delete the build
> cache (`rm -rf build`) before re-running so the change is recompiled.
