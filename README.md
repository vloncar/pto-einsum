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
  (`EINSUM_TILE_SIZE`, default 128); output dims are arbitrary — a boundary tile
  pads its operand up to a whole tile (zero-filled) and clamps the store to the real
  dim. The matmul K-loop is double-buffered.
- **2D transpose** uses the `TTRANS` hardware tile op (`vnchwconv`) rather than a
  scalar element copy — a large speedup on transposing equations. It is general for
  `float32`/`float16`, including large and *unaligned* (non-16-multiple) tensors.
  The blocked (large-tensor) transpose is distributed across all Vector cores.
- **Multi-batch-axis contractions.** Equations with two or more batch axes (batch
  + heads, e.g. attention `bshd, bthd -> bsht`) need an N-D non-identity transpose
  into the canonical `[batch, free, contract]` layout. These are handled by a
  strided gather (when the innermost axis is preserved) or a batched 2D `TTRANS`
  (when it moves), for both `K==C` and non-16-aligned `K!=C` contractions.
- **Elementwise (Hadamard) products.** An einsum with no contracted index and the
  same index order on both inputs and the output (e.g. `ts, ts -> ts` — the decay/
  mask step of linear attention `einsum("tn,sn,sp,ts->tp", …)` written as three
  two-way contractions) is a pure elementwise multiply, not a matmul. It is routed
  to a dedicated streaming Vector kernel (across all cores, any size) instead of a
  batch of degenerate 1×1×1 matmuls. Output is `float32`; `float16` inputs are
  up-cast and multiplied in float. Any `N` is handled directly — each block carries
  a dynamic valid extent so the short tail needs no padding or size floor.
- **Broadcast / scaling products.** An einsum with no contracted index where one
  operand carries every output axis and the other a strict subset (e.g. `bsd, d ->
  bsd` rms-norm scaling, `bshd, sd -> bshd` RoPE, `hqk, h -> hqk` ALiBi) is a pure
  broadcast multiply, routed to a dedicated Vector kernel (hardware column-expand or
  scalar multiply) rather than a matmul. NPU only; output is `float32`.
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
├── tests/
│   ├── reference.py       # pure-Python transpose + matmul reference (helper)
│   ├── test_transpose.py  # component: standalone transpose kernel
│   ├── test_matmul.py     # component: standalone transpose + Cube matmul vs reference
│   ├── test_einsum.py     # base: general einsum correctness + builder/validation/cleanup
│   └── test_llm_ops.py    # LLM: attention contractions, linear attention, broadcast/scaling
├── benchmarks/
│   ├── base/              # generic einsum primitives (bench_einsum.py, bench_transpose.py)
│   ├── complex/           # LLM workloads (bench_llm_contractions.py, bench_linear_attention.py)
│   └── bench_results/     # generated plots
├── IMPLEMENTATION.md      # kernel design notes
├── pyproject.toml / setup.cfg / setup.py
└── requirements.txt
```

## Running the tests

```bash
export PTO_LIB_PATH=/path/to/pto-isa
pytest                              # whole suite (testpaths=tests is configured)
```

A bare `pytest` from the repo root is the single entry point — `testpaths` in
`pyproject.toml` collects every module under `tests/`. The suite is split by scope:

- `test_transpose.py` / `test_matmul.py` — **components**: the standalone transpose
  and Cube-matmul kernels, checked against the pure-Python reference (`reference.py`).
- `test_einsum.py` — **base**: general end-to-end einsum correctness, with cases
  grouped by the kernel path they exercise (identity/transposed/blocked layouts,
  non-identity output, batched, `K!=C`, elementwise, degenerate free dims, split-K,
  partial tiles), plus builder reuse, input validation and cleanup. It also carries
  the **codegen gates** — proven-teeth self-tests that assert the generated config
  actually selects the intended path (`in_nt = N` read modes, `out_fusible = N`
  fused store) so a silent regression onto a slower-but-correct path fails loudly —
  and the **determinism guards** (bit-exact across repeated runs) for the pipelined
  and direct-input Cube paths. Known-unsupported equations are kept as `skip`s (with
  their loud-failure contract pinned) so they flip to real checks once a path lands.
- `test_llm_ops.py` — **LLM**: multi-batch-axis attention contractions, the
  linear-attention chain, and broadcast/scaling ops.

Verify across tile sizes, e.g. `EINSUM_TILE_SIZE=32 pytest tests/test_einsum.py`.

## Benchmark

Benchmarks compare the custom kernels against `torch.einsum` on the NPU and are
grouped into `base/` (generic einsum primitives) and `complex/` (realistic LLM
workloads). All plots are written to `benchmarks/bench_results/`.

**Base — primitive einsum patterns swept over tensor sizes:**

```bash
export PTO_LIB_PATH=/path/to/pto-isa
python benchmarks/base/bench_einsum.py            # all primitive benchmarks
python benchmarks/base/bench_einsum.py matmul     # a single benchmark
python benchmarks/base/bench_transpose.py         # 2D-transpose micro-benchmark
```

`bench_einsum.py` runs with no argument for the whole sweep, or pass a benchmark
name to run just one (`matmul`, `matmul-fp16`, `transpose`, `outer`, `dot-product`,
`batch-matmul`, `contraction`, `custom-layout`, `unaligned`, `batch-unaligned`); it
emits one image per benchmark plus a `summary.png` of each pattern's mean speedup
with its min/max range.

**Complex — realistic LLM workloads:**

```bash
python benchmarks/complex/bench_llm_contractions.py   # attention, SwiGLU, LoRA, MoE, RoPE, …
python benchmarks/complex/bench_linear_attention.py   # the 4-way linear-attention chain
```

`bench_llm_contractions.py` writes each pattern once as an einsum expression and runs
it with both `torch.einsum` and the custom kernel, reporting correctness (relative
diff) and per-pattern timing. Patterns the kernel does not cover are listed as
`unsupported` with the reason.

> **Note:** the JIT build cache keys only on the generated `.cpp` (config), not on
> the bundled headers. If you edit `src/pto_einsum/include/*.h`, delete the build
> cache (`rm -rf build`) before re-running so the change is recompiled.
