# Einsum Implementation Overview

`pto-einsum` turns a two-operand einsum equation into NPU kernels that are
JIT-compiled and dispatched on the active PyTorch NPU stream. The pipeline is:

```
input0 ─▶ transpose ─┐
                     ├─▶ batched matmul (Cube) ─▶ transpose ─▶ output
input1 ─▶ transpose ─┘
```

Everything is driven from Python; the heavy lifting is in two C++ headers that
encode the kernels with the Huawei PTO-ISA (built on Ascend C).

## Package structure

| File | Role |
|------|------|
| `src/pto_einsum/builder.py` | Equation parsing, plan/recipe extraction, code generation, JIT compilation (`bisheng`/`g++`), `ctypes` loading, dispatch. Public `einsum()` and `EinsumBuilder`. |
| `src/pto_einsum/templates.py` | C++ config-struct templates filled per equation. |
| `src/pto_einsum/include/pto_einsum.h` | NPU kernels: Vector-core transpose + tiled Cube matmul + host launchers. |
| `src/pto_einsum/include/cpu_einsum.h` | CPU reference kernels (validation / `device="cpu"`). |

## 1. Python side (`builder.py`)

The Python layer is a dynamic JIT compiler and launcher.

- **Equation parsing & validation** (`_validate_einsum_expr`, `parse_recipe`):
  the equation (e.g. `bij, bjk -> bik`) is validated, then each subscript is
  classified into four roles:
  - **inplace** (`I`): indices common to both inputs *and* the output — the batch
    dimensions.
  - **invariant0 / invariant1** (`L0` / `L1`): free indices that survive to the
    output, from input 0 and input 1 respectively.
  - **contract** (`C`): indices common to both inputs but absent from the output —
    the reduction (matmul K) dimension.

  These define a permutation of each operand into the canonical layout the Cube
  matmul expects: input0 → `[inplace, invariant0, contract]` (an `I·L0 × C`
  stack of matrices), input1 → `[inplace, contract, invariant1]` (an `I·C × L1`
  stack). The flattened products `L0`, `L1`, `C`, `I` are the matmul dims.

- **Configuration generation** (`transpose_config_gen`, `einsum_config_gen`):
  emits compile-time C++ structs describing the source/destination shapes, the
  permutation and its strides for each transpose, and the matmul dims plus the
  tile sizes (`tile_m/tile_n/tile_k`, from `EINSUM_TILE_SIZE`, default 128).

- **JIT compilation** (`generate_code`, `compile`): the configs are spliced into a
  C++ source that includes the kernel header, then compiled with `bisheng`
  (`--npu-arch`) for the NPU or `g++` for the CPU path. Output is keyed by an MD5
  of the generated source under the build dir and reused if present.
  `PTO_LIB_PATH` (pto-isa headers) is **required** for the NPU path.

- **Dispatch** (`build`/`run`): a device workspace buffer is allocated for the
  intermediate transposed/contracted matrices, inputs are made contiguous, and
  the compiled `run_einsum` is called via `ctypes` on the current NPU stream. The
  returned callable validates shape/dtype on each call so a built kernel can be
  reused.

## 2. PTO / C++ side (`pto_einsum.h`)

The three pipeline stages are launched by the host (`einsum()` in the header) as
separate kernels on a single stream, so they execute in order without an explicit
cross-core barrier. (A `SyncAll()` full-barrier helper is provided for future
fused kernels but is not used by the current pipeline.)

### Phase 1 & 3 — Vector-core transpose

Each operand (and the final result) is reordered by a Vector-core kernel. Two
paths:

- **Identity permutation** → a straight tiled copy through the Unified Buffer
  (UB), 128 elements at a time. Handles arbitrary lengths.
- **2D non-identity permutation** → a scalar transpose staged in UB. To keep UB
  bounded this is a **hybrid**:
  - *Whole-tensor* (fits the UB budget): load all of `src` contiguously,
    transpose element-by-element in UB, store all of `dst` contiguously. Both GM
    transfers are contiguous, so arbitrary (even unaligned) dims work.
  - *Blocked* (too large for UB): transpose in `BR×BC` source blocks, loading
    one source row at a time and storing one destination row at a time. The
    per-row GM offsets are `r·srcCols` / `c·srcRows`, so this path requires
    16-aligned dims to keep every transfer 32-byte aligned (a `static_assert`
    enforces it; unaligned-large is Stage 2).

### Phase 2 — Cube-core tiled matmul

The contraction is `C[i] = A[i] · B[i]` for each batch `i ∈ [0, I)`, with
`A[i]` an `L0 × C` matrix and `B[i]` a `C × L1` matrix.

**Fractal alignment.** The on-chip NZ `Mat`/`Left`/`Right`/`Acc` tiles require
dimensions that are multiples of 16 (NZ `Mat` InnerRows = 16; the `Acc` tile's
`fractalCSize` needs InnerRows = InnerCols = 16). 16 is also a multiple of the C0
block (fp32 = 8, fp16 = 16) so the `Mat` InnerCols = C0 constraint is satisfied
too. The logical dims are therefore padded up: `M = ⌈L0/16⌉·16`, `K = ⌈C/16⌉·16`,
`N = ⌈L1/16⌉·16`.

**Contraction padding (host).** `batched_matmul` guarantees the GM `A`/`B`
buffers are `K`-wide and zero-padded along the contraction. This sidesteps a
Left-operand ND→NZ quirk where a *partial-C0* valid column count misplaces rows
across fractal blocks; loading a full `K`-wide `A` avoids that path, and the zero
padding contributes nothing to the sum.

**Stage 1 tiling.** The per-batch output is partitioned into an `Mt × Nt` tile
grid (tile sizes are `min(config, padded-dim)`, multiples of 16). Stage 1
requires `M%Mt == N%Nt == K%Kt == 0` (full grid, no partial tiles — that is
Stage 2), enforced by a `static_assert`. All `(batch, m-tile, n-tile)` units are
flattened into a 1D index space and block-distributed across the AI cores. For
each output tile:

1. The accumulator (an FP32 `Acc` tile in L0C) is built by walking the
   contraction in `Kt`-wide steps: the first step uses `TMATMUL` (initialise),
   subsequent steps `TMATMUL_ACC` (accumulate). FP32 accumulation is mandatory
   even for fp16 inputs.
2. The result tile is stored to GM. Output is always FP32.

**GM windows carry real dims.** The ND→NZ loader takes its row/column transfer
counts (`nValue`/`dValue`) directly from the GM `Shape`, *not* from the tile's
valid extents. So the GM `Shape` for each tile must describe the real data
window — `validM × Kt` for `A`, `Kt × validN` for `B`, `validM × validN` for the
store — using `DYNAMIC` shape dims set at construction. Using the padded `Mt`/`Nt`
here would over-read past the physical `L0`/`L1` extent and corrupt boundary
tiles (this is exactly what breaks a degenerate `validN = 1` dot product if
missed).

### CPU reference (`cpu_einsum.h`)

A straightforward `transpose → triple-nested-loop matmul → transpose` in plain
C++, used to validate the NPU path. FP32 only.

## Build cache caveat

The build cache key is an MD5 of the **generated `.cpp`** (the config structs and
`extern "C"` wrappers), which does *not* include the kernel headers. After editing
`include/*.h`, delete the build dir (`rm -rf build`) before re-running, otherwise a
stale `.so` is reused and the change appears to have no effect.

## Roadmap

- **Stage 1.5 — double buffering.** The K-accumulation loop currently uses a
  coarse `pipe_barrier(PIPE_ALL)` between steps. Replacing it with ping-pong
  (double-buffered) L1↔L0 tiles will overlap the next load with the current
  matmul. The tile allocations are structured with this in mind.
- **Stage 2 — tails.** Support partial tiles (problem dims not divisible by the
  tile size) and unaligned large 2D transposes, removing the Stage 1
  divisibility/alignment constraints.
