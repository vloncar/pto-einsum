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

- **Dispatch** (`build`/`run`): the returned callable validates shape/dtype, makes
  inputs contiguous, and launches on the current NPU stream via `ctypes`. The
  device workspace (intermediate transposed/contracted matrices) is allocated
  **once** on the first call and reused — `run_einsum_setup` allocates it and zeros
  the contraction-pad regions once (they stay zero because the transposes only
  overwrite the data columns and the matmul never writes `ws0`/`ws1`);
  `run_einsum_exec` then only launches the kernel (no `malloc`/`memset`/`free` and
  no `aclrtSynchronizeStream` — same-stream ordering serialises reuse and the
  output's downstream torch consumers, as `torch.einsum` is itself async);
  `run_einsum_teardown` frees the workspace at `cleanup()` (or on builder GC). This
  hoists the per-call allocator churn and host-device sync out of the hot path — a
  large win on the small/copy-bound cases, where they dominated (e.g. the K≠C
  batched-unaligned case dropped ~6×). The one-shot `run_einsum` (setup + exec +
  sync + teardown) is kept for the CPU path and non-reusing callers.

## 2. PTO / C++ side (`pto_einsum.h`)

`einsum()` runs the whole transpose → matmul → transpose pipeline in a **single
fused mix-kernel launch** (`einsum_fused_kernel`) where the AIC (Cube) and AIV
(Vector) cores cooperate, replacing the previous four separate kernel launches.
The phases are separated by full cross-core barriers (`SyncAll<false>()`):

```
[Vec ]  Phase A: transpose data0->ws0, data1->ws1
-- SyncAll<false>() --   (Vec -> Cube)
[Cube]  Phase B: ws0 @ ws1 -> ws_res
-- SyncAll<false>() --   (Cube -> Vec)
[Vec ]  Phase C: transpose ws_res->res
```

The host fetches the FFTS control address (`rtGetC2cCtrlAddr`, forward-declared to
avoid a profiling-header dependency; link `-lruntime`) and the kernel calls
`set_ffts_base_addr` before the barriers. Vector work is distributed over
`block_num*2` sub-block lanes (`block_idx*2 + subblockid`); Cube work over
`block_num`. The matmul body (`batched_matmul_inline`) and the transpose dispatch
(`transpose_inline`) are factored out so the standalone kernels and the fused
kernel share one implementation.

**Contraction padding without a host round-trip.** The Cube needs a `K`-wide
operand A and `K`-row operand B (`K = ⌈C/16⌉·16`; the Left-operand partial-C0
ND→NZ bug). The fused kernel allocates `ws0`/`ws1` K-padded, zeroes them once with
`aclrtMemsetAsync` (only when `K != C`), and the **input transposes write the
padded layout directly**:

- `ws0` (`A`): the contract dim is innermost, so the input0 transpose pads it from
  `C` to `K` per row via a `DST_PAD` template arg (2D `TTRANS` path) or
  `padded_copy_inline` (identity path). The pad is uniform across all `I·L0` rows,
  so any batch count works.
- `ws1` (`B`): the contract dim is the per-batch *row* count. With a single batch
  (or `K==C`) the one valid `[C, L1]` block sits at the front of the pre-zeroed
  `[K, L1]` buffer, so a plain transpose suffices. With `K!=C` *and* batching, each
  batch's `C` valid rows must land at the front of its `K`-row slot (batch stride
  `K·L1`); a contiguous write would mis-stride the batches, so `batched_pad_copy_inline`
  repacks them with a per-batch `K`-row pad.

This removes the old mid-pipeline `malloc`/`memcpy`/stream-sync (a ~21× speedup on
the degenerate outer-product case).

**Scope.** The fused path covers every supported config: `K==C` and `K!=C`, each
with any batch count. The multi-launch path (`einsum_multilaunch`, host-padding via
`batched_matmul`) is retained only as a reference / debugging fallback and is no
longer on the default dispatch.

### Phase 1 & 3 — Vector-core transpose

Each operand (and the final result) is reordered by a Vector-core kernel. Two
paths:

- **Identity permutation** → a straight tiled copy through the Unified Buffer
  (UB), 128 elements at a time. Handles arbitrary lengths.
- **2D non-identity permutation** → a **hardware** transpose via `TTRANS` (the
  PTO `vnchwconv`-based tile op). `TTRANS(dst, src, tmp)` transposes a RowMajor
  `src` UB tile into a `dst` UB tile using a scratch `tmp` tile; see below.

#### `TTRANS` — hardware tile transpose

`TTRANS` replaced the previous element-by-element scalar transpose (`SetValue`/
`GetValue` per element), which was very slow on the Vector core. It is a
**general** 2D transpose for our dtypes — no case-by-case fallback is needed at
our level:

- **Dtypes**: it supports b8/b16/b32; we use `float32` (b32) and `float16`
  (b16). The transpose unit works on `[16,8]→[8,16]` (b32) / `[16,16]→[16,16]`
  (b16) sub-tiles.
- **Arbitrary valid extents**: any `validRow × validCol` is accepted. `TTRANS`
  runs the rows in HW in 16-row sub-tiles and handles a `<16`-row Y-remainder
  with a *tiny library-internal* scalar loop (at most 15 rows per tile). This is
  the only residual scalar work and it lives inside the pto-isa library, not in
  our kernel.
- **Alignment is on the UB tile row-strides, not the logical dims**: `TTRANS`
  takes the fast HW path only when `srcStride % (32/sizeof(T)) == 0` and
  `dstStride % 16 == 0` (else it falls back internally to a fully scalar
  transpose). Those strides are the *allocated tile widths*, which we control, so
  we **pad** them — `srcTW = ⌈srcCols/blk⌉·blk` (`blk = 32/sizeof(T)`),
  `dstTW = tmpTW = ⌈srcRows/16⌉·16` — guaranteeing the HW path for any logical
  `srcRows × srcCols`.

Because `TTRANS` is UB-resident (src + dst + tmp tiles must all fit in UB), the
hybrid structure is kept, but each path now uses `TTRANS` for the transpose
itself:

  - *Whole-tensor* (the three padded tiles fit the ~184 KB UB budget): a single
    `TLOAD → TTRANS → TSTORE`. Both GM transfers are one (possibly strided) 2D
    move, so arbitrary (even unaligned) dims work — small odd-shaped tensors land
    here. A single tile is one unit of work, so only one Vec core runs it (these
    tensors are small by definition).
  - *Blocked* (too large for UB): `TTRANS` one `BR×BC` (64×64) block at a time,
    each block a single strided `TLOAD → TTRANS → TSTORE`. The blocks are
    independent, so the `⌈srcRows/BR⌉ × ⌈srcCols/BC⌉` block grid is flattened to a
    1D index space and **block-distributed across all Vec cores** (mirroring how
    the Cube matmul distributes its output-tile grid) — large transposes use the
    whole vector engine, not a single core. The per-block GM rows start at
    `r·srcCols` / `c·srcRows`, so this path still requires 16-aligned dims to keep
    every transfer 32-byte aligned (a **GM-DMA** constraint, not a `TTRANS` one; a
    `static_assert` enforces it; unaligned-large is not yet supported).

So the cases are: (1) any tensor that fits UB → fully general hardware transpose;
(2) large 16-aligned tensors → blocked hardware transpose; (3) large *unaligned*
tensors → not yet supported (blocked by the GM-DMA alignment, not by `TTRANS`).

### Phase 2 — Cube-core tiled matmul

The contraction is `C[i] = A[i] · B[i]` for each batch `i ∈ [0, I)`, with
`A[i]` an `L0 × C` matrix and `B[i]` a `C × L1` matrix.

**Fractal alignment.** The on-chip NZ `Mat`/`Left`/`Right`/`Acc` tiles require
dimensions that are multiples of 16 (NZ `Mat` InnerRows = 16; the `Acc` tile's
`fractalCSize` needs InnerRows = InnerCols = 16). 16 is also a multiple of the C0
block (fp32 = 8, fp16 = 16) so the `Mat` InnerCols = C0 constraint is satisfied
too. The logical dims are therefore padded up: `M = ⌈L0/16⌉·16`, `K = ⌈C/16⌉·16`,
`N = ⌈L1/16⌉·16`.

**Contraction padding.** The caller guarantees the GM `A`/`B` buffers are `K`-wide
and zero-padded along the contraction (the fused kernel's input transposes write
the padded layout directly; the multi-launch fallback pads host-side). This
sidesteps a Left-operand ND→NZ quirk where a *partial-C0* valid column count
misplaces rows across fractal blocks; loading a full `K`-wide `A` avoids that path,
and the zero padding contributes nothing to the sum.

**Tiling.** The per-batch output is partitioned into an `Mt × Nt` tile grid (tile
sizes are `min(config, padded-dim)`, multiples of 16). It requires
`M%Mt == N%Nt == K%Kt == 0` (full grid, no partial tiles), enforced by a
`static_assert`. All `(batch, m-tile, n-tile)` units are
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

- Support partial tiles (problem dims not divisible by the tile size), removing
  the matmul divisibility constraint.
- Support unaligned large 2D transposes, removing the blocked-transpose 16-aligned
  constraint.
