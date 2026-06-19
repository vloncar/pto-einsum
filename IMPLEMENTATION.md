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

---

# Part 1 — The core algorithm

This part describes the baseline transpose → matmul → transpose pipeline as if no
performance work had been layered on top. The optimizations in Part 2 are all
refinements of this skeleton; none of them change *what* is computed.

## 1.1 Python side (`builder.py`)

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

  **Elementwise special case.** When both inputs and the output carry the *same
  index list in the same order* (e.g. `ts, ts -> ts`) there is no contracted index
  (`C == 1`), no transpose and no broadcast — the einsum is a pure Hadamard product
  `res[i] = in0[i] * in1[i]`. `parse_recipe` flags this (`recipe.elementwise`) and
  routes it — for any `N` — to a dedicated elementwise kernel instead of the matmul
  pipeline (which would otherwise issue `N` degenerate `1×1×1` matmuls). See §1.3.

- **Configuration generation** (`transpose_config_gen`, `einsum_config_gen`):
  emits compile-time C++ structs describing the source/destination shapes, the
  permutation and its strides for each transpose, and the matmul dims plus the
  tile sizes (`tile_m`/`tile_n`/`tile_k`).

- **JIT compilation** (`generate_code`, `compile`): the configs are spliced into a
  C++ source that includes the kernel header, then compiled with `bisheng`
  (`--npu-arch`) for the NPU or `g++` for the CPU path. Output is keyed by an MD5
  of the generated source under the build dir and reused if present.
  `PTO_LIB_PATH` (pto-isa headers) is **required** for the NPU path.

- **Dispatch** (`build`/`run`): the returned callable validates shape/dtype, makes
  inputs contiguous, and launches on the current NPU stream via `ctypes`.

## 1.2 PTO / C++ side (`pto_einsum.h`)

`einsum()` runs the whole transpose → matmul → transpose pipeline in a **single
fused mix-kernel launch** (`einsum_fused_kernel`) where the AIC (Cube) and AIV
(Vector) cores cooperate. The phases are separated by full cross-core barriers
(`SyncAll<false>()`):

```
[Vec ]  Phase A: transpose data0->ws0, data1->ws1
-- SyncAll<false>() --   (Vec -> Cube)
[Cube]  Phase B: ws0 @ ws1 -> ws_res
-- SyncAll<false>() --   (Cube -> Vec)
[Vec ]  Phase C: transpose ws_res->res
```

Vector work is distributed over `block_num*2` sub-block lanes
(`block_idx*2 + subblockid`); Cube work over `block_num`. The matmul body
(`batched_matmul_inline`) and the transpose dispatch (`transpose_inline`) are
factored out so the standalone kernels and the fused kernel share one
implementation. The host fetches the FFTS control address (`rtGetC2cCtrlAddr`,
forward-declared to avoid a profiling-header dependency; link `-lruntime`) and the
kernel calls `set_ffts_base_addr` before the barriers.

### Phases A & C — Vector-core transpose

Each operand (and, in the general case, the final result) is reordered by a
Vector-core kernel. The dispatch (`transpose_inline`) has these paths:

- **Identity permutation** (any rank) → a straight tiled copy through the Unified
  Buffer (UB), 128 elements at a time (`transpose_copy_inline`). Handles arbitrary
  lengths.
- **2D non-identity permutation** → a **hardware** transpose via `TTRANS` (the
  PTO `vnchwconv`-based tile op). `TTRANS(dst, src, tmp)` transposes a RowMajor
  `src` UB tile into a `dst` UB tile using a scratch `tmp` tile.
- **N-D (rank > 2) non-identity permutation** → see *N-D transposes* below. These
  arise when an operand has two or more batch axes (e.g. attention's batch + heads,
  `bshd,bthd->bsht`) and must be permuted into the canonical `[batch, free,
  contract]` matmul layout.

#### `TTRANS` — hardware tile transpose

`TTRANS` replaced an earlier element-by-element scalar transpose (`SetValue`/
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
implementation (`transpose_2d_inline`) has two structural cases:

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
  `r·srcCols` / `c·srcRows`, so this path requires 16-aligned dims to keep
  every transfer 32-byte aligned. The block *base* address is always 32-byte
  aligned (`r0`/`c0` are multiples of `BR`/`BC` = 64), and the GM↔UB transfers are
  element-granular ND copies, so a non-16-aligned row stride (`srcCols` / `dpad`)
  only shifts successive rows to sub-32B offsets — which the ND DMA handles — so
  arbitrary (unaligned) dims work here too, validated on dav-2201. Boundary
  blocks are partial (`< BR/BC`); each `TLOAD`/`TSTORE` therefore carries a
  **DYNAMIC GM window of the real `rb/cb`** dims, not the padded `BR×BC` — a static
  window over-reads the source and over-writes the neighbouring output block (a
  latent bug masked while every blocked test used 64-multiple dims).

So the cases are: (1) any tensor that fits UB → fully general hardware transpose;
(2) large tensors → blocked hardware transpose, arbitrary (including unaligned and
odd) dims — the dynamic per-block GM window carries the real `rb`/`cb`, and the
element-granular ND DMA tolerates non-32-byte-aligned row strides.

#### N-D transposes

The Cube matmul already flattens *all* batch (`n_inplace`) axes into one batch
loop, so an einsum with multiple batch axes (batch + heads, …) needs nothing new
from the contraction — only an N-D-aware Phase A/C. A permutation that maps the
operand into the grouped `[batch, free, contract]` layout always has one of two
shapes, distinguished by the innermost output axis. Both read the config's
`to_shape` (output extents) and `perm_strides` (each output axis' source stride):
for any permutation the contiguous output block over the trailing axes maps to a
strided source block, so the **destination stays the natural matmul layout** and
only the source side carries the permutation.

- **Innermost axis preserved** (`perm[dims-1] == dims-1`) → a strided gather
  (`transpose_nd_copy_inline`): each contiguous inner run is copied to its natural
  dst slot, its source base offset the mixed-radix sum of the outer indices times
  `perm_strides`. No element transpose. Covers the inputs/output whose inner axis
  is unchanged (e.g. `bshd→bhsd`, contract `d` stays innermost). Inherits the
  `DST_PAD` contract-padding the 2D path uses.
- **Innermost axis changes** → a **batched** 2D `TTRANS`
  (`transpose_nd_batched_2d_inline`): the trailing two output axes form the
  transpose, all leading axes are batch. Per batch the `[Bb, A]` strided source
  block (row stride `perm_strides[-1]`, the second-innermost axis source-
  contiguous) is `TTRANS`-d into its contiguous `[A, Bb]` dst slot
  (`ttrans_block_inline`, the 2D fits-UB body parameterised by strides). Batches
  are distributed across the Vec lanes.

The contract padding (`K!=C`) is threaded through both N-D paths. The contract
dim sits at a different axis for the two operands, so each pads a different
destination stride:

- **input0** (`[batch.., L0, C]`, contract = innermost): the batched-`TTRANS` path
  widens each transposed row's stride `C→K` (`PAD_INNER`/`PAD_STRIDE` → `DST_RS`),
  matching the gather/2D paths' `DST_PAD`.
- **input1** (`[batch.., C, L1]`, contract = second-innermost row count): a
  non-identity input1 is transposed *and* row-padded by `transpose_inline_rowpad`,
  which lands each batch's `C` valid rows at the front of its `K`-row slot —
  per-batch dst block `C·L1 → K·L1` (the gather's `(ROWS, PAD_BATCH)` knob for the
  innermost-preserved case, `transpose_nd_batched_2d_inline`'s `DST_BLOCK` override
  for the innermost-changes case). An *identity* input1 stays the contiguous
  `batched_pad_copy_inline` repack.

The pad regions are pre-zeroed once and the transposes write only the valid window,
so the padding holds across calls (the §2.1 persistent-workspace invariant).

A per-batch `[Bb, A]` block is transposed whole when it fits UB; when it does not,
it is tiled into `BR×BC` sub-blocks and the `(batch, sub-block)` grid is flattened
into a 1D work space across the Vec lanes (so one large per-batch transpose uses the
whole vector engine, like the 2D blocked path). Boundary sub-blocks are partial, so
each `TLOAD`/`TSTORE` carries a **DYNAMIC GM window set to the real `rb/cb`** dims
(`ttrans_block_dyn_inline`) — a static `BR×BC` window would over-read the source and
over-write neighbouring output (the same "GM windows carry real dims" rule the
matmul follows; the 2D blocked path was fixed the same way, it had a latent
partial-block bug masked by only-64-multiple tests).

Still guarded by `static_assert`: the `srcColStride==1` requirement that the
second-innermost output axis be source-contiguous in the batched-`TTRANS` case.
Lifting it needs a non-contiguous gather (the GM DMA has no strided-inner mode), a
separate mechanism that overlaps the broadcast-op roadmap item — not part of the
`K!=C` / large-tile scope.

### Phase B — Cube-core tiled matmul

The contraction is `C[i] = A[i] · B[i]` for each batch `i ∈ [0, I)`, with
`A[i]` an `L0 × C` matrix and `B[i]` a `C × L1` matrix.

**Fractal alignment.** The on-chip NZ `Mat`/`Left`/`Right`/`Acc` tiles require
dimensions that are multiples of 16 (NZ `Mat` InnerRows = 16; the `Acc` tile's
`fractalCSize` needs InnerRows = InnerCols = 16). 16 is also a multiple of the C0
block (fp32 = 8, fp16 = 16) so the `Mat` InnerCols = C0 constraint is satisfied
too. The logical dims are therefore padded up: `M = ⌈L0/16⌉·16`, `K = ⌈C/16⌉·16`,
`N = ⌈L1/16⌉·16`.

**Contraction padding (the requirement).** The Cube must be fed a `K`-wide
operand A and `K`-row operand B that are zero-padded along the contraction. This
sidesteps a Left-operand ND→NZ quirk where a *partial-C0* valid column count
misplaces rows across fractal blocks; loading a full `K`-wide `A` avoids that path,
and the zero padding contributes nothing to the sum. *How* the padded buffers are
produced without a host round-trip is an optimization — see §2.2.

**Tiling.** The per-batch output is partitioned into an `Mt × Nt` tile grid (tile
sizes are `min(config, padded-dim)`, multiples of 16). The grid tiles over the
*operand-padded* extents `Ma = ⌈M/Mt⌉·Mt` and `Na = ⌈N/Nt⌉·Nt` (and `K`, which
`tile_k` always divides), so every tile is full — `Ma%Mt == Na%Nt == K%Kt == 0`.
A free dim that is not a whole tile multiple (e.g. `L0 = 200 → M = 208 → Ma = 256`)
is handled by padding the *operand buffer* up to `Ma` rows / `Na` cols, zero-filled
(see the "GM windows carry real dims" load/store note below); the output store still
clamps to the real `L0/L1`, so output dims are arbitrary. All `(batch, m-tile, n-tile)` units
are flattened into a 1D index space (`total_tiles = I·(Ma/Mt)·(Na/Nt)`) and
block-distributed across the AI cores. For each output tile
(`matmul_one_tile_inline`):

1. The accumulator (an FP32 `Acc` tile in L0C) is built by walking the
   contraction in `Kt`-wide steps: the first step uses `TMATMUL` (initialise),
   subsequent steps `TMATMUL_ACC` (accumulate). FP32 accumulation is mandatory
   even for fp16 inputs.
2. The result tile is stored to GM. Output is always FP32.

`matmul_one_tile_inline` is the baseline tile shown here (and the split-K tile of
§2.6); the plain schedule actually runs each tile through the deeper-pipelined
`matmul_one_tile_deep` (§2.7), which computes the same thing.

**GM windows carry real dims.** The ND→NZ loader takes its row/column transfer
counts (`nValue`/`dValue`) directly from the GM `Shape`, *not* from the tile's
valid extents (set via `DYNAMIC` shape dims at construction). Two regimes:

- **Store** always uses the *real* window `validM × validN` (`validM = min(L0−row0,
  Mt)`), with stride `L1` over the unpadded output. Using the padded `Mt`/`Nt` here
  would over-write past the physical output (it is exactly what breaks a degenerate
  `validN = 1` dot product if missed).
- **Load** uses `validM`/`validN` for an *aligned* operand (`A_PADDED`/`B_PADDED`
  false — the buffer is exactly `L0`/`L1` and the last tile's valid extent is always
  `> Mt−16`, so every fractal block is touched). For a *partial* operand the buffer
  is padded to `Ma` rows / `Na` cols (zero tail) and the load reads a **full** tile
  (`loadM = Mt`, `loadN = Nt`). This is required, not just defensive: the row/col
  analogue of the partial-C0 quirk corrupts a tile whenever a *trailing 16-row
  fractal block is fully empty* (valid extent `≤ Mt−16`), so the operand load must
  fill every block. The store still clamps to `validM × validN`, so the zero pad
  rows/cols produce discarded zero output.

### CPU reference (`cpu_einsum.h`)

A straightforward `transpose → triple-nested-loop matmul → transpose` in plain
C++, used to validate the NPU path. FP32 only. The elementwise path has its own
trivial reference (`cpu::elementwise_mul`).

## 1.3 Elementwise (Hadamard) path (`elementwise_mul_inline`)

The no-contraction case (`ts, ts -> ts`, etc.; see §1.1) bypasses the
transpose/matmul pipeline entirely. The flat element range `[0, N)` is streamed
through UB in fixed-width blocks (`elementwise_mul_block`): load both operands,
multiply, store the `float` result. `float16` inputs are up-cast to `float` with
`TCVT` and multiplied in float (the matmul path likewise accumulates fp16 in
float), so the output is always `float32`. Blocks are distributed across all Vector
cores; each core takes a contiguous span, processing it in `TILE`-wide blocks whose
final block carries a dynamic valid extent for the short tail — so any `N` works
with no padding and no size floor.

The one subtlety is on the fp16 path: the up-cast is two `TCVT`s into `af`/`bf`
followed by an in-place `TMUL` that reads them, which is a read-after-write hazard on
the Vector pipe. A `pipe_barrier(PIPE_V)` between the converts and the multiply is
required; without it the `TMUL` can read pre-convert data. On a large tile the
multi-repeat `TCVT` latency hides the window, which is why this once looked like a
"small/partial-tile fp16 hardware quirk" — it is not. Bare `TCVT` is correct at every
size. fp32 has no `TCVT`, so no intra-`V` hazard.

## 1.4 Broadcast / scaling path (`broadcast_mul`)

Another no-contraction case: one operand (the "full" operand) carries every output
axis in output order, the other a strict subset, broadcast over the axes it lacks —
e.g. `bsd,d->bsd` (rms_norm), `bsd,sd->bsd` (token_pos), `bshd,sd->bshd` (rope),
`hqk,h->hqk` (alibi). There is no contraction and no transpose, so like the Hadamard
path it bypasses the matmul pipeline; `parse_recipe` flags it (`recipe.broadcast`,
NPU only) and `_try_broadcast` classifies the layout. The output is always `float32`.

The broadcast operand cannot be loaded with a stride-0 GM access (the MTE rejects it
as out of range), so it is materialized in the Vector engine. Two kernel modes:

- **Mode 0 — `TCOLEXPANDMUL`** when the innermost output axis is present in the
  broadcast operand. The full operand is streamed as `[rb × Cc]` row blocks and the
  broadcast operand as a `[1 × Cc]` row vector that the hardware column-expand
  multiply broadcasts down the rows (`dst[r,c] = A[r,c] · B[c]`). `Cc` is the inner
  contiguous run of present axes (capped at `BCAST_TILECAP`; a single inner axis
  larger than the cap falls through to the matmul path), `Rr` the adjacent broadcast
  rows. Any further outer axes are walked by a mixed-radix odometer (`bcast_baseB`)
  that projects each output group to its broadcast-operand offset (stride 0 on absent
  axes), which is what makes rope's interior-strided broadcast work.
- **Mode 1 — scalar (`Tile::GetValue` + `TMULS`)** when the innermost axis is absent,
  so the broadcast operand is constant per group (e.g. alibi's per-head slope). The
  (small) operand is loaded once into UB and each group's scalar is read with
  `GetValue` and applied with a tile×scalar multiply.

Rows are flattened over `(group, row)` and split across cores; each lane's span is
rounded up to a `BCAST_LINE_ELEMS`-aligned boundary so adjacent cores never write a
shared cache line (sub-line spans would clobber each other and drop rows). Two
subtleties bit here and are worth recording:

- **Runtime row extent.** A lane's trailing block has `rb < RB` rows. The GM tensor's
  row dimension must therefore be the *runtime* `rb` (a `-1` dynamic dim, as the
  matmul uses for `validM`), not the static `RB` — otherwise `TLOAD`/`TSTORE` iterate
  the full `RB` rows and the last lane reads/writes past the operand buffer (OOB →
  `NaN`). The 1-row scalar path is immune (its `SetValidCol` bounds the load length).
- **Final-store drain.** Each block's `WAR` flag (`MTE3→MTE2`) only orders a store
  ahead of the *next* load, so the last block's store is never awaited. Multi-core
  paths get an implicit drain from the cross-core barrier, but a small broadcast op
  runs on core 0 alone, so its store can still be in flight when the host reads the
  output back — yielding stale data at reused GM addresses. A single
  `pipe_barrier(PIPE_ALL)` at the end of the kernel drains it.

---

# Part 2 — Optimizations

Each of these is layered on the Part 1 skeleton. They are independent; with all of
them disabled the kernel would still be correct, just slower.

## 2.1 Persistent-workspace dispatch (split setup / exec / teardown)

The naive dispatch allocates the intermediate workspace, zeros its pad regions,
launches, syncs, and frees on *every* call. For the sub-0.1 ms kernels this
allocator churn and host-device sync dominate. The fused dispatch is therefore
split into three entry points so a reused runner pays the per-call cost only for
the launch:

- `einsum_setup` allocates the K-padded workspace **once** and zeros its
  contraction-pad regions **once** (only when `K != C`). The per-call transposes
  overwrite only the data columns/rows (cols/rows `C..K` stay 0) and the matmul
  never writes `ws0`/`ws1`, so the zeroing holds across calls.
- `einsum_exec` **only** launches the kernel — no `malloc`/`memset`/`free` and no
  `aclrtSynchronizeStream`. Same-stream ordering serialises reuse and the output's
  downstream torch consumers, exactly as `torch.einsum` is itself async.
- `einsum_teardown` frees the workspace at `cleanup()` (or on builder GC).

This is a large win on the small / copy-bound cases where the allocator and sync
dominated (e.g. the `K != C` batched-unaligned case dropped ~6×). The one-shot
`einsum()` (setup + exec + sync + teardown) is kept for the CPU-symmetric path and
non-reusing callers.

## 2.2 In-kernel contraction padding (no host round-trip)

Part 1 §Phase B requires a `K`-wide, zero-padded A and `K`-row, zero-padded B. The
naive way to get them is a host-side `malloc`/`memcpy2d`/stream-sync between
transpose and matmul (this is what the standalone `batched_matmul` host function
still does — kept as a reference path, exercised by `test_matmul.py`). The fused
kernel instead allocates `ws0`/`ws1` K-padded, zeroes them once (§2.1), and the
**input transposes write the padded layout directly**:

- `ws0` (`A`): the contract dim is innermost, so the input0 transpose pads it from
  `C` to `K` per row via a `DST_PAD` template arg (2D `TTRANS` path) or
  `padded_copy_inline` (identity path). The pad is uniform across all `I·L0` rows,
  so any batch count works.
- `ws1` (`B`): the contract dim is the per-batch *row* count. With a single batch
  (or `K==C`) the one valid `[C, L1]` block sits at the front of the pre-zeroed
  `[K, L1]` buffer, so a plain transpose suffices. With `K!=C` *and* batching, each
  batch's `C` valid rows must land at the front of its `K`-row slot (batch stride
  `K·L1`); a contiguous write would mis-stride the batches, so
  `batched_pad_copy_inline` repacks them with a per-batch `K`-row pad.

This removes the mid-pipeline `malloc`/`memcpy`/stream-sync — a ~21× speedup on
the degenerate outer-product case. The fused path covers every supported config
(`K==C` and `K!=C`, each with any batch count).

## 2.3 Identity-output fast path (skip Phase C)

When the output permutation is identity (`IsIdentityPerm<tpose_out_conf>` —
`ij,jk->ik`, `bij,bjk->bik`, `ijk,jkl->il`, … the common case) the requested
output index order already equals the matmul-natural `[free0, free1]` layout, so
Phase C is a plain copy. In that case:

- the Cube writes its result **straight to `res`** (`mm_out = OUT_IDENTITY ? res : ws_res`),
- the **second `SyncAll<false>()` and Phase C are both dropped** (`if constexpr (!OUT_IDENTITY)`),
- `einsum_setup` **does not allocate `ws_res`** (`wsr_bytes = OUT_IDENTITY ? 0 : …`).

This removes a whole Vec copy pass, one cross-core barrier, and the output-buffer
allocation for the typical matmul. It is correct because the matmul writes every
real `M×N` output element (partial tiles still cover the full real output), so the
`res` buffer is fully written. Only non-identity outputs (e.g. some `ai,ja->ij`
layouts, or a multi-batch-axis result like `bshd,bthd->bsht`) still run Phase C.
Phase C writes `res` in the **final** (post-permutation) order, so the runner
allocates the output torch tensor with the true output shape (`recipe.out_shape`),
not the matmul-natural `out_interpret_shape` — the two differ exactly when the
output permutation is non-identity. Its input-side mirror — skipping the redundant
*input* copies — is §2.8.

## 2.4 L1 Mat-tile double buffering

Inside `matmul_one_tile_inline` the L1 `Mat` tiles are ping-ponged across two
buffers (`matA[2]`/`matB[2]`) so the GM→L1 load (MTE2, the slow stage) of the next
K step overlaps the matmul (M pipe) of the current step. The L0 `Left`/`Right`
tiles stay **single**-buffered: a full `Mt × Kt` tile already fills L0A/L0B for the
default tile, so two won't fit; the mov→matmul chain stays serialized but the
expensive GM loads are hidden behind compute. A `static_assert` checks the
double-buffered L1 tiles fit the 512 KB CBUF.

(A *shallow* L0 double-buffering — shrinking `Kt` so two `Left`/`Right` tiles fit —
was tried and reverted: ~12% slower for fp32-128, only ~5% faster for fp16, and it
needed an extra `PIPE_ALL`. A *deeper* L0 ping-pong that keeps `Kt` and extracts L0
sub-tiles from a larger L1 chunk does pay off and is the plain schedule's per-tile
path — see §2.7. `matmul_one_tile_inline` itself now serves only the split-K path.)

## 2.5 Adaptive `tile_k` (thin-problem K depth)

`builder._tile_k` grows the K-tile depth (`tile_m`/`tile_n` stay at the base 128)
as large as the on-chip buffers allow for *thin* problems (small padded M/N):
`Mt·Kt·ds ≤ 64 KB` (L0A), `Nt·Kt·ds ≤ 64 KB` (L0B), `2(Mt+Nt)·Kt·ds ≤ 512 KB`
(CBUF), picking the largest divisor of padded `K` that fits. For square/large M,N
those caps pin it back to 128, so the tuned matmul path is byte-for-byte
unchanged; for a long-`K` reduction it cuts the serial K-step count. On its own it
gives ~1.6× on the dot product, then plateaus — one core's load bandwidth is the
floor, which §2.6 addresses.

## 2.6 Split-K (use idle cores for long-K / thin grids)

The output-tile grid is `total_tiles = I·⌈L0/Mt⌉·⌈L1/Nt⌉`; the plain schedule
gives one tile per core, so when `total_tiles < block_num` the surplus cores idle.
The dot product `i,i->` is the extreme — a single 1×1 tile over a long `K` runs on
*one* of ~20 cores.

Split-K (`batched_matmul_inline<…, SPLITK=true>`) instead assigns
`ksplit = block_num/total_tiles` cores to each tile, each contracting a disjoint
K-slice (`matmul_one_tile_inline` over `[s·kpc, s·kpc + kpc)`) and **atomic-adding**
(`TSTORE<…, AtomicType::AtomicAdd>`) its partial into the output. `ksplit` is
chosen at runtime, clamped to a divisor of `nK`; `ksplit == 1` falls back to the
plain schedule.

- **Compile-time gate** (`SPLITK_ELIGIBLE` in `einsum_fused_kernel`): identity
  output **and** `nK ≥ 2` **and** `total_tiles < 16`. `SplitK<CONFIG_T>` carries
  the shared geometry so the host eligibility check (`builder._splitk_eligible`)
  and the kernel agree.
- **The output must start zeroed**, and zeroing it **in-kernel on the Vec side is
  not coherent** with the Cube's fixpipe atomic RMW across `SyncAll`: the MTE3→FIX
  path differs from the MTE3→MTE2 path the input transposes rely on, so the Cube
  atomic-adds onto a stale value — silent, deterministic, input-dependent
  corruption (e.g. a randn dot off by ~1–2% while ramp/ones look fine, because the
  error only surfaces under cancellation). **Fix:** the *host* pre-zeros the output
  (`builder._splitk_eligible` → `torch.zeros` instead of `torch.empty`); the
  `torch.zeros` launch is a kernel boundary = full GM sync, so the atomic-adds see
  clean zeros. The builder eligibility must stay in lockstep with the kernel's
  `SPLITK_ELIGIBLE` constant.

Net: the dot product drops ~1.9 ms → ~0.19 ms (~10×, from 4× slower than torch to
~2.4× faster). All other benchmark cases are unaffected (gated off).

## 2.7 Deep-pipelined Cube tile (`matmul_one_tile_deep`)

§2.4's per-`Kt` loop single-buffers L0, so the Cube stalls on the L1→L0 mov. The
plain (non-split-K) schedule instead runs each tile through `matmul_one_tile_deep`,
which adds two overlaps:

- **L1 chunk reuse**: A `[Mt, Kd]` and B `[Kd, Nt]` are loaded once per `Kd`-deep
  chunk and `TEXTRACT`-ed into `Kd/Kq` L0 sub-tiles, amortizing the big GM→L1
  transfers. `Kq` (L0 compute granularity) and `Kd` (L1 chunk depth) are constexpr
  from `Mt/Nt/K/dtype` (`deep_pick_kq`/`deep_pick_kd`) and both divide `K`, so the
  chunk and phase loops are clean — awkward or small `K` collapses to `Kd = Kq = K`.
- **L0 ping-pong**: the `Left`/`Right` tiles double-buffer so the `TEXTRACT` of phase
  `p+1` (MTE1) overlaps the `TMATMUL` of phase `p` (M); the L1 chunks also
  double-buffer to prefetch the next chunk's GM→L1 load under the current compute.

Each tile is self-contained (inits its event flags, drains them before the store).
This brings the matmul itself to ~parity with `torch` (fp32 2048² 403→313 µs; fp16
155 vs 140 µs), and is correct across fp16/fp32, batched, `K!=C`, and partial tiles.

(An L2 tile-swizzle was also ported here from the pto-kernels matmul-swizzle example
and then **removed**. Measured cleanly it was a no-op on square grids — they are
HBM-bound on redundant operand streaming, which a schedule reorder cannot fix — and a
regression on non-square grids. The schedule is a plain contiguous-chunk distribution.)

## 2.8 Identity-input fast path (skip Phase A)

The input-side mirror of §2.3. When an input's transpose permutation is the identity
**and** the contraction needs no fractal padding (`K==C`), its canonical workspace
layout is byte-identical to the raw tensor, so copying it into `ws0`/`ws1` is pure
overhead — and on plain matmuls that copy *dominates* the fused kernel (the GM→UB→GM
copy of both operands cost ~14× the matmul on a 2048² case; it was the real gap, not
the matmul). `EinsumSkip<CONFIG_T>` (static `constexpr inp0`/`inp1`) gates it:

- the kernel **skips that input's Phase A transpose** (`if constexpr (!SKIP0) …`) and
  the matmul reads `data0`/`data1` **directly** (`mmA = SKIP0 ? data0 : ws0`);
- `einsum_setup`/`einsum_exec` drop the skipped buffer from the workspace layout (a
  token byte is allocated if nothing remains — e.g. the dot product — so the runner
  still caches a non-null handle instead of re-running setup every call).

It must be a struct with `static constexpr` members, not a `constexpr` function — a
plain `constexpr` function defaults to host-only and the `__global__` kernel cannot
call it. `K!=C` forces both flags false (those inputs genuinely need §2.2's pad
repack). Net: the full einsum collapses onto the matmul — 2048² fp16 2339→178 µs
(13×), fp32 3913→338 µs (~parity with torch).

---

## Build cache caveat

The build cache key is an MD5 of the **generated `.cpp`** (the config structs and
`extern "C"` wrappers), which does *not* include the kernel headers. After editing
`include/*.h`, delete the build dir (`rm -rf build`) before re-running, otherwise a
stale `.so` is reused and the change appears to have no effect.

## Roadmap

- Lift the N-D batched-`TTRANS` `srcColStride==1` requirement (source whose
  second-innermost output axis is non-contiguous) — needs a non-contiguous gather,
  not a single strided DMA.
- Batched partial matmul tiles. Partial M/N tiles (a free dim not a multiple of the
  tile) are supported for single-batch (`I==1`) only; a batched partial config needs
  the per-batch row/col block padded and is currently rejected with a `ValueError`.
- Broadcast/scaling ops with an inner present-axis run larger than `BCAST_TILECAP`
  (e.g. token_pos at `d=512` → `Cc=8192`) fall through to the matmul path — in-kernel
  column blocking of the `[rb × Cc]` tile (§1.4) would lift it.
- Single-operand reduction axes — an index in exactly one operand and not in the
  output (e.g. `ij,jk->k`, where torch sums `i`). The transpose→matmul pipeline has
  no reduce stage, so `parse_recipe` rejects these with a clean `ValueError` today
  (it previously dropped the axis and miscomputed silently). **Option A (in-pipeline
  Vector reduction):** detect the solo axes in `parse_recipe`, reorder them outermost
  on the owning operand, and add a Vector reduce-sum stage that collapses them into a
  workspace before Phase A — mirroring the elementwise/broadcast kernels (new codegen,
  an FFTS barrier, workspace sizing, dispatch plumbing). The reduced operand then
  feeds the existing transpose→matmul→transpose pipeline unchanged. Keeps the op fused
  and on-device.
