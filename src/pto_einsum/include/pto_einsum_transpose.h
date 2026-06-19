#pragma once
#include "pto_einsum_utils.h"

namespace pto_einsum {

// 1. Transpose input/output (Vector core). TTRANS hardware tile transpose
// (vnchwconv) for 2D non-identity permutations; TLOAD/TSTORE tiled copy for
// identity permutations.

// Width budget (elems/row) for one GM->GM copy chunk through UB. A hardcoded 128
// was far too small: at fp16 that is a 256-byte DMA per chunk, so a wide contiguous
// run is split into many minuscule transfers, each carrying a full RAW/WAR pipe sync
// — starving the MTE engine (measured: the strided-gather Phase A is gather-DMA-bound,
// and a single wide chunk is ~2.9x faster on wide inner runs). Size the copy tile to
// the run width, capped here to a fixed UB byte budget (~16 KB: 8192 fp16 / 4096 fp32
// elems) so a single UB tile stays modest. Narrow runs (e.g. head_dim) size below the
// cap and see no change; that is correct — they were never DMA-starved.
template <typename data_T>
AICORE inline constexpr unsigned copy_tile_cap() { return 16384u / sizeof(data_T); }

// Copy one chunk (a single row, <= TILE_COLS wide with valid extent `w`) from GM
// `src` to GM `dst` through a reused UB tile, with the RAW (load->store) and WAR
// (store->next-load) pipe sync. This is the shared inner body of the three GM->GM
// copy paths below (identity copy, padded copy, batched padded copy); only their
// address arithmetic and chunking differ, so they each set up the UB tile once and
// drive this per chunk.
template <typename data_T, unsigned TILE_COLS, typename TileT>
AICORE inline void copy_chunk_through_ub(TileT& ubTile, __gm__ const data_T* src, __gm__ data_T* dst, unsigned w) {
    using GlobalData = GlobalTensor<data_T, pto::Shape<1, 1, 1, 1, TILE_COLS>, pto::Stride<1, 1, 1, TILE_COLS, 1>>;
    ubTile.SetValidRow(1);
    ubTile.SetValidCol(w);
    GlobalData sg(const_cast<__gm__ data_T*>(src));
    GlobalData dg(dst);

    TLOAD(ubTile, sg);

    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);   // RAW: store waits its load
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

    TSTORE(dg, ubTile);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);   // WAR: next load reuses ubTile
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

// Identity copy: TLOAD src → TSTORE dst as flat 1D chunks through UB (src and
// dst share the same layout). Caller distributes the [start, end) range.
template <typename data_T, typename CONFIG_T>
AICORE inline void transpose_copy_inline(__gm__ const data_T* src, __gm__ data_T* dst, unsigned start, unsigned end) {
    constexpr unsigned TILE_COLS = copy_tile_cap<data_T>();  // wide chunk; SetValidCol clamps the tail
    using TileData = Tile<TileType::Vec, data_T, 1, TILE_COLS, BLayout::RowMajor, -1, -1>;

    TileData ubTile(1, TILE_COLS);
    TASSIGN(ubTile, 0x0);

    for (unsigned i = start; i < end; i += TILE_COLS) {
        unsigned chunk = ((i + TILE_COLS) <= end) ? TILE_COLS : (end - i);
        copy_chunk_through_ub<data_T, TILE_COLS>(ubTile, src + i, dst + i, chunk);
    }
}

// Row-padded copy for an identity-permutation input that feeds the Cube: the
// contiguous [nrows, rowW] source is copied into a [nrows, dstStride] destination
// (dstStride >= rowW; the pad columns rowW..dstStride are pre-zeroed by the
// caller). Used by the fused kernel for the input0 transpose when the contraction
// C is padded to K. Rows are distributed across the vector lanes.
template <typename data_T>
AICORE inline void padded_copy_inline(__gm__ const data_T* src, __gm__ data_T* dst,
                                      unsigned nrows, unsigned rowW, unsigned dstStride,
                                      unsigned lane, unsigned nlanes) {
    constexpr unsigned TILE_COLS = copy_tile_cap<data_T>();
    using TileData = Tile<TileType::Vec, data_T, 1, TILE_COLS, BLayout::RowMajor, -1, -1>;

    TileData ubTile(1, TILE_COLS);
    TASSIGN(ubTile, 0x0);

    unsigned chunk = (nrows + nlanes - 1) / nlanes;
    unsigned r0 = lane * chunk;
    unsigned r1 = (r0 + chunk) < nrows ? (r0 + chunk) : nrows;

    for (unsigned r = r0; r < r1; r++) {
        for (unsigned c = 0; c < rowW; c += TILE_COLS) {
            unsigned w = (c + TILE_COLS) <= rowW ? TILE_COLS : (rowW - c);
            copy_chunk_through_ub<data_T, TILE_COLS>(ubTile, src + r * rowW + c, dst + r * dstStride + c, w);
        }
    }
}

// Batched row-padded copy for an identity input1 feeding the Cube when the
// contraction C is padded to K. The canonical input1 is a contiguous [I, C, L1]
// stack (batch stride C*L1), but the Cube reads each batch as a [K, L1] block
// (batch stride dstBatchStride = K*L1) with the K-C tail rows zero (pre-zeroed by
// the caller). Each batch's C valid rows are copied to the front of its K-row
// slot; a plain contiguous copy would mis-stride the batches. The I*C source rows
// are distributed across the vector lanes.
template <typename data_T>
AICORE inline void batched_pad_copy_inline(__gm__ const data_T* src, __gm__ data_T* dst,
                                           unsigned nbatch, unsigned rows, unsigned rowW,
                                           unsigned dstBatchStride, unsigned lane, unsigned nlanes) {
    constexpr unsigned TILE_COLS = copy_tile_cap<data_T>();
    using TileData = Tile<TileType::Vec, data_T, 1, TILE_COLS, BLayout::RowMajor, -1, -1>;

    TileData ubTile(1, TILE_COLS);
    TASSIGN(ubTile, 0x0);

    unsigned totalRows = nbatch * rows;
    unsigned chunk = (totalRows + nlanes - 1) / nlanes;
    unsigned g0 = lane * chunk;
    unsigned g1 = (g0 + chunk) < totalRows ? (g0 + chunk) : totalRows;

    for (unsigned g = g0; g < g1; g++) {
        unsigned i = g / rows;       // batch
        unsigned c = g - i * rows;   // row within the batch's C valid rows
        for (unsigned col = 0; col < rowW; col += TILE_COLS) {
            unsigned w = (col + TILE_COLS) <= rowW ? TILE_COLS : (rowW - col);
            copy_chunk_through_ub<data_T, TILE_COLS>(
                ubTile, src + g * rowW + col, dst + i * dstBatchStride + c * rowW + col, w);
        }
    }
}

// 2D transpose using the TTRANS hardware tile op (vnchwconv). TTRANS transposes
// a RowMajor src UB tile into a dst UB tile via a scratch tmp tile; it is general
// over the valid extents (any validRow x validCol) for our b32/b16 dtypes. The
// only structural requirement is that the three tiles fit in UB, so large tensors
// are blocked.
//
// TTRANS takes its fast HW path only when srcStride % (32/sizeof) == 0 and
// dstStride % 16 == 0 (else it falls back internally to a scalar transpose). The
// strides are the *UB tile row-strides*, which we control, so we pad the tile
// widths to always hit the HW path; the logical dims can be arbitrary.
template <typename data_T, typename CONFIG_T, unsigned DST_PAD = 0>
AICORE inline void transpose_2d_inline(__gm__ const data_T* src, __gm__ data_T* dst,
                                       unsigned block_idx, unsigned block_num) {
    constexpr unsigned srcRows = CONFIG_T::from_shape[0];
    constexpr unsigned srcCols = CONFIG_T::from_shape[1];

    // Destination innermost row stride. The transposed output is [srcCols, srcRows]
    // with natural row stride srcRows; for an input0 transpose feeding the Cube the
    // innermost (contract) dim is padded to DST_PAD >= srcRows (the pad columns are
    // pre-zeroed by the caller). DST_PAD == 0 means "no padding" (standalone path).
    constexpr unsigned dpad = (DST_PAD > srcRows) ? DST_PAD : srcRows;

    // X sub-tile width for vnchwconv: 8 elems (fp32) / 16 elems (fp16).
    constexpr unsigned blk = 32u / sizeof(data_T);

    // Padded tile widths (= RowStride). srcTW is a multiple of blk so the src
    // RowStride satisfies the HW path; dstTW/tmpTW are multiples of 16 (which is
    // also 32-byte aligned for both fp32 and fp16) so the dst/tmp strides do too.
    constexpr unsigned srcTW = align_up(srcCols, blk);
    constexpr unsigned dstTW = align_up(srcRows, 16u);
    constexpr unsigned tmpTW = dstTW;

    constexpr unsigned srcBytes = srcRows * srcTW * sizeof(data_T);  // src tile (srcRows x srcTW)
    constexpr unsigned dstBytes = srcCols * dstTW * sizeof(data_T);  // dst tile (srcCols x dstTW)
    constexpr unsigned tmpBytes = srcCols * tmpTW * sizeof(data_T);  // tmp tile (srcCols x tmpTW)

    // Leave headroom under the 192 KB UB; whole-tensor TTRANS needs all 3 tiles.
    constexpr unsigned UB_BUDGET = 184u * 1024u;

    if constexpr (srcBytes + dstBytes + tmpBytes <= UB_BUDGET) {
        // Whole tensor fits in UB: one TTRANS over the full srcRows x srcCols.
        // Both GM transfers are a single (possibly strided) 2D move, so arbitrary
        // (even unaligned) dims work here. One tile == one unit of work, so only
        // one core runs it; these tensors are small by definition.
        if (block_idx != 0) {
            return;
        }
        using SrcShape = pto::Shape<1, 1, 1, srcRows, srcCols>;
        using SrcStride = pto::Stride<1, 1, 1, srcCols, 1>;
        using DstShape = pto::Shape<1, 1, 1, srcCols, srcRows>;
        using DstStride = pto::Stride<1, 1, 1, dpad, 1>;
        using SrcGlobal = GlobalTensor<data_T, SrcShape, SrcStride>;
        using DstGlobal = GlobalTensor<data_T, DstShape, DstStride>;

        using SrcTile = Tile<TileType::Vec, data_T, srcRows, srcTW, BLayout::RowMajor, -1, -1>;
        using DstTile = Tile<TileType::Vec, data_T, srcCols, dstTW, BLayout::RowMajor, -1, -1>;
        using TmpTile = Tile<TileType::Vec, data_T, srcCols, tmpTW, BLayout::RowMajor, -1, -1>;

        SrcTile srcTile(srcRows, srcCols);
        DstTile dstTile(srcCols, srcRows);
        TmpTile tmpTile(srcCols, tmpTW);
        TASSIGN(srcTile, 0);
        TASSIGN(dstTile, srcBytes);
        TASSIGN(tmpTile, srcBytes + dstBytes);

        SrcGlobal srcGlobal(const_cast<__gm__ data_T*>(src));
        DstGlobal dstGlobal(dst);

        TLOAD(srcTile, srcGlobal);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);   // RAW: transpose waits its load
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TTRANS(dstTile, srcTile, tmpTile);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);   // RAW: store waits the transpose
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);  // one-shot drain (single tile, no next iteration)
    } else {
        // Too large for UB: TTRANS one BR x BC source block at a time, each a single
        // strided 2D load -> HW transpose -> strided 2D store. The block grid is
        // ceil'd, so boundary blocks are partial -- the DYNAMIC GM windows below carry
        // the real per-block rb/cb (not the padded BR/BC), transferring exactly the
        // valid region. Every block base address is 32-byte aligned (r0/c0 are
        // multiples of 64), and the GM↔UB transfers are element-granular ND copies, so
        // a non-16-aligned row stride (srcCols / dpad) only makes successive rows start
        // at sub-32B offsets, which the ND DMA handles -- arbitrary (unaligned) dims
        // work here, same as the fits-UB and N-D batched paths.
        constexpr unsigned BR = 64;
        constexpr unsigned BC = 64;
        constexpr unsigned subSrcTW = align_up(BC, blk);   // src sub-tile width  (RowStride)
        constexpr unsigned subDstTW = align_up(BR, 16u);   // dst/tmp sub-tile width (RowStride)

        // DYNAMIC GM windows carry the real per-block dims (rb/cb), not the padded
        // BR/BC -- a static window over-reads the source and over-writes neighbouring
        // output on partial (non-BR/BC-multiple) boundary blocks. Set per block below.
        using SrcShape = pto::Shape<1, 1, 1, -1, -1>;
        using SrcStride = pto::Stride<1, 1, 1, srcCols, 1>;
        using DstShape = pto::Shape<1, 1, 1, -1, -1>;
        using DstStride = pto::Stride<1, 1, 1, dpad, 1>;
        using SrcGlobal = GlobalTensor<data_T, SrcShape, SrcStride>;
        using DstGlobal = GlobalTensor<data_T, DstShape, DstStride>;

        using SrcTile = Tile<TileType::Vec, data_T, BR, subSrcTW, BLayout::RowMajor, -1, -1>;
        using DstTile = Tile<TileType::Vec, data_T, BC, subDstTW, BLayout::RowMajor, -1, -1>;
        using TmpTile = Tile<TileType::Vec, data_T, BC, subDstTW, BLayout::RowMajor, -1, -1>;

        constexpr unsigned subSrcBytes = BR * subSrcTW * sizeof(data_T);
        constexpr unsigned subDstBytes = BC * subDstTW * sizeof(data_T);
        SrcTile srcTile(BR, BC);
        DstTile dstTile(BC, BR);
        TmpTile tmpTile(BC, subDstTW);
        TASSIGN(srcTile, 0);
        TASSIGN(dstTile, subSrcBytes);
        TASSIGN(tmpTile, subSrcBytes + subDstBytes);

        // The BR x BC source blocks are independent (each is a self-contained
        // load -> TTRANS -> store), so the block grid is flattened into a 1D index
        // space and block-distributed across the Vec cores, mirroring how the Cube
        // matmul distributes its output-tile grid. Each core owns its own UB tiles.
        constexpr unsigned nBR = (srcRows + BR - 1) / BR;
        constexpr unsigned nBC = (srcCols + BC - 1) / BC;
        constexpr unsigned nBlocks = nBR * nBC;
        unsigned chunk = (nBlocks + block_num - 1) / block_num;
        unsigned bstart = block_idx * chunk;
        unsigned bend = (bstart + chunk) < nBlocks ? (bstart + chunk) : nBlocks;

        for (unsigned b = bstart; b < bend; b++) {
            unsigned r0 = (b / nBC) * BR;
            unsigned c0 = (b % nBC) * BC;
            unsigned rb = (srcRows - r0) < BR ? (srcRows - r0) : BR;
            unsigned cb = (srcCols - c0) < BC ? (srcCols - c0) : BC;

            // Load the rb x cb source block, transpose to cb x rb, store it.
            srcTile.SetValidRow(rb);
            srcTile.SetValidCol(cb);
            dstTile.SetValidRow(cb);
            dstTile.SetValidCol(rb);
            SrcGlobal sg(const_cast<__gm__ data_T*>(src + r0 * srcCols + c0), SrcShape(rb, cb));
            DstGlobal dg(dst + c0 * dpad + r0, DstShape(cb, rb));

            TLOAD(srcTile, sg);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);   // RAW: transpose waits its load
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TTRANS(dstTile, srcTile, tmpTile);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);   // RAW: store waits the transpose
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            TSTORE(dg, dstTile);

            // WAR for the next block's reuse of the single src/dst/tmp UB tiles: the
            // next load must not clobber srcTile before this transpose read it, and
            // the next transpose must not clobber dst/tmp before this store read them.
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        }
    }
}

// ─── N-D non-identity transpose ──────────────────────────────────────────────
// The 2D TTRANS path above covers rank-2 operands; einsums with two or more
// batch axes (attention: bshd,bthd->bsht, …) need to permute a >2D operand into
// the canonical [batch, free, contract] matmul layout. Such a permutation always
// splits into one of two shapes, distinguished by the innermost output axis:
//
//   (a) innermost axis preserved (perm[dims-1] == dims-1): the contiguous inner
//       run is gathered to permuted outer slots — a strided copy, no transpose.
//   (b) innermost axis changes: the last two output axes form a 2D transpose for
//       every fixed setting of the leading (batch) axes — a batched TTRANS.
//
// Both read the (already emitted) `to_shape` (output extents) and `perm_strides`
// (source stride of each output axis): for any permutation the contiguous output
// block over the trailing axes maps to a strided source block, so dst stays the
// natural matmul layout and only the source side carries the permutation.
//
// The Cube matmul flattens all batch axes into one `n_inplace` already, so the
// contraction needs no change — only this marshalling did.

// Case (a): innermost-preserved strided gather. Each of the N/inner contiguous
// inner runs is copied (through UB) to its natural slot in the dst; its source
// base offset is the mixed-radix sum of the outer indices times perm_strides.
// PAD_INNER/PAD_STRIDE pad the destination inner stride (contract C -> K) exactly
// like the 2D path -- used for an *input0* transpose (contract innermost); equal/
// zero ⇒ no padding.
// ROWS/PAD_BATCH express the orthogonal *input1* row pad: the last outer axis is
// the contract row count (= ROWS = C), and each batch's ROWS valid rows must land
// at the front of its PAD_BATCH (= K*L1)-wide slot. With ROWS>0 the dst run offset
// becomes (g/ROWS)*PAD_BATCH + (g%ROWS)*dstStride; both 0 ⇒ contiguous (the default).
// Runs are distributed across lanes.
template <typename data_T, typename CONFIG_T, unsigned PAD_INNER = 0, unsigned PAD_STRIDE = 0,
          unsigned ROWS = 0, unsigned PAD_BATCH = 0>
AICORE inline void transpose_nd_copy_inline(__gm__ const data_T* src, __gm__ data_T* dst,
                                            unsigned lane, unsigned nlanes) {
    constexpr unsigned dims  = CONFIG_T::dims;
    constexpr unsigned inner = CONFIG_T::to_shape[dims - 1];
    constexpr unsigned nouter = CONFIG_T::N / inner;
    constexpr unsigned dstStride = (PAD_STRIDE > inner) ? PAD_STRIDE : inner;
    // Size the copy tile to the contiguous inner run (capped): an inner run <= cap
    // moves as a single wide DMA instead of inner/128 tiny gated chunks. The tile
    // width must be 32-byte aligned (a RowMajor Vec tile requires it), so round the
    // run width up to the DMA granule; SetValidCol clamps the actual copied extent.
    constexpr unsigned TILE_COLS = (inner < copy_tile_cap<data_T>())
                                       ? align_up(inner, 32u / sizeof(data_T))
                                       : copy_tile_cap<data_T>();
    using TileData = Tile<TileType::Vec, data_T, 1, TILE_COLS, BLayout::RowMajor, -1, -1>;
    using GlobalData = GlobalTensor<data_T, pto::Shape<1, 1, 1, 1, TILE_COLS>,
                                    pto::Stride<1, 1, 1, TILE_COLS, 1>>;

    // Double-buffered GM->GM copy: ping-pong two UB tiles so the MTE2 load of chunk
    // k+1 overlaps the MTE3 store of chunk k. The single-buffer copy_chunk_through_ub
    // forces next-load-waits-this-store (WAR) every chunk, fully serialising the two
    // memory engines -- which is the bulk of Phase A for the strided gather (each run
    // is a separate tiny gated DMA). Here the WAR wait is deferred two chunks (the
    // reload of a buffer waits only ITS own prior store), so loads run ahead of stores.
    // Mirrors the deep-matmul L1 pipeline idiom (parity p, e_raw/e_war, drain at end).
    constexpr unsigned tileBytes = TILE_COLS * sizeof(data_T);
    TileData ub0(1, TILE_COLS), ub1(1, TILE_COLS);
    TASSIGN(ub0, 0x0);          // two distinct UB buffers (offset = one tile apart) so
    TASSIGN(ub1, tileBytes);    // the ping-pong load/store do not alias
    TileData* ub[2] = {&ub0, &ub1};

    unsigned chunk = (nouter + nlanes - 1) / nlanes;
    unsigned g0 = lane * chunk;
    unsigned g1 = (g0 + chunk) < nouter ? (g0 + chunk) : nouter;

    // Odometer over the outer axes [0, dims-1): decode this lane's first flat index
    // once, then carry per step instead of a per-iteration mixed-radix div/mod.
    PermOdometer<CONFIG_T, dims - 1> odo;
    odo.init(g0);
    unsigned k = 0;   // flat chunk counter (drives the double-buffer parity)
    for (unsigned g = g0; g < g1; g++, odo.advance()) {
        __gm__ const data_T* s = src + odo.base;
        size_t doff = (ROWS > 0) ? (size_t)(g / ROWS) * PAD_BATCH + (size_t)(g % ROWS) * dstStride
                                 : (size_t)g * dstStride;
        __gm__ data_T* d = dst + doff;
        for (unsigned c = 0; c < inner; c += TILE_COLS, k++) {
            unsigned w = (c + TILE_COLS) <= inner ? TILE_COLS : (inner - c);
            unsigned p = k & 1u;
            event_t e_raw = p ? EVENT_ID1 : EVENT_ID0;   // RAW: store waits its load
            event_t e_war = p ? EVENT_ID3 : EVENT_ID2;   // WAR: reload waits prior store
            TileData& t = *ub[p];
            t.SetValidRow(1);
            t.SetValidCol(w);
            GlobalData sg(const_cast<__gm__ data_T*>(s + c));
            GlobalData dg(d + c);

            if (k >= 2) wait_flag(PIPE_MTE3, PIPE_MTE2, e_war);   // this buffer free
            TLOAD(t, sg);
            set_flag(PIPE_MTE2, PIPE_MTE3, e_raw);
            wait_flag(PIPE_MTE2, PIPE_MTE3, e_raw);
            TSTORE(dg, t);
            set_flag(PIPE_MTE3, PIPE_MTE2, e_war);               // buffer free in 2 chunks
        }
    }
    // Drain the trailing WAR flags the last one/two chunks set but nothing consumed,
    // so no flag state leaks into subsequent kernel code.
    if (k >= 1) wait_flag(PIPE_MTE3, PIPE_MTE2, ((k - 1) & 1u) ? EVENT_ID3 : EVENT_ID2);
    if (k >= 2) wait_flag(PIPE_MTE3, PIPE_MTE2, ((k - 2) & 1u) ? EVENT_ID3 : EVENT_ID2);
}

// One fixed-size TTRANS of an [SR, SC] source block (row stride SRC_RS, unit col
// stride) into its [SC, SR] transpose written contiguously (row stride DST_RS).
// A parameterised extraction of transpose_2d_inline's "fits-UB" body: callers own
// the lane assignment (no block gating), so it runs the whole block on this lane.
template <typename data_T, unsigned SR, unsigned SC, unsigned SRC_RS, unsigned DST_RS>
AICORE inline void ttrans_block_inline(__gm__ const data_T* src, __gm__ data_T* dst) {
    constexpr unsigned blk = 32u / sizeof(data_T);
    constexpr unsigned srcTW = align_up(SC, blk);
    constexpr unsigned dstTW = align_up(SR, 16u);
    constexpr unsigned tmpTW = dstTW;
    constexpr unsigned srcBytes = SR * srcTW * sizeof(data_T);
    constexpr unsigned dstBytes = SC * dstTW * sizeof(data_T);
    constexpr unsigned tmpBytes = SC * tmpTW * sizeof(data_T);
    static_assert(srcBytes + dstBytes + tmpBytes <= 184u * 1024u,
                  "ND inner transpose block must fit in UB (this is the small per-batch "
                  "tile path; larger tiles take the blocked sub-tile path instead)");

    using SrcShape = pto::Shape<1, 1, 1, SR, SC>;
    using SrcStride = pto::Stride<1, 1, 1, SRC_RS, 1>;
    using DstShape = pto::Shape<1, 1, 1, SC, SR>;
    using DstStride = pto::Stride<1, 1, 1, DST_RS, 1>;
    using SrcGlobal = GlobalTensor<data_T, SrcShape, SrcStride>;
    using DstGlobal = GlobalTensor<data_T, DstShape, DstStride>;

    using SrcTile = Tile<TileType::Vec, data_T, SR, srcTW, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, data_T, SC, dstTW, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, data_T, SC, tmpTW, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(SR, SC);
    DstTile dstTile(SC, SR);
    TmpTile tmpTile(SC, tmpTW);
    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, srcBytes);
    TASSIGN(tmpTile, srcBytes + dstBytes);

    SrcGlobal srcGlobal(const_cast<__gm__ data_T*>(src));
    DstGlobal dstGlobal(dst);

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);   // RAW: transpose waits its load
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TTRANS(dstTile, srcTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);   // RAW: store waits the transpose
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
    pipe_barrier(PIPE_ALL);  // drain before this lane reuses the UB for its next batch
}

// Dynamic-extent single sub-block TTRANS for the *blocked* N-D path: transpose one
// [rb, cb] source sub-block (max BR x BC, row stride SRC_RS, unit col stride) into
// its [cb, rb] dst (row stride DST_RS). The valid extents rb/cb are runtime (the
// boundary sub-blocks of a per-batch [Bb, A] tile are partial). Self-contained
// (own UB tiles, internal PIPE_ALL drain), so the caller just loops work items --
// the blocked-path analogue of ttrans_block_inline.
//
// The GM windows carry the *real* dims (DYNAMIC Shape set to rb/cb), not the padded
// BR/BC tile size, so a partial sub-block transfers exactly its valid region -- a
// static BR x BC window would over-read the source and over-write neighbouring
// output rows (mirrors the matmul's "GM windows carry real dims", Phase B).
template <typename data_T, unsigned BR, unsigned BC, unsigned SRC_RS, unsigned DST_RS>
AICORE inline void ttrans_block_dyn_inline(__gm__ const data_T* src, __gm__ data_T* dst,
                                           unsigned rb, unsigned cb) {
    constexpr unsigned blk = 32u / sizeof(data_T);
    constexpr unsigned subSrcTW = align_up(BC, blk);   // src sub-tile width (RowStride)
    constexpr unsigned subDstTW = align_up(BR, 16u);   // dst/tmp sub-tile width (RowStride)
    constexpr unsigned subSrcBytes = BR * subSrcTW * sizeof(data_T);
    constexpr unsigned subDstBytes = BC * subDstTW * sizeof(data_T);
    static_assert(subSrcBytes + 2u * subDstBytes <= 184u * 1024u,
                  "ND blocked transpose sub-tile must fit in UB");

    using SrcShape = pto::Shape<1, 1, 1, -1, -1>;
    using SrcStride = pto::Stride<1, 1, 1, SRC_RS, 1>;
    using DstShape = pto::Shape<1, 1, 1, -1, -1>;
    using DstStride = pto::Stride<1, 1, 1, DST_RS, 1>;
    using SrcGlobal = GlobalTensor<data_T, SrcShape, SrcStride>;
    using DstGlobal = GlobalTensor<data_T, DstShape, DstStride>;

    using SrcTile = Tile<TileType::Vec, data_T, BR, subSrcTW, BLayout::RowMajor, -1, -1>;
    using DstTile = Tile<TileType::Vec, data_T, BC, subDstTW, BLayout::RowMajor, -1, -1>;
    using TmpTile = Tile<TileType::Vec, data_T, BC, subDstTW, BLayout::RowMajor, -1, -1>;

    SrcTile srcTile(BR, BC);
    DstTile dstTile(BC, BR);
    TmpTile tmpTile(BC, subDstTW);
    TASSIGN(srcTile, 0);
    TASSIGN(dstTile, subSrcBytes);
    TASSIGN(tmpTile, subSrcBytes + subDstBytes);

    srcTile.SetValidRow(rb);
    srcTile.SetValidCol(cb);
    dstTile.SetValidRow(cb);
    dstTile.SetValidCol(rb);

    SrcGlobal srcGlobal(const_cast<__gm__ data_T*>(src), SrcShape(rb, cb));
    DstGlobal dstGlobal(dst, DstShape(cb, rb));

    TLOAD(srcTile, srcGlobal);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);   // RAW: transpose waits its load
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TTRANS(dstTile, srcTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);   // RAW: store waits the transpose
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
    pipe_barrier(PIPE_ALL);  // drain before this lane reuses the UB for its next sub-block
}

// Case (b): batched 2D transpose. The trailing two output axes (A = to_shape[-2],
// Bb = to_shape[-1]) form the transpose; all leading axes are batch. For each
// batch the source block is [Bb, A] with row stride perm_strides[-1] and unit col
// stride (the second-innermost axis is source-contiguous), transposed into the
// [A, Bb] dst slot. Batches are distributed across lanes.
//
// Contract padding (K!=C) on a transposed input is expressed by two orthogonal
// destination knobs, depending on which output axis is the contract dim:
//   - input0 (contract = innermost Bb): PAD_INNER=C / PAD_STRIDE=K widen each
//     transposed row's stride Bb -> K (DST_RS), so the per-batch block is A*K.
//   - input1 (contract = second-innermost A=rows): DST_BLOCK overrides the
//     per-batch dst block from A*Bb -> K*Bb, landing each batch's A valid rows at
//     the front of its K-row slot (innermost Bb=L1 is not padded, DST_RS stays Bb).
// The pad regions are pre-zeroed by the caller; TSTORE writes only the [A,Bb]
// valid window, so they hold across calls (§2.1 invariant). DST_BLOCK==0 ⇒ default.
//
// Each per-batch [Bb, A] block is transposed whole when it fits UB (one TTRANS per
// batch, batches distributed across lanes). When it does not, the block is tiled
// into BR x BC sub-blocks and the (batch, sub-block) grid is flattened into a 1D
// work space distributed across lanes -- so a single large per-batch transpose uses
// the whole vector engine instead of one lane (mirrors the 2D blocked path).
template <typename data_T, typename CONFIG_T, unsigned PAD_INNER = 0,
          unsigned PAD_STRIDE = 0, unsigned DST_BLOCK = 0>
AICORE inline void transpose_nd_batched_2d_inline(__gm__ const data_T* src, __gm__ data_T* dst,
                                                  unsigned lane, unsigned nlanes) {
    constexpr unsigned dims = CONFIG_T::dims;
    constexpr unsigned A   = CONFIG_T::to_shape[dims - 2];
    constexpr unsigned Bb  = CONFIG_T::to_shape[dims - 1];
    constexpr unsigned srcRowStride = CONFIG_T::perm_strides[dims - 1];
    constexpr unsigned srcColStride = CONFIG_T::perm_strides[dims - 2];
    static_assert(srcColStride == 1,
                  "ND batched transpose: second-innermost output axis must be "
                  "source-contiguous");
    // Innermost-axis (input0 contract) padding: widen each transposed row's stride.
    constexpr bool PAD = (PAD_STRIDE > PAD_INNER);
    static_assert(!PAD || Bb == PAD_INNER,
                  "ND batched transpose: a padded innermost axis must be the contract dim");
    constexpr unsigned dstRS = PAD ? PAD_STRIDE : Bb;        // transposed-row stride in dst
    constexpr unsigned block = A * Bb;                       // source / N-decode geometry (unpadded)
    constexpr unsigned dblock = DST_BLOCK ? DST_BLOCK : A * dstRS;  // per-batch dst block
    constexpr unsigned nbatch = CONFIG_T::N / block;

    // Does a whole per-batch [Bb, A] block (src + dst + tmp UB tiles) fit UB?
    constexpr unsigned blk = 32u / sizeof(data_T);
    constexpr unsigned blockBytes =
        (Bb * align_up(A, blk) + 2u * A * align_up(Bb, 16u)) * sizeof(data_T);
    constexpr bool FITS = blockBytes <= 184u * 1024u;

    if constexpr (FITS) {
        unsigned chunk = (nbatch + nlanes - 1) / nlanes;
        unsigned b0 = lane * chunk;
        unsigned b1 = (b0 + chunk) < nbatch ? (b0 + chunk) : nbatch;

        // Odometer over the batch axes [0, dims-2): the trailing two axes form the
        // per-batch [Bb, A] transpose, so only the leading axes index the source base.
        PermOdometer<CONFIG_T, dims - 2> odo;
        odo.init(b0);
        for (unsigned b = b0; b < b1; b++, odo.advance()) {
            ttrans_block_inline<data_T, Bb, A, srcRowStride, dstRS>(src + odo.base, dst + (size_t)b * dblock);
        }
    } else {
        // Blocked: tile each [Bb, A] block into BR x BC sub-blocks; flatten
        // (batch, sub-block) into a 1D work space across lanes. Source sub-block
        // (r0 over Bb at srcRowStride, c0 over A at unit stride); its [cb, rb]
        // transpose lands at dst row c0 (stride dstRS), col r0.
        constexpr unsigned BR = 64;
        constexpr unsigned BC = 64;
        constexpr unsigned nBR = (Bb + BR - 1) / BR;
        constexpr unsigned nBC = (A + BC - 1) / BC;
        constexpr unsigned bpb = nBR * nBC;            // sub-blocks per batch
        constexpr unsigned totalWork = nbatch * bpb;

        unsigned chunk = (totalWork + nlanes - 1) / nlanes;
        unsigned w0 = lane * chunk;
        unsigned w1 = (w0 + chunk) < totalWork ? (w0 + chunk) : totalWork;

        for (unsigned w = w0; w < w1; w++) {
            unsigned b = w / bpb;
            unsigned sb = w - b * bpb;
            unsigned rem = b, src_base = 0;
            for (int k = int(dims) - 3; k >= 0; k--) {
                unsigned e = CONFIG_T::to_shape[k];
                src_base += (rem % e) * CONFIG_T::perm_strides[k];
                rem /= e;
            }
            unsigned r0 = (sb / nBC) * BR;             // along Bb (source rows)
            unsigned c0 = (sb % nBC) * BC;             // along A  (source cols)
            unsigned rb = (Bb - r0) < BR ? (Bb - r0) : BR;
            unsigned cb = (A - c0) < BC ? (A - c0) : BC;
            ttrans_block_dyn_inline<data_T, BR, BC, srcRowStride, dstRS>(
                src + src_base + (size_t)r0 * srcRowStride + c0,
                dst + (size_t)b * dblock + (size_t)c0 * dstRS + r0, rb, cb);
        }
    }
}

// Transpose dispatch body: identity -> tiled copy (chunked across lanes, any rank);
// 2D non-identity -> TTRANS (block grid distributed across lanes); N-D non-identity
// -> innermost-preserved strided gather, or batched 2D TTRANS over the leading axes
// (see the "N-D non-identity transpose" helpers above). `lane`/`nlanes`
// are the caller's vector-lane index/count: standalone pure-AIV launch passes
// block_idx/block_num; the fused mix-kernel passes block_idx*2+subblockid over
// block_num*2 sub-block lanes.
// PAD_INNER/PAD_STRIDE pad the destination's innermost (contract) dim from
// PAD_INNER to PAD_STRIDE (>= PAD_INNER) for an input0 transpose feeding the Cube;
// the pad region is pre-zeroed by the caller. Both 0 (or equal) ⇒ no padding.
template <typename data_T, typename CONFIG_T, unsigned PAD_INNER = 0, unsigned PAD_STRIDE = 0>
AICORE inline void transpose_inline(__gm__ const data_T* data, __gm__ data_T* res,
                                    unsigned lane, unsigned nlanes) {
    if constexpr (IsIdentityPerm<CONFIG_T>::value) {
        if constexpr (PAD_INNER > 0 && PAD_STRIDE > PAD_INNER) {
            // Identity input feeding the Cube with a padded contract dim: copy the
            // [N/PAD_INNER, PAD_INNER] rows into a PAD_STRIDE-wide destination.
            padded_copy_inline<data_T>(data, res, CONFIG_T::N / PAD_INNER, PAD_INNER, PAD_STRIDE, lane, nlanes);
        } else {
            unsigned chunk = (CONFIG_T::N + nlanes - 1) / nlanes;
            // Lane-distribution granule (keep 32B-aligned lane boundaries) — NOT the
            // copy chunk width; transpose_copy_inline sizes its own wide UB tile.
            constexpr unsigned LANE_GRAIN = 128;
            chunk = (chunk + LANE_GRAIN - 1) / LANE_GRAIN * LANE_GRAIN;
            unsigned start = lane * chunk;
            unsigned end = (start + chunk) < CONFIG_T::N ? (start + chunk) : CONFIG_T::N;
            transpose_copy_inline<data_T, CONFIG_T>(data, res, start, end);
        }
    } else if constexpr (CONFIG_T::dims == 2) {
        transpose_2d_inline<data_T, CONFIG_T, PAD_STRIDE>(data, res, lane, nlanes);
    } else if constexpr (CONFIG_T::perm[CONFIG_T::dims - 1] == CONFIG_T::dims - 1) {
        // N-D, innermost axis preserved -> strided gather of contiguous inner runs.
        transpose_nd_copy_inline<data_T, CONFIG_T, PAD_INNER, PAD_STRIDE>(data, res, lane, nlanes);
    } else {
        // N-D, innermost axis changes -> batched 2D TTRANS over the leading axes.
        transpose_nd_batched_2d_inline<data_T, CONFIG_T, PAD_INNER, PAD_STRIDE>(data, res, lane, nlanes);
    }
}

// input1 marshalling for K!=C with batching (I>1) and a *non-identity* permutation.
// input1's matmul layout is [batch.., C, L1]; the contract C is the second-innermost
// (row) axis, so padding lands each batch's ROWS(=C) valid rows at the front of its
// PAD_ROWS(=K)-row slot (per-batch dst block PAD_ROWS*INNER, the tail rows pre-zeroed)
// -- the row-pad analogue of transpose_inline's innermost-pad. Identity input1 is the
// simple contiguous repack and is handled by the caller (batched_pad_copy_inline);
// only the two non-identity N-D shapes reach here.
template <typename data_T, typename CONFIG_T, unsigned ROWS, unsigned PAD_ROWS, unsigned INNER>
AICORE inline void transpose_inline_rowpad(__gm__ const data_T* data, __gm__ data_T* res,
                                           unsigned lane, unsigned nlanes) {
    static_assert(CONFIG_T::dims >= 3,
                  "input1 row-pad transpose: batching (I>1) implies a >=3D operand");
    constexpr unsigned PAD_BATCH = PAD_ROWS * INNER;   // K*L1 per-batch dst block
    if constexpr (CONFIG_T::perm[CONFIG_T::dims - 1] == CONFIG_T::dims - 1) {
        // innermost (L1) preserved -> strided gather; pad each batch's C rows into
        // its K-row slot via the (ROWS, PAD_BATCH) knob (innermost L1 unpadded).
        transpose_nd_copy_inline<data_T, CONFIG_T, 0, 0, ROWS, PAD_BATCH>(data, res, lane, nlanes);
    } else {
        // innermost changes -> batched 2D TTRANS; the trailing axes are [C, L1], so
        // override the per-batch dst block C*L1 -> K*L1 (DST_BLOCK); L1 stays unpadded.
        transpose_nd_batched_2d_inline<data_T, CONFIG_T, 0, 0, PAD_BATCH>(data, res, lane, nlanes);
    }
}

template <typename data_T, typename CONFIG_T>
__global__ AICORE void transpose_kernel_standalone(__gm__ const data_T* data, __gm__ data_T* res) {
    if constexpr (DAV_VEC) {

        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        transpose_inline<data_T, CONFIG_T>(data, res, get_block_idx(), get_block_num());
    }
}

template <typename data_T, typename CONFIG_T>
void transpose(const data_T* data, data_T* res, void* stream = nullptr) {
    int64_t core_num = cached_core_num();

    transpose_kernel_standalone<data_T, CONFIG_T><<<core_num, nullptr, stream>>>(data, res);
}

} // namespace pto_einsum
