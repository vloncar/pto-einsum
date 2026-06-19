#pragma once
#include <stdint.h>
#include <pto/pto-inst.hpp>
#include <pto/common/debug.h>
#include "acl/acl.h"

using namespace pto;

#ifndef DAV_VEC
#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif
#endif

#ifndef DAV_CUBE
#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif
#endif

// FFTS control-address query used by the fused kernel's host launcher. We
// forward-declare it instead of including <runtime/rt_ffts.h>, which drags in a
// profiling header absent from some CANN toolkits. Provided by libruntime
// (link -lruntime). The kernel-side set_ffts_base_addr comes from pto-inst.hpp.
extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace pto_einsum {

// ─── SyncAll: full cross-core barrier ────────────────────────
constexpr uint16_t SYNC_AIV_FLAG = 12;
constexpr uint16_t SYNC_AIC_FLAG = 11;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 13;
constexpr uint16_t SYNC_AIV_ONLY_ALL = 14;
constexpr uint16_t SYNC_MODE_SHIFT_VALUE = 4;
constexpr uint16_t SYNC_FLAG_SHIFT_VALUE = 8;

AICORE inline uint16_t GetffstMsg(uint16_t mode, uint16_t flagId) {
  return (0x1 + ((mode & 0x3) << SYNC_MODE_SHIFT_VALUE) +
          ((flagId & 0xf) << SYNC_FLAG_SHIFT_VALUE));
}

template <bool isAIVOnly = true>
AICORE inline void SyncAll() {
  pipe_barrier(PIPE_ALL);
  if constexpr (isAIVOnly) {
    ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_AIV_ONLY_ALL));
    wait_flag_dev(SYNC_AIV_ONLY_ALL);
    return;
  }
#if defined(__DAV_CUBE__)
  wait_flag_dev(SYNC_AIV_FLAG);
  ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_AIC_FLAG));
  wait_flag_dev(SYNC_AIC_FLAG);
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIC_AIV_FLAG));
#elif defined(__DAV_VEC__)
  ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_AIV_FLAG));
  wait_flag_dev(SYNC_AIC_AIV_FLAG);
#endif
}

// 1. Transpose input/output (Vector core). TTRANS hardware tile transpose
// (vnchwconv) for 2D non-identity permutations; TLOAD/TSTORE tiled copy for
// identity permutations.

// constexpr check: is the permutation the identity?
template <typename CONFIG_T, unsigned D, bool InBounds>
struct IsIdentityPermImpl;

template <typename CONFIG_T, unsigned D>
struct IsIdentityPermImpl<CONFIG_T, D, false> {
    // D >= CONFIG_T::dims: past the end, all checked dims were identity
    static constexpr bool value = true;
};

template <typename CONFIG_T, unsigned D>
struct IsIdentityPermImpl<CONFIG_T, D, true> {
    // D < CONFIG_T::dims: check this dim and recurse
    static constexpr bool value = (CONFIG_T::perm[D] == D) &&
        IsIdentityPermImpl<CONFIG_T, D + 1, (D + 1 < CONFIG_T::dims)>::value;
};

template <typename CONFIG_T>
struct IsIdentityPerm {
    static constexpr bool value = IsIdentityPermImpl<CONFIG_T, 0, (0 < CONFIG_T::dims)>::value;
};

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
    constexpr unsigned TILE_COLS = 128;  // elems/row; 32-byte aligned for fp16
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
    constexpr unsigned TILE_COLS = 128;
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
    constexpr unsigned TILE_COLS = 128;
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

// Align `v` up to the next multiple of `m` (m a power-of-two factor of the dims
// we care about), used to pad UB tile row-strides for the TTRANS hardware path.
AICORE inline constexpr unsigned align_up(unsigned v, unsigned m) { return (v + m - 1) / m * m; }

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
    constexpr unsigned TILE_COLS = 128;
    using TileData = Tile<TileType::Vec, data_T, 1, TILE_COLS, BLayout::RowMajor, -1, -1>;

    TileData ubTile(1, TILE_COLS);
    TASSIGN(ubTile, 0x0);

    unsigned chunk = (nouter + nlanes - 1) / nlanes;
    unsigned g0 = lane * chunk;
    unsigned g1 = (g0 + chunk) < nouter ? (g0 + chunk) : nouter;

    for (unsigned g = g0; g < g1; g++) {
        // Decode the flat outer index into its per-axis components (innermost-most
        // outer axis first) and accumulate the source base offset.
        unsigned rem = g, src_base = 0;
        for (int k = int(dims) - 2; k >= 0; k--) {
            unsigned e = CONFIG_T::to_shape[k];
            src_base += (rem % e) * CONFIG_T::perm_strides[k];
            rem /= e;
        }
        __gm__ const data_T* s = src + src_base;
        size_t doff = (ROWS > 0) ? (size_t)(g / ROWS) * PAD_BATCH + (size_t)(g % ROWS) * dstStride
                                 : (size_t)g * dstStride;
        __gm__ data_T* d = dst + doff;
        for (unsigned c = 0; c < inner; c += TILE_COLS) {
            unsigned w = (c + TILE_COLS) <= inner ? TILE_COLS : (inner - c);
            copy_chunk_through_ub<data_T, TILE_COLS>(ubTile, s + c, d + c, w);
        }
    }
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

        for (unsigned b = b0; b < b1; b++) {
            unsigned rem = b, src_base = 0;
            for (int k = int(dims) - 3; k >= 0; k--) {
                unsigned e = CONFIG_T::to_shape[k];
                src_base += (rem % e) * CONFIG_T::perm_strides[k];
                rem /= e;
            }
            ttrans_block_inline<data_T, Bb, A, srcRowStride, dstRS>(src + src_base, dst + (size_t)b * dblock);
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

// 2. Einsum contraction (Cube core).
//
// The per-batch output (L0 x L1) is partitioned into an (Mt x Nt) tile grid over
// the padded dims M/N; the contraction is walked in Kt-wide steps and accumulated
// on-chip (TMATMUL then TMATMUL_ACC). Each output tile is one unit of work; all
// tiles across all batches are flattened into a 1D index space and
// block-distributed across the AI cores.
//
// Fractal alignment: the on-chip Mat/Left/Right/Acc tiles need multiples of 16
// (NZ Mat InnerRows==16; Acc fractalCSize InnerRows==InnerCols==16). 16 is a
// multiple of the C0 block (fp32=8, fp16=16), so InnerCols==C0 holds too. Tile
// sizes are min(config, padded-dim), and the padded dim must divide evenly (no
// partial tiles).
//
// The contraction loads full-Kt-wide (valid col == Kt): the caller guarantees the
// GM A/B buffers are K-wide and zero-padded, so every Kt step has a full (>=C0)
// valid column count, sidestepping the Left-operand partial-C0 ND->NZ
// row-misplacement bug (the zero padding adds 0). Only M (rows) and N (cols) carry
// a partial valid extent on boundary tiles (GM is L0xK / KxL1, not padded),
// handled via SetValidRow/SetValidCol.
//
// Padded matmul geometry, the single source of truth shared by the per-tile
// kernel, the batched scheduler, and the host/kernel split-K eligibility check, so
// they all agree on the padded dims, tile clamps, tile counts, and split-K
// geometry. Mirrors the fractal-alignment (M/K/N = ⌈dim/16⌉·16) and tile-clamp
// (min(config, padded-dim)) rules described in Phase B.
template <typename CONFIG_T>
struct MatmulGeom {
    static constexpr unsigned L0 = CONFIG_T::n_free0;
    static constexpr unsigned L1 = CONFIG_T::n_free1;
    static constexpr unsigned C  = CONFIG_T::n_contract;
    static constexpr unsigned I  = CONFIG_T::n_inplace;
    static constexpr unsigned M  = (L0 + 15) / 16 * 16;
    static constexpr unsigned N  = (L1 + 15) / 16 * 16;
    static constexpr unsigned K  = (C + 15) / 16 * 16;
    static constexpr unsigned Mt = CONFIG_T::tile_m < M ? CONFIG_T::tile_m : M;
    static constexpr unsigned Nt = CONFIG_T::tile_n < N ? CONFIG_T::tile_n : N;
    static constexpr unsigned Kt = CONFIG_T::tile_k < K ? CONFIG_T::tile_k : K;
    // Operand row/col padding for partial tiles. A boundary tile whose padded dim is
    // not a whole multiple of the clamped tile would leave fully-empty trailing 16-row
    // fractal blocks in the Mat operand, which the ND->NZ load misplaces (corrupting
    // the tile). So pad A's rows up to Ma and B's cols up to Na (whole-tile multiples),
    // zero-filled, so every loaded tile is full; the output store still clamps to the
    // real L0/L1. The padded per-batch block has a single-batch layout only, so a
    // genuinely partial dim (A_PADDED/B_PADDED) requires I==1 (asserted in the kernel).
    static constexpr unsigned Ma = (M + Mt - 1) / Mt * Mt;   // A rows -> whole tiles
    static constexpr unsigned Na = (N + Nt - 1) / Nt * Nt;   // B cols -> whole tiles
    static constexpr bool A_PADDED = (Ma > M);
    static constexpr bool B_PADDED = (Na > N);
    // Per-batch buffer extents the matmul addresses: the padded count when partial
    // (read a full tile from the zero-filled tail), else the real L0/L1 (the existing
    // path -- a full grid whose last tile carries a >Mt-16 partial valid extent).
    static constexpr unsigned Arows = A_PADDED ? Ma : L0;
    static constexpr unsigned Bcols = B_PADDED ? Na : L1;
    static constexpr unsigned nN = Na / Nt;
    static constexpr unsigned nK = K / Kt;
    static constexpr unsigned tiles_per_batch = (Ma / Mt) * nN;
    static constexpr unsigned total_tiles = I * tiles_per_batch;
};

// One output tile, basic per-Kt schedule: contract the K-tile range
// [kStart, kStart+kCount) of tile `t` and store the result. This is the split-K
// tile (the plain schedule uses the deeper-pipelined matmul_one_tile_deep below).
// ATOMIC selects an atomic-add store — used by split-K, where several cores each
// contract a disjoint K-slice of the same tile into a pre-zeroed output — versus a
// plain overwriting store.
template <typename data_T, typename CONFIG_T, bool ATOMIC>
AICORE inline void matmul_one_tile_inline(__gm__ const data_T* ws0, __gm__ const data_T* ws1,
                                          __gm__ float* ws_res, unsigned t,
                                          unsigned kStart, unsigned kCount) {
        using G = MatmulGeom<CONFIG_T>;
        constexpr unsigned L0 = G::L0;
        constexpr unsigned L1 = G::L1;
        constexpr unsigned K  = G::K;
        constexpr unsigned Mt = G::Mt;
        constexpr unsigned Nt = G::Nt;
        constexpr unsigned Kt = G::Kt;
        constexpr unsigned nN = G::nN;
        constexpr unsigned tiles_per_batch = G::tiles_per_batch;
        // Padded-operand geometry (see MatmulGeom). When a dim is partial the buffer
        // carries Ma/Na rows/cols (zero-filled tail) and the matmul loads a *full* tile
        // so no fractal block is empty; the output store still clamps to real L0/L1.
        static_assert(G::I == 1 || (!G::A_PADDED && !G::B_PADDED),
                      "partial M/N tiles (free dim not a multiple of the tile) require I==1");
        constexpr unsigned Arows = G::Arows;
        constexpr unsigned Bcols = G::Bcols;

        using ShapeA = pto::Shape<1, 1, 1, -1, Kt>;
        using StrideA = pto::Stride<1, 1, 1, K, 1>;
        using ShapeB = pto::Shape<1, 1, 1, Kt, -1>;
        using StrideB = pto::Stride<1, 1, 1, Bcols, 1>;
        using ShapeC = pto::Shape<1, 1, 1, -1, -1>;
        using StrideC = pto::Stride<1, 1, 1, L1, 1>;
        using MatTileA = pto::Tile<pto::TileType::Mat, data_T, Mt, Kt, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, 512>;
        using MatTileB = pto::Tile<pto::TileType::Mat, data_T, Kt, Nt, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, 512>;
        using LeftTileA = pto::TileLeft<data_T, Mt, Kt, -1, -1>;
        using RightTileB = pto::TileRight<data_T, Kt, Nt, -1, -1>;
        using AccTileC = pto::TileAcc<float, Mt, Nt, -1, -1>;

        unsigned i = t / tiles_per_batch;
        unsigned rem = t % tiles_per_batch;
        unsigned row0 = (rem / nN) * Mt;  // output row offset within the matrix
        unsigned col0 = (rem % nN) * Nt;  // output col offset within the matrix

        // Boundary tiles store fewer real rows/cols than the padded tile (output GM is
        // L0xL1). row0 < L0 and col0 < L1 always hold. The *load* reads a full tile when
        // the operand is padded (Arows/Bcols-tall zero-filled buffer), else the real
        // valid extent (the aligned path, whose last tile is a >Mt-16 partial extent).
        unsigned validM = (L0 - row0) < Mt ? (L0 - row0) : Mt;
        unsigned validN = (L1 - col0) < Nt ? (L1 - col0) : Nt;
        unsigned loadM = G::A_PADDED ? Mt : validM;
        unsigned loadN = G::B_PADDED ? Nt : validN;

        // Double buffering. The L1 Mat tiles are ping-ponged across two buffers
        // so the GM->L1 load (MTE2, the slow stage) of the next K step overlaps
        // the matmul (M) of the current step. The L0 Left/Right
        // tiles stay single-buffered: a full Mt x Kt tile already fills L0A/L0B
        // for the default tile, so two won't fit; the mov->matmul chain stays
        // serialized but the expensive GM loads are hidden behind compute.
        constexpr unsigned matBytesA = Mt * Kt * sizeof(data_T);
        constexpr unsigned matBytesB = Kt * Nt * sizeof(data_T);
        static_assert(2 * matBytesA + 2 * matBytesB <= 512u * 1024u,
                      "Double-buffered L1 Mat tiles exceed the 512 KB CBUF.");

        MatTileA matA[2];
        MatTileB matB[2];
        LeftTileA leftA;
        RightTileB rightB;
        AccTileC accC;

        TASSIGN(matA[0], 0);
        TASSIGN(matA[1], matBytesA);
        TASSIGN(matB[0], 2 * matBytesA);
        TASSIGN(matB[1], 2 * matBytesA + matBytesB);
        TASSIGN(leftA, 0);
        TASSIGN(rightB, 0);
        TASSIGN(accC, 0);

        // Valid extents are constant across the K-loop, so set them once. The Mat/Left/
        // Right operands load the full tile (loadM/loadN) so the ND->NZ packing has no
        // empty fractal block; only the accumulator/store carry the real validM/validN.
        matA[0].SetValidRow(loadM); matA[0].SetValidCol(Kt);
        matA[1].SetValidRow(loadM); matA[1].SetValidCol(Kt);
        matB[0].SetValidRow(Kt);     matB[0].SetValidCol(loadN);
        matB[1].SetValidRow(Kt);     matB[1].SetValidCol(loadN);
        leftA.SetValidRow(loadM);   leftA.SetValidCol(Kt);
        rightB.SetValidRow(Kt);      rightB.SetValidCol(loadN);
        accC.SetValidRow(validM);    accC.SetValidCol(validN);

        // Event IDs (per L1 buffer where two can be in flight):
        //   ID0/ID1  load-done   (MTE2 -> MTE1, RAW: mov waits its load)
        //   ID2/ID3  matA free   (MTE1 -> MTE2, WAR: load waits prior mov)
        //   ID4      mov-done    (MTE1 -> M,    RAW: matmul waits its mov)
        //   ID5      leftA free  (M    -> MTE1, WAR: mov waits prior matmul)
        // j is the local K-step (0..kCount-1); the global K-tile is kStart + j, so
        // the accumulator-init (TMATMUL vs TMATMUL_ACC) keys on the local index.
        for (unsigned j = 0; j < kCount; j++) {
            unsigned p = j & 1;
            event_t e_ld  = p ? EVENT_ID1 : EVENT_ID0;
            event_t e_ldf = p ? EVENT_ID3 : EVENT_ID2;
            unsigned k0 = (kStart + j) * Kt;
            pto::GlobalTensor<data_T, ShapeA, StrideA> aGlobal(
                const_cast<__gm__ data_T*>(ws0 + i * Arows * K + row0 * K + k0), ShapeA(loadM));
            pto::GlobalTensor<data_T, ShapeB, StrideB> bGlobal(
                const_cast<__gm__ data_T*>(ws1 + i * K * Bcols + k0 * Bcols + col0), ShapeB(loadN));

            // Load GM -> L1 buffer p (wait until a prior mov freed this buffer).

            if (j >= 2) wait_flag(PIPE_MTE1, PIPE_MTE2, e_ldf);

            TLOAD(matA[p], aGlobal);
            TLOAD(matB[p], bGlobal);

            set_flag(PIPE_MTE2, PIPE_MTE1, e_ld);

            // Move L1 -> L0 (single buffer): wait this load, and the prior matmul.

            wait_flag(PIPE_MTE2, PIPE_MTE1, e_ld);
            if (j >= 1) wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);

            TMOV(leftA, matA[p]);
            TMOV(rightB, matB[p]);

            set_flag(PIPE_MTE1, PIPE_MTE2, e_ldf);    // matA[p] free to reload
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);   // leftA/rightB ready

            // Matmul: first Kt step initialises the accumulator, the rest accumulate.

            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);

            if (j == 0) {
                TMATMUL(accC, leftA, rightB);
            } else {
                TMATMUL_ACC(accC, leftA, rightB);
            }

            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);   // leftA/rightB free

        }

        // Drain the WAR-free flags the final step(s) set but nothing consumed,
        // so no flag state leaks into the next output tile.

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        if (kCount >= 2) wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);   // RAW: the L0C store waits the matmul
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        pto::GlobalTensor<float, ShapeC, StrideC> cGlobal(
            ws_res + i * L0 * L1 + row0 * L1 + col0, ShapeC(validM, validN));
        if constexpr (ATOMIC) {
            TSTORE<AccTileC, pto::GlobalTensor<float, ShapeC, StrideC>, pto::AtomicType::AtomicAdd>(cGlobal, accC);
        } else {
            TSTORE(cGlobal, accC);
        }

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);   // WAR: next tile's matmul reuses accC (L0C)
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

// Deep-pipeline granularities for matmul_one_tile_deep, derived at compile time
// from the tile dims and dtype size. Kq is the L0 compute granularity: the largest
// multiple of 16 dividing K whose ping-pong Left/Right tiles fit their 64 KB L0
// banks. Kd is the L1 chunk depth: the largest multiple of Kq dividing K whose
// double-buffered A+B Mat tiles fit the 512 KB CBUF. Both divide K exactly, so the
// chunk loop (K/Kd) and phase loop (Kd/Kq) are clean; awkward/small K collapses
// gracefully to Kd==Kq==K (a single chunk, single phase = the plain matmul).
AICORE inline constexpr unsigned deep_pick_kq(unsigned Mt, unsigned Nt, unsigned K, unsigned ds) {
    unsigned capA = 65536u / (2u * Mt * ds);
    unsigned capB = 65536u / (2u * Nt * ds);
    unsigned cap = capA < capB ? capA : capB;
    cap -= cap % 16u;
    if (cap > K) cap = K;
    if (cap < 16u) return 16u;
    for (unsigned d = cap; d >= 16u; d -= 16u)
        if (K % d == 0u) return d;
    return 16u;
}
AICORE inline constexpr unsigned deep_pick_kd(unsigned Mt, unsigned Nt, unsigned K, unsigned Kq, unsigned ds) {
    unsigned cap = (512u * 1024u) / (2u * (Mt + Nt) * ds);
    cap -= cap % Kq;
    if (cap > K) cap = K;
    if (cap < Kq) return Kq;
    for (unsigned d = cap; d >= Kq; d -= Kq)
        if (K % d == 0u) return d;
    return Kq;
}

// One output tile, deep-pipelined — the compute schedule from the pto-kernels
// matmul-swizzle example, ported and generalized to fp32/fp16 and the einsum's
// padded/batched shapes (the example's L2 tile-swizzle was not kept — it measured a
// no-op/regression here). It always contracts the full K [0, K) internally, so it
// slots straight into the plain schedule; the split-K path keeps matmul_one_tile_inline.
//
// Two overlap mechanisms hide the memory pipes behind the cube, which the basic
// per-Kt routine (single-buffered L0, serialized TMOV->TMATMUL) does not:
//   1. L1 chunk reuse: A [Mt,Kd] and B [Kd,Nt] are loaded once per Kd-chunk and
//      TEXTRACTed into nP=Kd/Kq L0 sub-tiles, amortizing the big GM->L1 transfers.
//   2. L0 ping-pong: the L0 Left/Right tiles double-buffer, so the TEXTRACT of the
//      next phase (MTE1) overlaps the TMATMUL of the current phase (M) instead of
//      serializing. The L1 A/B chunks also double-buffer to prefetch chunk c+1's
//      GM->L1 load under chunk c's compute.
// Each tile is self-contained (inits its event flags, drains them before the
// store), so the scheduler can hand tiles to cores in any order.
template <typename data_T, typename CONFIG_T>
AICORE inline void matmul_one_tile_deep(__gm__ const data_T* ws0, __gm__ const data_T* ws1,
                                        __gm__ float* ws_res, unsigned t) {
        using G = MatmulGeom<CONFIG_T>;
        constexpr unsigned L0 = G::L0;
        constexpr unsigned L1 = G::L1;
        constexpr unsigned K  = G::K;
        constexpr unsigned Mt = G::Mt;
        constexpr unsigned Nt = G::Nt;
        constexpr unsigned nN = G::nN;
        constexpr unsigned tiles_per_batch = G::tiles_per_batch;
        // Padded-operand geometry (see MatmulGeom): a partial dim loads a full tile from
        // an Ma/Na-padded zero-filled buffer; the store still clamps to real L0/L1.
        static_assert(G::I == 1 || (!G::A_PADDED && !G::B_PADDED),
                      "partial M/N tiles (free dim not a multiple of the tile) require I==1");
        constexpr unsigned Arows = G::Arows;
        constexpr unsigned Bcols = G::Bcols;

        constexpr unsigned ds  = sizeof(data_T);
        constexpr unsigned Kq  = deep_pick_kq(Mt, Nt, K, ds);
        constexpr unsigned Kd  = deep_pick_kd(Mt, Nt, K, Kq, ds);
        constexpr unsigned nKd = K / Kd;   // L1 chunk count
        constexpr unsigned nP  = Kd / Kq;  // L0 phases per chunk

        using ShapeA = pto::Shape<1, 1, 1, -1, Kd>;
        using StrideA = pto::Stride<1, 1, 1, K, 1>;
        using ShapeB = pto::Shape<1, 1, 1, Kd, -1>;
        using StrideB = pto::Stride<1, 1, 1, Bcols, 1>;
        using ShapeC = pto::Shape<1, 1, 1, -1, -1>;
        using StrideC = pto::Stride<1, 1, 1, L1, 1>;
        using MatTileA = pto::Tile<pto::TileType::Mat, data_T, Mt, Kd, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, 512>;
        using MatTileB = pto::Tile<pto::TileType::Mat, data_T, Kd, Nt, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, 512>;
        using LeftTileA = pto::TileLeft<data_T, Mt, Kq, -1, -1>;
        using RightTileB = pto::TileRight<data_T, Kq, Nt, -1, -1>;
        using AccTileC = pto::TileAcc<float, Mt, Nt, -1, -1>;

        unsigned i = t / tiles_per_batch;
        unsigned rem = t % tiles_per_batch;
        unsigned row0 = (rem / nN) * Mt;
        unsigned col0 = (rem % nN) * Nt;
        unsigned validM = (L0 - row0) < Mt ? (L0 - row0) : Mt;
        unsigned validN = (L1 - col0) < Nt ? (L1 - col0) : Nt;
        unsigned loadM = G::A_PADDED ? Mt : validM;   // full-tile load when padded
        unsigned loadN = G::B_PADDED ? Nt : validN;

        constexpr unsigned matBytesA = Mt * Kd * ds;
        constexpr unsigned matBytesB = Kd * Nt * ds;
        static_assert(2 * matBytesA + 2 * matBytesB <= 512u * 1024u,
                      "Deep-pipeline L1 A/B chunks exceed the 512 KB CBUF.");
        constexpr unsigned l0aBytes = Mt * Kq * ds;
        constexpr unsigned l0bBytes = Kq * Nt * ds;
        static_assert(2 * l0aBytes <= 64u * 1024u, "Deep-pipeline L0A ping-pong exceeds L0A.");
        static_assert(2 * l0bBytes <= 64u * 1024u, "Deep-pipeline L0B ping-pong exceeds L0B.");

        MatTileA a_l1[2];
        MatTileB b_l1[2];
        LeftTileA a_l0[2];
        RightTileB b_l0[2];
        AccTileC c_l0;
        TASSIGN(a_l1[0], 0);
        TASSIGN(a_l1[1], matBytesA);
        TASSIGN(b_l1[0], 2 * matBytesA);
        TASSIGN(b_l1[1], 2 * matBytesA + matBytesB);
        TASSIGN(a_l0[0], 0);
        TASSIGN(a_l0[1], l0aBytes);
        TASSIGN(b_l0[0], 0);
        TASSIGN(b_l0[1], l0bBytes);
        TASSIGN(c_l0, 0);

        // Operands load the full tile (loadM/loadN) so the ND->NZ packing has no empty
        // fractal block; only the accumulator/store carry the real validM/validN.
        a_l1[0].SetValidRow(loadM); a_l1[0].SetValidCol(Kd);
        a_l1[1].SetValidRow(loadM); a_l1[1].SetValidCol(Kd);
        b_l1[0].SetValidRow(Kd);     b_l1[0].SetValidCol(loadN);
        b_l1[1].SetValidRow(Kd);     b_l1[1].SetValidCol(loadN);
        a_l0[0].SetValidRow(loadM); a_l0[0].SetValidCol(Kq);
        a_l0[1].SetValidRow(loadM); a_l0[1].SetValidCol(Kq);
        b_l0[0].SetValidRow(Kq);     b_l0[0].SetValidCol(loadN);
        b_l0[1].SetValidRow(Kq);     b_l0[1].SetValidCol(loadN);
        c_l0.SetValidRow(validM);    c_l0.SetValidCol(validN);

        const __gm__ data_T* aBase = ws0 + i * Arows * K + row0 * K;
        const __gm__ data_T* bBase = ws1 + i * K * Bcols + col0;

        // Init: both L0 ping-pong buffers free, both L1 A/B buffers free.
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);

        // Load chunk 0 into buffer 0 (A on event lanes 0/1, B on 2/3).
        {
            pto::GlobalTensor<data_T, ShapeA, StrideA> ag(const_cast<__gm__ data_T*>(aBase), ShapeA(loadM));
            pto::GlobalTensor<data_T, ShapeB, StrideB> bg(const_cast<__gm__ data_T*>(bBase), ShapeB(loadN));
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
            TLOAD(a_l1[0], ag);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
            TLOAD(b_l1[0], bg);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
        }

        unsigned gphase = 0;
        for (unsigned c = 0; c < nKd; c++) {
            unsigned cur = c & 1;
            event_t aLd = cur ? EVENT_ID1 : EVENT_ID0;
            event_t bLd = cur ? EVENT_ID3 : EVENT_ID2;

            // Prefetch chunk c+1 into the other L1 buffer (overlaps this chunk's compute).
            if (c + 1 < nKd) {
                unsigned nb = cur ^ 1;
                event_t aLdN = nb ? EVENT_ID1 : EVENT_ID0;
                event_t bLdN = nb ? EVENT_ID3 : EVENT_ID2;
                unsigned k0 = (c + 1) * Kd;
                pto::GlobalTensor<data_T, ShapeA, StrideA> ag(const_cast<__gm__ data_T*>(aBase + k0), ShapeA(loadM));
                pto::GlobalTensor<data_T, ShapeB, StrideB> bg(const_cast<__gm__ data_T*>(bBase + size_t(k0) * Bcols), ShapeB(loadN));
                wait_flag(PIPE_MTE1, PIPE_MTE2, aLdN);
                TLOAD(a_l1[nb], ag);
                set_flag(PIPE_MTE2, PIPE_MTE1, aLdN);
                wait_flag(PIPE_MTE1, PIPE_MTE2, bLdN);
                TLOAD(b_l1[nb], bg);
                set_flag(PIPE_MTE2, PIPE_MTE1, bLdN);
            }

            // Wait for this chunk's GM->L1 load to complete.
            wait_flag(PIPE_MTE2, PIPE_MTE1, aLd);
            wait_flag(PIPE_MTE2, PIPE_MTE1, bLd);

            for (unsigned p = 0; p < nP; p++) {
                unsigned pp = gphase & 1;
                event_t l0f = pp ? EVENT_ID1 : EVENT_ID0;

                wait_flag(PIPE_M, PIPE_MTE1, l0f);              // L0[pp] free (matmul 2 phases ago)
                TEXTRACT(a_l0[pp], a_l1[cur], 0, p * Kq);
                TEXTRACT(b_l0[pp], b_l1[cur], p * Kq, 0);
                if (p == nP - 1) {                              // last extract frees this L1 chunk
                    set_flag(PIPE_MTE1, PIPE_MTE2, aLd);
                    set_flag(PIPE_MTE1, PIPE_MTE2, bLd);
                }
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);         // RAW: matmul waits its extract
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                if (gphase == 0)
                    TMATMUL(c_l0, a_l0[pp], b_l0[pp]);
                else
                    TMATMUL_ACC(c_l0, a_l0[pp], b_l0[pp]);
                set_flag(PIPE_M, PIPE_MTE1, l0f);               // L0[pp] consumed, free to extract
                gphase++;
            }
        }

        // Drain: regardless of nKd/nP, both L1 A buffers (ID0/ID1), both L1 B
        // buffers (ID2/ID3) and both L0 ping-pong buffers (ID0/ID1) end up signalled
        // "free" and unconsumed; wait them so no flag state leaks into the next tile.
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        // Store accumulator -> GM (F32 acc, NZ->ND), then free L0C for the next tile.
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        pto::GlobalTensor<float, ShapeC, StrideC> cg(
            ws_res + i * L0 * L1 + row0 * L1 + col0, ShapeC(validM, validN));
        TSTORE(cg, c_l0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

// Factored out so the standalone kernel and the fused kernel share it; the caller
// guards with `if constexpr (DAV_CUBE)` and supplies the block index/count.
// SPLITK enables the split-K schedule (see batched_matmul_inline body); it is off
// for the standalone matmul and on only for fused configs with idle cores.
template <typename data_T, typename CONFIG_T, bool SPLITK = false>
AICORE inline void batched_matmul_inline(__gm__ const data_T* ws0, __gm__ const data_T* ws1,
                                         __gm__ float* ws_res, unsigned block_idx, unsigned block_num) {
        using G = MatmulGeom<CONFIG_T>;

        // The grid tiles over the operand-padded extents Ma/Na (whole-tile multiples by
        // construction) and K (tile_k divides padded K), so every tile is full; a
        // partial free dim is absorbed by the zero-padded operand buffers. Partial dims
        // currently have a single-batch layout only.
        static_assert(G::Ma % G::Mt == 0 && G::Na % G::Nt == 0 && G::K % G::Kt == 0,
                      "Padded grid Ma/Na/K must divide the (clamped) tile size.");
        static_assert(G::I == 1 || (!G::A_PADDED && !G::B_PADDED),
                      "partial M/N tiles (free dim not a multiple of the tile) require I==1");

        constexpr unsigned nK = G::nK;
        constexpr unsigned total_tiles = G::total_tiles;

        // Split-K. The output-tile grid is `total_tiles` units; the plain schedule
        // hands one tile to each core, so when total_tiles < block_num the surplus
        // cores idle (the dot product is the extreme: a single 1x1 tile over a long
        // K runs entirely on one core). Split-K instead assigns `ksplit` cores to
        // each tile, each contracting a disjoint K-slice and atomic-adding its
        // partial into the (caller-pre-zeroed) output. ksplit is chosen at runtime
        // from the real core count and clamped to a divisor of nK; ksplit==1 falls
        // back to the plain path. Only enabled (SPLITK) for fused configs whose
        // output is identity and tile grid is small — see einsum_fused_kernel.
        if constexpr (SPLITK) {
            unsigned ksplit = total_tiles ? block_num / total_tiles : 1u;
            if (ksplit > nK) ksplit = nK;
            while (ksplit > 1 && (nK % ksplit) != 0) ksplit--;
            if (ksplit >= 2) {
                unsigned total_work = total_tiles * ksplit;
                if (block_idx < total_work) {
                    unsigned kpc = nK / ksplit;        // K-tiles per slice
                    unsigned t = block_idx / ksplit;   // which output tile
                    unsigned s = block_idx % ksplit;   // which K-slice
                    matmul_one_tile_inline<data_T, CONFIG_T, true>(ws0, ws1, ws_res, t, s * kpc, kpc);
                }
                return;
            }
            // ksplit == 1: no idle cores to exploit, use the plain schedule below.
        }

        // Plain schedule: block-distribute the output tiles (one contiguous chunk per
        // core), full K per tile via the deep-pipelined tile. (An L2 tile-swizzle was
        // tried here and removed — measured a no-op on square grids, HBM-bound, and a
        // regression on non-square ones.)
        unsigned chunk = (total_tiles + block_num - 1) / block_num;
        unsigned start = block_idx * chunk;
        unsigned end = (start + chunk) < total_tiles ? (start + chunk) : total_tiles;
        for (unsigned t = start; t < end; t++) {
            matmul_one_tile_deep<data_T, CONFIG_T>(ws0, ws1, ws_res, t);
        }
}

// Number of AI cores on the active device, queried once per process. The count is
// fixed for the device model, so memoizing it keeps the per-launch exec path off the
// host ACL query round-trip (aclrtGetDevice + aclrtGetDeviceInfo), which otherwise ran
// on every launch and showed up in the launch-bound small-N regime. The function-local
// static initialises exactly once (thread-safe under C++11).
inline int64_t cached_core_num() {
    static int64_t core_num = [] {
        int32_t device_id = 0;
        aclrtGetDevice(&device_id);
        int64_t n = 1;
        aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_AICORE_CORE_NUM, &n);
        return n;
    }();
    return core_num;
}

// Standalone Cube matmul kernel: thin wrapper over batched_matmul_inline.
template <typename data_T, typename CONFIG_T>
__global__ AICORE void batched_matmul_kernel_standalone(__gm__ const data_T* ws0, __gm__ const data_T* ws1, __gm__ float* ws_res) {
    if constexpr (DAV_CUBE) {
        batched_matmul_inline<data_T, CONFIG_T>(ws0, ws1, ws_res, get_block_idx(), get_block_num());
    }
}

template <typename data_T, typename CONFIG_T>
void batched_matmul(const data_T* ws0, const data_T* ws1, float* ws_res, void* stream = nullptr) {
    int64_t core_num = cached_core_num();

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C  = CONFIG_T::n_contract;
    constexpr unsigned I  = CONFIG_T::n_inplace;
    constexpr unsigned K  = (C + 15) / 16 * 16;  // fractal-aligned contraction width

    if constexpr (K == C) {
        // Contraction already 16-aligned: the GM buffers are exactly K-wide.
        batched_matmul_kernel_standalone<data_T, CONFIG_T><<<core_num, nullptr, stream>>>(ws0, ws1, ws_res);
    } else {
        // The kernel expects the contraction dim to be K-wide and zero-padded. Build
        // padded copies of A (I*L0, C)->(I*L0, K) and B (I, C, L1)->(I, K, L1) with the
        // tail columns/rows zeroed, so the (un-)contracted padding contributes nothing.
        const size_t a_bytes = sizeof(data_T) * size_t(I) * L0 * K;
        const size_t b_bytes = sizeof(data_T) * size_t(I) * K * L1;

        void* a_pad = nullptr;
        void* b_pad = nullptr;
        aclrtMalloc(&a_pad, a_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        aclrtMalloc(&b_pad, b_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

        aclrtMemsetAsync(a_pad, a_bytes, 0, a_bytes, stream);
        aclrtMemsetAsync(b_pad, b_bytes, 0, b_bytes, stream);

        // A: each of the I*L0 rows holds C valid elements, padded out to K.
        aclrtMemcpy2dAsync(a_pad, K * sizeof(data_T),
                           ws0, C * sizeof(data_T),
                           C * sizeof(data_T), size_t(I) * L0,
                           ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        // B: each of the I batches holds C*L1 contiguous valid elements (rows 0..C of an
        // (K, L1) block); the remaining (K-C) rows stay zero.
        aclrtMemcpy2dAsync(b_pad, size_t(K) * L1 * sizeof(data_T),
                           ws1, size_t(C) * L1 * sizeof(data_T),
                           size_t(C) * L1 * sizeof(data_T), I,
                           ACL_MEMCPY_DEVICE_TO_DEVICE, stream);

        batched_matmul_kernel_standalone<data_T, CONFIG_T><<<core_num, nullptr, stream>>>(
            static_cast<data_T*>(a_pad), static_cast<data_T*>(b_pad), ws_res);

        aclrtSynchronizeStream(stream);
        aclrtFree(a_pad);
        aclrtFree(b_pad);
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
            constexpr unsigned TILE_COLS = 128;
            chunk = (chunk + TILE_COLS - 1) / TILE_COLS * TILE_COLS;
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

// ─── Elementwise multiply (Hadamard) ─────────────────────────────────────────
// res[i] = in0[i] * in1[i] for i in [0, N). A pure elementwise product: an einsum
// with no contracted index and identical index order on both inputs and the output
// (e.g. `TS,TS->TS`, the L-mask step of linear attention). This is *not* a matmul,
// so it is routed here instead of through the Cube, which would otherwise run N
// degenerate 1x1x1 matmuls. Streamed in flat full-width blocks through UB across all
// Vector lanes, so any N works with no size/alignment restriction.
//
// Output is always float. fp16 inputs are up-cast to float and multiplied in float
// (matching the matmul path, which also accumulates fp16 in float). The fp16 up-cast
// is two TCVTs into af/bf followed by an in-place TMUL that reads them: that read is
// a RAW hazard on the Vector pipe, so a pipe_barrier(PIPE_V) sits between the converts
// and the multiply. Without it the TMUL can read pre-convert data; on a large tile the
// multi-repeat TCVT latency hides the window, which is why the hazard masqueraded as a
// "small/partial-tile fp16 quirk" (it is not — bare TCVT is correct at every size). The
// barrier lets blocks carry a dynamic valid extent, so the tail is just a short block
// (no full-valid / backward-overlap dance, no minimum size). fp32 has no TCVT and so no
// intra-V hazard. See pto-einsum/bug_report/FINDINGS.md for the isolating reproducer.

// One block at `base`, covering `valid` (<= CAP) elements. CAP is the static tile
// capacity; `valid` is the dynamic transfer/compute extent (a short tail uses valid<CAP).
template <typename data_T, unsigned CAP>
AICORE inline void elementwise_mul_block(__gm__ const data_T* in0, __gm__ const data_T* in1,
                                         __gm__ float* res, unsigned base, unsigned valid) {
    using GlobIn  = GlobalTensor<data_T, pto::Shape<1, 1, 1, 1, CAP>, pto::Stride<1, 1, 1, CAP, 1>>;
    using GlobOut = GlobalTensor<float,  pto::Shape<1, 1, 1, 1, CAP>, pto::Stride<1, 1, 1, CAP, 1>>;
    using TileIn = Tile<TileType::Vec, data_T, 1, CAP, BLayout::RowMajor, -1, -1>;   // dynamic valid
    using TileF  = Tile<TileType::Vec, float,  1, CAP, BLayout::RowMajor, -1, -1>;

    TileIn a, b;
    TASSIGN(a, 0x0);
    TASSIGN(b, sizeof(data_T) * CAP);
    TileF af, bf;   // fp16 only: float up-casts of the inputs (unused for fp32)
    TASSIGN(af, sizeof(data_T) * CAP * 2);
    TASSIGN(bf, sizeof(data_T) * CAP * 2 + sizeof(float) * CAP);
    a.SetValidRow(1);  a.SetValidCol(valid);
    b.SetValidRow(1);  b.SetValidCol(valid);
    af.SetValidRow(1); af.SetValidCol(valid);
    bf.SetValidRow(1); bf.SetValidCol(valid);

    GlobIn g0(const_cast<__gm__ data_T*>(in0 + base));
    GlobIn g1(const_cast<__gm__ data_T*>(in1 + base));
    GlobOut gr(res + base);

    TLOAD(a, g0);
    TLOAD(b, g1);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);   // RAW: compute waits its loads
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (sizeof(data_T) == 2) {
        TCVT(af, a, RoundMode::CAST_RINT);     // half -> float (widening; exact)
        TCVT(bf, b, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);                  // RAW: TMUL reads the just-converted af/bf
        TMUL(af, af, bf);                      // product in float
    } else {
        TMUL(a, a, b);
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);   // RAW: store waits the compute
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    if constexpr (sizeof(data_T) == 2)
        TSTORE(gr, af);
    else
        TSTORE(gr, a);                         // fp32: data_T == float, store directly

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);  // WAR: next loads reuse the tiles
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

template <typename data_T, typename CONFIG_T>
AICORE inline void elementwise_mul_inline(__gm__ const data_T* in0, __gm__ const data_T* in1,
                                          __gm__ float* res, unsigned lane, unsigned nlanes) {
    constexpr unsigned N = CONFIG_T::N;
    constexpr unsigned TILE = 2048;  // per-block tile capacity
    // Each lane owns a contiguous TILE-aligned span and streams it in TILE blocks; the
    // final block is a short (valid < TILE) tail covering whatever remains. No backward
    // overlap and no size floor -- the dynamic valid extent + pipe_barrier handle any N.
    unsigned chunk = (N + nlanes - 1) / nlanes;
    chunk = (chunk + TILE - 1) / TILE * TILE;
    unsigned start = lane * chunk;
    unsigned end = (start + chunk) < N ? (start + chunk) : N;
    for (unsigned i = start; i < end; i += TILE) {
        unsigned valid = (i + TILE <= end) ? TILE : (end - i);
        elementwise_mul_block<data_T, TILE>(in0, in1, res, i, valid);
    }
}

template <typename data_T, typename CONFIG_T>
__global__ AICORE void elementwise_mul_kernel_standalone(
        __gm__ const data_T* in0, __gm__ const data_T* in1, __gm__ float* res) {
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        elementwise_mul_inline<data_T, CONFIG_T>(in0, in1, res, get_block_idx(), get_block_num());
    }
}

// One-shot launcher (synchronous), the elementwise analogue of einsum().
template <typename data_T, typename CONFIG_T>
void elementwise_mul(const data_T* in0, const data_T* in1, float* res, void* stream = nullptr) {
    int64_t core_num = cached_core_num();
    elementwise_mul_kernel_standalone<data_T, CONFIG_T><<<core_num, nullptr, stream>>>(in0, in1, res);
    aclrtSynchronizeStream(stream);
}

// Persistent-runner ABI parity with einsum_setup/exec/teardown. The op is a pure
// stream (no workspace), so setup returns a token byte so the runner caches a
// non-null handle, exec just launches, and einsum_teardown frees the byte.
inline void* elementwise_setup(void* /*stream*/ = nullptr) {
    void* workspace = nullptr;
    aclrtMalloc(&workspace, sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    return workspace;
}

template <typename data_T, typename CONFIG_T>
void elementwise_exec(const data_T* in0, const data_T* in1, float* res,
                      void* /*workspace*/, void* stream = nullptr) {
    int64_t core_num = cached_core_num();
    elementwise_mul_kernel_standalone<data_T, CONFIG_T><<<core_num, nullptr, stream>>>(in0, in1, res);
}

// ─── Broadcast / scaling product (no contraction, one full + one broadcast operand) ──
// einsums where the contracted index set is empty but the two operands differ in shape:
// one operand `A` carries every output axis (contiguous, in output order), the other `B`
// carries a strict subset -- so `res[i] = A[i] * B[proj(i)]`, with B broadcast over the
// axes it lacks. Examples (see bench_llm_contractions.py): rms_norm `bsd,d->bsd`,
// token_pos_embed `bsd,sd->bsd`, rope `bshd,sd->bshd`, alibi `hqk,h->hqk`. These are NOT
// matmuls (they would otherwise become a batch of degenerate 1xC matmuls and trip the
// batched-partial-tile guard), so they route here, to a Vector-only broadcast multiply.
//
// The output is partitioned into OUTER groups; each group is one contiguous `Inner`-element
// region of A (offset g*Inner). B's offset for a group is a mixed-radix projection of the
// group's multi-index onto B's axes (bcast_baseB) -- stride 0 on the axes B lacks. Two
// shapes of broadcast arise, chosen host-side:
//   * mode 0 (ColExpand): B varies along the innermost contiguous run (Cc columns) and is
//     reused across Rr "broadcast rows" (the next run of B-absent axes). Each block is a
//     [rb x Cc] HW column-expand multiply (TCOLEXPANDMUL: dst[r,c] = A[r,c] * B[c]).
//   * mode 1 (scalar): B is constant across the whole inner block (B has no inner axis),
//     so the group is multiplied by a single scalar B[baseB] (loaded once into UB, read
//     per group with GetValue, applied with TMULS).
// Output is always float; fp16 inputs up-cast with TCVT (PIPE_V RAW barrier before the
// multiply, as in the elementwise path). Work is distributed across all Vector lanes by
// flattening the (group, row)/(group, chunk) space, so OUTER==1 cases still parallelise.

constexpr unsigned BCAST_TILECAP = 4096;  // max A elements streamed per Vector block
// Output GM is float; different Vector cores must not write the same cache line or
// concurrent stores clobber each other (lost rows). Lane row-spans are aligned so each
// starts on a 512-byte boundary: BCAST_LINE_ELEMS float elements per cache line.
constexpr unsigned BCAST_LINE_ELEMS = 2048;  // 512 bytes / sizeof(float)

// Mixed-radix decode of broadcast-operand base offset for output group g (OUTER axes
// outermost-first). outer_rank==0 (single group) returns 0.
template <typename CONFIG_T>
AICORE inline unsigned bcast_baseB(unsigned g) {
    unsigned base = 0;
    for (int a = (int)CONFIG_T::outer_rank - 1; a >= 0; --a) {
        unsigned d = CONFIG_T::outer_dims[a];
        base += (g % d) * CONFIG_T::outer_bstride[a];
        g /= d;
    }
    return base;
}

// One ColExpand block: dst[0:rb, 0:CC] = a[0:rb, 0:CC] * b[0:CC] (b reused across rows).
template <typename data_T, unsigned RB, unsigned CC>
AICORE inline void bcast_col_block(__gm__ const data_T* a, __gm__ const data_T* b,
                                   __gm__ float* res, unsigned rb) {
    // The row count is a runtime extent (rb <= RB): a lane's trailing block has fewer
    // than RB rows, so a static RB-row GM shape would make TLOAD/TSTORE over-read /
    // over-write past the operand buffer (OOB -> NaN at the last lane). Use a dynamic
    // (-1) row dim set to rb, exactly as the matmul does with validM.
    using SA = pto::Shape<1, 1, 1, -1, CC>;
    using GA = GlobalTensor<data_T, SA, pto::Stride<1, 1, 1, CC, 1>>;
    using GB = GlobalTensor<data_T, pto::Shape<1, 1, 1, 1, CC>,  pto::Stride<1, 1, 1, CC, 1>>;
    using GR = GlobalTensor<float,  SA, pto::Stride<1, 1, 1, CC, 1>>;
    using TA  = Tile<TileType::Vec, data_T, RB, CC, BLayout::RowMajor, -1, -1>;
    using TB  = Tile<TileType::Vec, data_T, 1,  CC, BLayout::RowMajor, -1, -1>;
    using TF  = Tile<TileType::Vec, float,  RB, CC, BLayout::RowMajor, -1, -1>;
    using TBF = Tile<TileType::Vec, float,  1,  CC, BLayout::RowMajor, -1, -1>;

    TA ta; TB tb; TF dst; TBF bf; TF af;
    TASSIGN(ta,  0x0);
    TASSIGN(tb,  sizeof(data_T) * RB * CC);
    TASSIGN(dst, sizeof(data_T) * RB * CC + sizeof(data_T) * CC);
    TASSIGN(af,  sizeof(data_T) * RB * CC + sizeof(data_T) * CC + sizeof(float) * RB * CC);
    TASSIGN(bf,  sizeof(data_T) * RB * CC + sizeof(data_T) * CC + sizeof(float) * RB * CC * 2);
    ta.SetValidRow(rb); ta.SetValidCol(CC);
    tb.SetValidRow(1);  tb.SetValidCol(CC);
    dst.SetValidRow(rb); dst.SetValidCol(CC);
    af.SetValidRow(rb); af.SetValidCol(CC);
    bf.SetValidRow(1);  bf.SetValidCol(CC);

    GA ga(const_cast<__gm__ data_T*>(a), SA(rb));
    GB gb(const_cast<__gm__ data_T*>(b));
    GR gr(res, SA(rb));

    TLOAD(ta, ga);
    TLOAD(tb, gb);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);   // RAW: compute waits its loads
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if constexpr (sizeof(data_T) == 2) {
        TCVT(af, ta, RoundMode::CAST_RINT);
        TCVT(bf, tb, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);                 // RAW: expand-mul reads the just-converted af/bf
        TCOLEXPANDMUL(dst, af, bf);
    } else {
        TCOLEXPANDMUL(dst, ta, tb);           // fp32: data_T == float
    }

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);   // RAW: store waits the compute
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gr, dst);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);  // WAR: next loads reuse the tiles
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

template <typename data_T, typename CONFIG_T>
AICORE inline void bcast_col_inline(__gm__ const data_T* a, __gm__ const data_T* b,
                                    __gm__ float* res, unsigned lane, unsigned nlanes) {
    constexpr unsigned CC = CONFIG_T::Cc;
    constexpr unsigned Rr = CONFIG_T::Rr;
    constexpr unsigned RB = (BCAST_TILECAP / CC) ? (BCAST_TILECAP / CC) : 1;  // rows per block
    constexpr unsigned total_rows = CONFIG_T::Outer * Rr;
    // Flatten the (group, row) space and split it across lanes so OUTER==1 (single big
    // group, e.g. rms_norm) still uses every core. A block never crosses a group boundary
    // (one B vector per group) nor the lane's span. Lane boundaries are rounded up to
    // ALIGN_ROWS so every lane's first row starts on a 512-byte cache line -- otherwise
    // adjacent cores writing sub-line (e.g. 256-byte) row spans clobber each other.
    // ALIGN_ROWS = LINE / gcd(CC, LINE); LINE is a power of two, so gcd is CC's low bit
    // (the largest power of two dividing CC), capped at LINE.
    constexpr unsigned CC_LOWBIT = CC & (~CC + 1u);
    constexpr unsigned ALIGN_ROWS =
        BCAST_LINE_ELEMS / (CC_LOWBIT < BCAST_LINE_ELEMS ? CC_LOWBIT : BCAST_LINE_ELEMS);
    unsigned chunk = (total_rows + nlanes - 1) / nlanes;
    chunk = (chunk + ALIGN_ROWS - 1) / ALIGN_ROWS * ALIGN_ROWS;
    unsigned start = lane * chunk;
    unsigned end = (start + chunk) < total_rows ? (start + chunk) : total_rows;
    unsigned r = start;
    while (r < end) {
        unsigned group = r / Rr;
        unsigned row_in_group = r - group * Rr;
        unsigned rb = RB;
        if (rb > Rr - row_in_group) rb = Rr - row_in_group;
        if (rb > end - r) rb = end - r;
        unsigned baseB = bcast_baseB<CONFIG_T>(group);
        bcast_col_block<data_T, RB, CC>(a + (size_t)r * CC, b + baseB, res + (size_t)r * CC, rb);
        r += rb;
    }
}

// One scalar block: res[0:valid] = a[0:valid] * scalar. BBYTES is the UB byte offset
// past the persistent broadcast-operand tile (which the caller holds at offset 0).
template <typename data_T, unsigned CAP, unsigned BBYTES>
AICORE inline void bcast_scalar_block(__gm__ const data_T* a, float scalar, __gm__ float* res,
                                      unsigned valid) {
    using GIn  = GlobalTensor<data_T, pto::Shape<1, 1, 1, 1, CAP>, pto::Stride<1, 1, 1, CAP, 1>>;
    using GOut = GlobalTensor<float,  pto::Shape<1, 1, 1, 1, CAP>, pto::Stride<1, 1, 1, CAP, 1>>;
    using TIn = Tile<TileType::Vec, data_T, 1, CAP, BLayout::RowMajor, -1, -1>;
    using TF  = Tile<TileType::Vec, float,  1, CAP, BLayout::RowMajor, -1, -1>;
    TIn ta; TF dst;
    TASSIGN(ta,  BBYTES);
    TASSIGN(dst, BBYTES + sizeof(data_T) * CAP);
    ta.SetValidRow(1);  ta.SetValidCol(valid);
    dst.SetValidRow(1); dst.SetValidCol(valid);
    GIn ga(const_cast<__gm__ data_T*>(a));
    GOut gr(res);
    TLOAD(ta, ga);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (sizeof(data_T) == 2) {
        TF af; TASSIGN(af, BBYTES + sizeof(data_T) * CAP + sizeof(float) * CAP);
        af.SetValidRow(1); af.SetValidCol(valid);
        TCVT(af, ta, RoundMode::CAST_RINT);
        pipe_barrier(PIPE_V);
        TMULS(dst, af, scalar);
    } else {
        TMULS(dst, ta, scalar);
    }
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(gr, dst);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

template <typename data_T, typename CONFIG_T>
AICORE inline void bcast_scalar_inline(__gm__ const data_T* a, __gm__ const data_T* b,
                                       __gm__ float* res, unsigned lane, unsigned nlanes) {
    constexpr unsigned Inner = CONFIG_T::Inner;
    constexpr unsigned TILE = BCAST_TILECAP;
    constexpr unsigned sizeB = CONFIG_T::sizeB;
    constexpr unsigned BLK = 32 / sizeof(data_T);
    constexpr unsigned sizeB_pad = (sizeB + BLK - 1) / BLK * BLK;  // 32B-aligned tile cols
    // Load the (small) broadcast operand once into UB at offset 0, then read each group's
    // scalar with GetValue. The per-block streaming tiles live above it (see CONFIG_BCAST_B_BYTES).
    using GB = GlobalTensor<data_T, pto::Shape<1, 1, 1, 1, sizeB_pad>, pto::Stride<1, 1, 1, sizeB_pad, 1>>;
    using TB = Tile<TileType::Vec, data_T, 1, sizeB_pad, BLayout::RowMajor, -1, -1>;
    TB bt; TASSIGN(bt, 0x0);
    bt.SetValidRow(1); bt.SetValidCol(sizeB);
    GB gb(const_cast<__gm__ data_T*>(b));
    TLOAD(bt, gb);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID2);   // B-vector load -> scalar reads
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID2);

    constexpr unsigned cpg = (Inner + TILE - 1) / TILE;  // chunks per group
    constexpr unsigned total = CONFIG_T::Outer * cpg;
    unsigned chunk = (total + nlanes - 1) / nlanes;
    unsigned start = lane * chunk;
    unsigned end = (start + chunk) < total ? (start + chunk) : total;
    for (unsigned u = start; u < end; ++u) {
        unsigned group = u / cpg;
        unsigned c = u - group * cpg;
        unsigned off = c * TILE;
        unsigned valid = (off + TILE <= Inner) ? TILE : (Inner - off);
        unsigned baseB = bcast_baseB<CONFIG_T>(group);
        float scalar = (float)bt.GetValue(baseB);
        bcast_scalar_block<data_T, TILE, sizeof(data_T) * sizeB_pad>(
            a + (size_t)group * Inner + off, scalar, res + (size_t)group * Inner + off, valid);
    }
}

template <typename data_T, typename CONFIG_T>
__global__ AICORE void broadcast_mul_kernel_standalone(
        __gm__ const data_T* a, __gm__ const data_T* b, __gm__ float* res) {
    if constexpr (DAV_VEC) {
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        if constexpr (CONFIG_T::mode == 0)
            bcast_col_inline<data_T, CONFIG_T>(a, b, res, get_block_idx(), get_block_num());
        else
            bcast_scalar_inline<data_T, CONFIG_T>(a, b, res, get_block_idx(), get_block_num());
        // Drain the final MTE3 store before the core retires. Each block's WAR flag
        // (MTE3->MTE2) only orders a store ahead of the *next* load, so the last block's
        // store is never awaited. For small problems only core 0 is active, so there is
        // no cross-core barrier to drain it implicitly (as the multi-core paths get); the
        // store can then still be in flight when the host reads back the output, yielding
        // stale data at reused GM addresses. This explicit drain closes that race.
        pipe_barrier(PIPE_ALL);
    }
}

// One-shot launcher (synchronous), the broadcast analogue of einsum().
template <typename data_T, typename CONFIG_T>
void broadcast_mul(const data_T* a, const data_T* b, float* res, void* stream = nullptr) {
    int64_t core_num = cached_core_num();
    broadcast_mul_kernel_standalone<data_T, CONFIG_T><<<core_num, nullptr, stream>>>(a, b, res);
    aclrtSynchronizeStream(stream);
}

// Persistent-runner ABI parity (no workspace; mirrors elementwise_setup/exec).
inline void* broadcast_setup(void* /*stream*/ = nullptr) {
    void* workspace = nullptr;
    aclrtMalloc(&workspace, sizeof(float), ACL_MEM_MALLOC_NORMAL_ONLY);
    return workspace;
}

template <typename data_T, typename CONFIG_T>
void broadcast_exec(const data_T* a, const data_T* b, float* res,
                    void* /*workspace*/, void* stream = nullptr) {
    int64_t core_num = cached_core_num();
    broadcast_mul_kernel_standalone<data_T, CONFIG_T><<<core_num, nullptr, stream>>>(a, b, res);
}

// ─── Fused single-kernel einsum ──────────────────────────────────────────────
// One mix-kernel launch (AIC + AIV) runs the whole pipeline, with full
// cross-core barriers between phases:
//   [Vec ]  Phase A: transpose data0->ws0, data1->ws1
//   -- SyncAll<false>() --   Vec -> Cube
//   [Cube]  Phase B: ws0 @ ws1 -> ws_res
//   -- SyncAll<false>() --   Cube -> Vec
//   [Vec ]  Phase C: transpose ws_res->res
// Vec distributes work over block_num*2 sub-block lanes; Cube over block_num.
//
// When the output permutation is identity (ij,jk->ik, bij,bjk->bik, … — the common
// case) Phase C is a plain copy, so the Cube writes straight to res and the second
// barrier + Phase C are dropped entirely (and ws_res is not even allocated).
// Identity-input fast path — the input-side mirror of the identity-output skip above.
// When an input's transpose is the identity AND the contraction needs no padding
// (K==C), its workspace copy is byte-identical to the raw tensor, so Phase A's copy is
// pure overhead (it dominates the fused kernel on plain matmuls); the matmul then reads
// data0/data1 directly. A struct of static constexpr flags — not a constexpr function,
// which would default to host-only and be uncallable from the __global__ kernel — so
// host (setup/exec, workspace layout) and device (kernel, gating) always agree.
template <typename CONFIG_T>
struct EinsumSkip {
    static constexpr unsigned C = CONFIG_T::n_contract;
    static constexpr unsigned K = (C + 15) / 16 * 16;
    // A partial free dim needs the K-padded *and* tile-padded workspace buffer (the
    // matmul loads full tiles from the zero-filled tail), so its copy is no longer a
    // byte-identical view of the raw tensor -> cannot skip.
    static constexpr bool inp0 = IsIdentityPerm<typename CONFIG_T::tpose_inp0_config>::value
                                 && (K == C) && !MatmulGeom<CONFIG_T>::A_PADDED;
    static constexpr bool inp1 = IsIdentityPerm<typename CONFIG_T::tpose_inp1_config>::value
                                 && (K == C) && !MatmulGeom<CONFIG_T>::B_PADDED;
};

template <typename data_T, typename CONFIG_T>
__global__ AICORE void einsum_fused_kernel(
        __gm__ const data_T* data0, __gm__ const data_T* data1,
        __gm__ data_T* ws0, __gm__ data_T* ws1, __gm__ float* ws_res,
        __gm__ float* res, uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);

    constexpr bool OUT_IDENTITY = IsIdentityPerm<typename CONFIG_T::tpose_out_conf>::value;
    // Inputs whose copy into the workspace is redundant (identity perm + K==C) are
    // read straight from data0/data1; their Phase A transpose is skipped below.
    constexpr bool SKIP0 = EinsumSkip<CONFIG_T>::inp0;
    constexpr bool SKIP1 = EinsumSkip<CONFIG_T>::inp1;
    // Split-K is enabled only when the output is identity (so the matmul target is
    // `res`, which the host pre-zeros for these configs — see builder._splitk so the
    // cores can atomic-add into it), there is K to split, and the output-tile grid
    // is small enough to leave cores idle on the plain schedule. The runtime
    // ksplit (>=2) gate inside collapses it to the plain path on small parts.
    constexpr bool SPLITK_ELIGIBLE =
        OUT_IDENTITY && (MatmulGeom<CONFIG_T>::nK >= 2) && (MatmulGeom<CONFIG_T>::total_tiles < 16);

    if constexpr (DAV_VEC) {

        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        constexpr unsigned C = CONFIG_T::n_contract;
        constexpr unsigned K = (C + 15) / 16 * 16;
        constexpr unsigned I = CONFIG_T::n_inplace;
        constexpr unsigned L1 = CONFIG_T::n_free1;
        unsigned lane = get_block_idx() * 2 + get_subblockid();
        unsigned nlanes = get_block_num() * 2;

        // input0's transpose output is [.., contract]; pad its innermost contract
        // dim C up to the fractal width K so the Cube reads a full K-wide A. The
        // pad is uniform across all I*L0 rows, so any batch count works directly.
        // SKIP0: ws0 would be a byte-identical copy of data0 — read data0 directly.
        if constexpr (!SKIP0)
            transpose_inline<data_T, typename CONFIG_T::tpose_inp0_config, C, K>(data0, ws0, lane, nlanes);

        // input1's contract dim is the per-batch row count. For K==C, or a single
        // batch, the K-C tail sits after the (one) valid block, so a plain
        // transpose into the pre-zeroed ws1 suffices. For K!=C with batching, each
        // batch's C valid rows must land at the front of its K-row slot (stride
        // K*L1) — a contiguous write would mis-stride the batches. An identity input1
        // is the contiguous repack (batched_pad_copy_inline); a non-identity input1
        // transposes *and* row-pads (transpose_inline_rowpad). SKIP1: ws1 == data1.
        if constexpr (!SKIP1) {
            if constexpr (K != C && I > 1) {
                if constexpr (IsIdentityPerm<typename CONFIG_T::tpose_inp1_config>::value) {
                    batched_pad_copy_inline<data_T>(data1, ws1, I, C, L1, K * L1, lane, nlanes);
                } else {
                    transpose_inline_rowpad<data_T, typename CONFIG_T::tpose_inp1_config, C, K, L1>(
                        data1, ws1, lane, nlanes);
                }
            } else if constexpr (MatmulGeom<CONFIG_T>::B_PADDED) {
                // Partial N (requires I==1): pad the innermost free dim L1 up to the
                // tile-aligned Na so the matmul reads a full Nt-col tile from the
                // zero-filled tail. (For K!=C the row tail still sits in the pre-zeroed
                // rows C..K of the K-tall ws1 slot.)
                transpose_inline<data_T, typename CONFIG_T::tpose_inp1_config, L1,
                                 MatmulGeom<CONFIG_T>::Na>(data1, ws1, lane, nlanes);
            } else {
                transpose_inline<data_T, typename CONFIG_T::tpose_inp1_config>(data1, ws1, lane, nlanes);
            }
        }
    }

    SyncAll<false>();  // inputs transposed -> matmul

    if constexpr (DAV_CUBE) {
        // Identity output: the matmul's natural layout already is res, so write it
        // directly and skip the whole output-transpose pass below. SPLITK_ELIGIBLE
        // configs (identity, thin grid) hand idle cores K-slices that atomic-add
        // into the pre-zeroed res. The matmul reads each input from the workspace, or
        // straight from the raw tensor when its copy was skipped (SKIP0/SKIP1).
        __gm__ float* mm_out = OUT_IDENTITY ? res : ws_res;
        __gm__ const data_T* mmA = SKIP0 ? data0 : static_cast<__gm__ const data_T*>(ws0);
        __gm__ const data_T* mmB = SKIP1 ? data1 : static_cast<__gm__ const data_T*>(ws1);
        batched_matmul_inline<data_T, CONFIG_T, SPLITK_ELIGIBLE>(mmA, mmB, mm_out, get_block_idx(), get_block_num());
    }

    if constexpr (!OUT_IDENTITY) {
        SyncAll<false>();  // matmul done -> output transpose

        if constexpr (DAV_VEC) {
            unsigned lane = get_block_idx() * 2 + get_subblockid();
            unsigned nlanes = get_block_num() * 2;
            transpose_inline<float, typename CONFIG_T::tpose_out_conf>(ws_res, res, lane, nlanes);
        }
    }
}

// The whole transpose -> matmul -> transpose pipeline runs as a single fused
// mix-kernel launch (einsum_fused_kernel). Covers every supported config: K==C
// (any batch) and K!=C (any batch, with the input transposes writing the K-padded
// layout). The workspace is K-padded so the Cube reads a full K-wide A / K-row B,
// and only the pad regions need zeroing.
// The fused dispatch is split so a reused runner pays the per-call cost only for
// the launch. einsum_setup allocates the K-padded workspace once and zeros its
// contraction-pad regions once; einsum_exec just launches; einsum_teardown frees.
// This hoists aclrtMalloc/Memset/Free and the host-device sync out of the hot path
// (they dominate these sub-0.1 ms kernels). einsum() keeps the one-shot all-in-one.

// Workspace element counts: [ws0 (I*Arows*K) | ws1 (I*K*Bcols) | ws_res (out, float)].
// Arows/Bcols are the K-padded free extents, themselves padded up to a whole tile when
// the dim is partial (A_PADDED/B_PADDED) so the matmul can load full tiles.
template <typename data_T, typename CONFIG_T>
inline size_t einsum_ws0_elems() {
    using G = MatmulGeom<CONFIG_T>;
    return size_t(G::I) * G::Arows * G::K;
}
template <typename data_T, typename CONFIG_T>
inline size_t einsum_ws1_elems() {
    using G = MatmulGeom<CONFIG_T>;
    return size_t(G::I) * G::K * G::Bcols;
}

// Allocate the workspace once and zero the pad regions once. The per-call
// transposes only overwrite the data columns/rows (cols/rows K..C stay 0) and the
// matmul never writes ws0/ws1, so the zeroing holds across calls. Returns the base.
template <typename data_T, typename CONFIG_T>
void* einsum_setup(void* stream = nullptr) {
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned K = (C + 15) / 16 * 16;
    // Skipped inputs (identity perm + K==C) are read straight from data0/data1, so
    // their workspace region is neither allocated nor copied into. Note K!=C forces
    // both SKIP flags false, so the pad-zeroing path below always has its buffers.
    constexpr bool SKIP0 = EinsumSkip<CONFIG_T>::inp0;
    constexpr bool SKIP1 = EinsumSkip<CONFIG_T>::inp1;
    const size_t ws0_elems = SKIP0 ? 0 : einsum_ws0_elems<data_T, CONFIG_T>();
    const size_t ws1_elems = SKIP1 ? 0 : einsum_ws1_elems<data_T, CONFIG_T>();
    const size_t ws0_bytes = sizeof(data_T) * ws0_elems;
    const size_t ws1_bytes = sizeof(data_T) * ws1_elems;
    // Identity output writes straight to res, so ws_res is never read — don't alloc it.
    constexpr bool OUT_IDENTITY = IsIdentityPerm<typename CONFIG_T::tpose_out_conf>::value;
    const size_t wsr_bytes = OUT_IDENTITY ? 0 : sizeof(float) * CONFIG_T::tpose_out_conf::N;

    // A pure plain matmul (both inputs skipped, identity output) needs no workspace at
    // all; allocate a token byte so the runner caches a non-null handle and does not
    // re-run setup every call.
    size_t total = ws0_bytes + ws1_bytes + wsr_bytes;
    if (total == 0) total = sizeof(data_T);

    void* workspace = nullptr;
    aclrtMalloc(&workspace, total, ACL_MEM_MALLOC_NORMAL_ONLY);

    // Zero the pad regions once: the contraction pad (K!=C) for both, plus the
    // tile-row/col pad (A_PADDED/B_PADDED) the matmul reads as a full-tile zero tail.
    // The per-call transposes only overwrite the data region, so the zeroing holds.
    constexpr bool ZERO0 = (K != C) || MatmulGeom<CONFIG_T>::A_PADDED;
    constexpr bool ZERO1 = (K != C) || MatmulGeom<CONFIG_T>::B_PADDED;
    if constexpr (ZERO0 || ZERO1) {
        data_T* ws0 = reinterpret_cast<data_T*>(workspace);
        data_T* ws1 = reinterpret_cast<data_T*>(ws0 + ws0_elems);
        if constexpr (ZERO0) aclrtMemsetAsync(ws0, ws0_bytes, 0, ws0_bytes, stream);
        if constexpr (ZERO1) aclrtMemsetAsync(ws1, ws1_bytes, 0, ws1_bytes, stream);
        aclrtSynchronizeStream(stream);  // pad must be zero before the first exec
    }
    return workspace;
}

// Launch the fused kernel against a pre-allocated workspace. No alloc/memset/sync/
// free: a reused runner enqueues on one stream, which orders the calls (and the
// output's downstream torch consumers), exactly as torch.einsum is itself async.
template <typename data_T, typename CONFIG_T>
void einsum_exec(const data_T* data0, const data_T* data1, float* res,
                 void* workspace, void* stream = nullptr) {
    int64_t core_num = cached_core_num();

    // Match einsum_setup's layout: skipped inputs occupy no workspace region.
    constexpr bool SKIP0 = EinsumSkip<CONFIG_T>::inp0;
    constexpr bool SKIP1 = EinsumSkip<CONFIG_T>::inp1;
    const size_t ws0_elems = SKIP0 ? 0 : einsum_ws0_elems<data_T, CONFIG_T>();
    const size_t ws1_elems = SKIP1 ? 0 : einsum_ws1_elems<data_T, CONFIG_T>();

    data_T* ws0 = reinterpret_cast<data_T*>(workspace);
    data_T* ws1 = reinterpret_cast<data_T*>(ws0 + ws0_elems);
    float* ws_res = reinterpret_cast<float*>(ws1 + ws1_elems);

    uint32_t ffts_len = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);

    einsum_fused_kernel<data_T, CONFIG_T><<<core_num, nullptr, stream>>>(
        data0, data1, ws0, ws1, ws_res, res, ffts_addr);
}

inline void einsum_teardown(void* workspace) {
    if (workspace) aclrtFree(workspace);
}

// One-shot all-in-one (setup + exec + sync + teardown). Used by the CPU-symmetric
// run_einsum entry and any caller that does not reuse a workspace.
template <typename data_T, typename CONFIG_T>
void einsum(const data_T data0[CONFIG_T::tpose_inp0_config::N], const data_T data1[CONFIG_T::tpose_inp1_config::N],
            float res[CONFIG_T::tpose_out_conf::N], void* stream = nullptr) {
    void* workspace = einsum_setup<data_T, CONFIG_T>(stream);
    einsum_exec<data_T, CONFIG_T>(data0, data1, res, workspace, stream);
    aclrtSynchronizeStream(stream);
    einsum_teardown(workspace);
}

} // namespace pto_einsum
