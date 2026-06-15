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

// Identity copy: TLOAD src → TSTORE dst as flat 1D chunks through UB (src and
// dst share the same layout). Caller distributes the [start, end) range.
template <typename data_T, typename CONFIG_T>
AICORE inline void transpose_copy_inline(__gm__ const data_T* src, __gm__ data_T* dst, unsigned start, unsigned end) {
    constexpr unsigned TILE_COLS = 128;  // elems/row; 32-byte aligned for fp16
    constexpr unsigned TILE_ROWS = 1;

    using TileShape = pto::Shape<1, 1, 1, TILE_ROWS, TILE_COLS>;
    using TileStride = pto::Stride<1, 1, 1, TILE_COLS, 1>;
    using GlobalData = GlobalTensor<data_T, TileShape, TileStride>;
    using TileData = Tile<TileType::Vec, data_T, TILE_ROWS, TILE_COLS, BLayout::RowMajor, -1, -1>;

    TileData ubTile(TILE_ROWS, TILE_COLS);
    TASSIGN(ubTile, 0x0);

    for (unsigned i = start; i < end; i += TILE_COLS) {
        unsigned chunk = ((i + TILE_COLS) <= end) ? TILE_COLS : (end - i);
        ubTile.SetValidCol(chunk);
        ubTile.SetValidRow(1);

        GlobalData srcGlobal(const_cast<__gm__ data_T*>(src + i));
        GlobalData dstGlobal(dst + i);

        TLOAD(ubTile, srcGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

        TSTORE(dstGlobal, ubTile);
        pipe_barrier(PIPE_ALL);
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
    using TileShape = pto::Shape<1, 1, 1, 1, TILE_COLS>;
    using TileStride = pto::Stride<1, 1, 1, TILE_COLS, 1>;
    using GlobalData = GlobalTensor<data_T, TileShape, TileStride>;
    using TileData = Tile<TileType::Vec, data_T, 1, TILE_COLS, BLayout::RowMajor, -1, -1>;

    TileData ubTile(1, TILE_COLS);
    TASSIGN(ubTile, 0x0);

    unsigned chunk = (nrows + nlanes - 1) / nlanes;
    unsigned r0 = lane * chunk;
    unsigned r1 = (r0 + chunk) < nrows ? (r0 + chunk) : nrows;

    for (unsigned r = r0; r < r1; r++) {
        for (unsigned c = 0; c < rowW; c += TILE_COLS) {
            unsigned w = (c + TILE_COLS) <= rowW ? TILE_COLS : (rowW - c);
            ubTile.SetValidRow(1);
            ubTile.SetValidCol(w);
            GlobalData sg(const_cast<__gm__ data_T*>(src + r * rowW + c));
            GlobalData dg(dst + r * dstStride + c);
            TLOAD(ubTile, sg);

            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

            TSTORE(dg, ubTile);
            pipe_barrier(PIPE_ALL);
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
    using TileShape = pto::Shape<1, 1, 1, 1, TILE_COLS>;
    using TileStride = pto::Stride<1, 1, 1, TILE_COLS, 1>;
    using GlobalData = GlobalTensor<data_T, TileShape, TileStride>;
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
            ubTile.SetValidRow(1);
            ubTile.SetValidCol(w);
            GlobalData sg(const_cast<__gm__ data_T*>(src + g * rowW + col));
            GlobalData dg(dst + i * dstBatchStride + c * rowW + col);
            TLOAD(ubTile, sg);

            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);

            TSTORE(dg, ubTile);
            pipe_barrier(PIPE_ALL);
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

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TTRANS(dstTile, srcTile, tmpTile);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        TSTORE(dstGlobal, dstTile);
        pipe_barrier(PIPE_ALL);
    } else {
        // Too large for UB: TTRANS one BR x BC source block at a time, each a single
        // strided 2D load -> HW transpose -> strided 2D store. Block GM rows start at
        // r*srcCols / c*srcRows, so 16-aligned dims keep every transfer 32-byte
        // aligned (a GM-DMA constraint, not a TTRANS one). Unaligned large tensors
        // are not yet supported.
        static_assert(srcRows % 16 == 0 && srcCols % 16 == 0,
                      "Blocked transpose requires 16-aligned dims for large tensors.");
        constexpr unsigned BR = 64;
        constexpr unsigned BC = 64;
        constexpr unsigned subSrcTW = align_up(BC, blk);   // src sub-tile width  (RowStride)
        constexpr unsigned subDstTW = align_up(BR, 16u);   // dst/tmp sub-tile width (RowStride)

        using SrcShape = pto::Shape<1, 1, 1, BR, BC>;
        using SrcStride = pto::Stride<1, 1, 1, srcCols, 1>;
        using DstShape = pto::Shape<1, 1, 1, BC, BR>;
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
            SrcGlobal sg(const_cast<__gm__ data_T*>(src + r0 * srcCols + c0));
            DstGlobal dg(dst + c0 * dpad + r0);

            TLOAD(srcTile, sg);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TTRANS(dstTile, srcTile, tmpTile);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

            TSTORE(dg, dstTile);
            pipe_barrier(PIPE_ALL);  // done with this block's UB before next load
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
// Factored out so the standalone kernel and the fused kernel share it; the caller
// guards with `if constexpr (DAV_CUBE)` and supplies the block index/count.
template <typename data_T, typename CONFIG_T>
AICORE inline void batched_matmul_inline(__gm__ const data_T* ws0, __gm__ const data_T* ws1,
                                         __gm__ float* ws_res, unsigned block_idx, unsigned block_num) {
        constexpr unsigned L0 = CONFIG_T::n_free0;
        constexpr unsigned L1 = CONFIG_T::n_free1;
        constexpr unsigned C = CONFIG_T::n_contract;
        constexpr unsigned I = CONFIG_T::n_inplace;

        // Padded (fractal-aligned) problem dims.
        constexpr unsigned M = (L0 + 15) / 16 * 16;
        constexpr unsigned K = (C + 15) / 16 * 16;
        constexpr unsigned N = (L1 + 15) / 16 * 16;

        // Tile sizes, clamped to the padded dim (config sizes are multiples of 16).
        constexpr unsigned Mt = CONFIG_T::tile_m < M ? CONFIG_T::tile_m : M;
        constexpr unsigned Nt = CONFIG_T::tile_n < N ? CONFIG_T::tile_n : N;
        constexpr unsigned Kt = CONFIG_T::tile_k < K ? CONFIG_T::tile_k : K;

        // Full tile grid only (no partial tiles).
        static_assert(M % Mt == 0 && N % Nt == 0 && K % Kt == 0,
                      "Tiling requires padded M/N/K divisible by the (clamped) tile size.");

        constexpr unsigned nM = M / Mt;
        constexpr unsigned nN = N / Nt;
        constexpr unsigned nK = K / Kt;
        constexpr unsigned tiles_per_batch = nM * nN;
        constexpr unsigned total_tiles = I * tiles_per_batch;

        // Per-tile GM views. The ND->NZ loader takes its row/column transfer counts
        // (nValue/dValue) directly from the GM Shape (not the tile's valid extents),
        // so the Shape must describe the *real* data window, not the padded tile:
        //   A: validM x Kt   (cols Kt are within the host-padded K, so always backed)
        //   B: Kt     x validN
        //   C: validM x validN
        // The boundary row/col counts are runtime, hence DYNAMIC dims passed at
        // construction. Row strides stay the full-matrix width (K / L1 / L1). Using
        // the padded Mt/Nt here would over-read past the physical L0/L1 extent.
        using ShapeA = pto::Shape<1, 1, 1, -1, Kt>;
        using StrideA = pto::Stride<1, 1, 1, K, 1>;
        using ShapeB = pto::Shape<1, 1, 1, Kt, -1>;
        using StrideB = pto::Stride<1, 1, 1, L1, 1>;
        using ShapeC = pto::Shape<1, 1, 1, -1, -1>;
        using StrideC = pto::Stride<1, 1, 1, L1, 1>;

        // On-chip tiles: fixed tile dims, DYNAMIC valid extents set per output tile.
        // ColMajor == NZ fractal; SLayout::RowMajor because the GM source is row-major.
        using MatTileA = pto::Tile<pto::TileType::Mat, data_T, Mt, Kt, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, 512>;
        using MatTileB = pto::Tile<pto::TileType::Mat, data_T, Kt, Nt, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, 512>;
        using LeftTileA = pto::TileLeft<data_T, Mt, Kt, -1, -1>;
        using RightTileB = pto::TileRight<data_T, Kt, Nt, -1, -1>;
        using AccTileC = pto::TileAcc<float, Mt, Nt, -1, -1>;

        unsigned chunk = (total_tiles + block_num - 1) / block_num;
        unsigned start = block_idx * chunk;
        unsigned end = (start + chunk) < total_tiles ? (start + chunk) : total_tiles;

        for (unsigned t = start; t < end; t++) {
            unsigned i = t / tiles_per_batch;
            unsigned rem = t % tiles_per_batch;
            unsigned row0 = (rem / nN) * Mt;  // output row offset within the matrix
            unsigned col0 = (rem % nN) * Nt;  // output col offset within the matrix

            // Boundary tiles see fewer real rows/cols than the padded tile (GM is
            // L0xK / KxL1). row0 < L0 and col0 < L1 always hold for a full grid.
            unsigned validM = (L0 - row0) < Mt ? (L0 - row0) : Mt;
            unsigned validN = (L1 - col0) < Nt ? (L1 - col0) : Nt;

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

            // Valid extents are constant across the K-loop, so set them once.
            matA[0].SetValidRow(validM); matA[0].SetValidCol(Kt);
            matA[1].SetValidRow(validM); matA[1].SetValidCol(Kt);
            matB[0].SetValidRow(Kt);     matB[0].SetValidCol(validN);
            matB[1].SetValidRow(Kt);     matB[1].SetValidCol(validN);
            leftA.SetValidRow(validM);   leftA.SetValidCol(Kt);
            rightB.SetValidRow(Kt);      rightB.SetValidCol(validN);
            accC.SetValidRow(validM);    accC.SetValidCol(validN);
            pipe_barrier(PIPE_ALL);  // PIPE_S: publish SetValid* before the tiles are used

            // Event IDs (per L1 buffer where two can be in flight):
            //   ID0/ID1  load-done   (MTE2 -> MTE1, RAW: mov waits its load)
            //   ID2/ID3  matA free   (MTE1 -> MTE2, WAR: load waits prior mov)
            //   ID4      mov-done    (MTE1 -> M,    RAW: matmul waits its mov)
            //   ID5      leftA free  (M    -> MTE1, WAR: mov waits prior matmul)
            for (unsigned k = 0; k < nK; k++) {
                unsigned p = k & 1;
                event_t e_ld  = p ? EVENT_ID1 : EVENT_ID0;
                event_t e_ldf = p ? EVENT_ID3 : EVENT_ID2;
                unsigned k0 = k * Kt;
                pto::GlobalTensor<data_T, ShapeA, StrideA> aGlobal(
                    const_cast<__gm__ data_T*>(ws0 + i * L0 * K + row0 * K + k0), ShapeA(validM));
                pto::GlobalTensor<data_T, ShapeB, StrideB> bGlobal(
                    const_cast<__gm__ data_T*>(ws1 + i * K * L1 + k0 * L1 + col0), ShapeB(validN));

                // Load GM -> L1 buffer p (wait until a prior mov freed this buffer).

                if (k >= 2) wait_flag(PIPE_MTE1, PIPE_MTE2, e_ldf);

                TLOAD(matA[p], aGlobal);
                TLOAD(matB[p], bGlobal);

                set_flag(PIPE_MTE2, PIPE_MTE1, e_ld);


                // Move L1 -> L0 (single buffer): wait this load, and the prior matmul.

                wait_flag(PIPE_MTE2, PIPE_MTE1, e_ld);
                if (k >= 1) wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);

                TMOV(leftA, matA[p]);
                TMOV(rightB, matB[p]);

                set_flag(PIPE_MTE1, PIPE_MTE2, e_ldf);    // matA[p] free to reload
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);   // leftA/rightB ready


                // Matmul: first Kt step initialises the accumulator, the rest accumulate.

                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID4);

                if (k == 0) {
                    TMATMUL(accC, leftA, rightB);
                } else {
                    TMATMUL_ACC(accC, leftA, rightB);
                }

                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);   // leftA/rightB free

            }

            // Drain the WAR-free flags the final step(s) set but nothing consumed,
            // so no flag state leaks into the next output tile.

            if constexpr (nK >= 2) {
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
            } else {
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
            }
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID5);

            pipe_barrier(PIPE_ALL);

            pto::GlobalTensor<float, ShapeC, StrideC> cGlobal(
                ws_res + i * L0 * L1 + row0 * L1 + col0, ShapeC(validM, validN));
            TSTORE(cGlobal, accC);
            pipe_barrier(PIPE_ALL);
        }
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
    int32_t device_id = 0;
    aclrtGetDevice(&device_id);
    int64_t core_num = 1;
    aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_AICORE_CORE_NUM, &core_num);

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

// Transpose dispatch body: identity -> tiled copy (chunked across lanes),
// 2D non-identity -> TTRANS (block grid distributed across lanes). `lane`/`nlanes`
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
    } else {
        static_assert(CONFIG_T::dims == 2, "Non-identity transpose only supported for 2D");
        transpose_2d_inline<data_T, CONFIG_T, PAD_STRIDE>(data, res, lane, nlanes);
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
    int32_t device_id = 0;
    aclrtGetDevice(&device_id);
    int64_t core_num = 1;
    aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_AICORE_CORE_NUM, &core_num);

    transpose_kernel_standalone<data_T, CONFIG_T><<<core_num, nullptr, stream>>>(data, res);
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
template <typename data_T, typename CONFIG_T>
__global__ AICORE void einsum_fused_kernel(
        __gm__ const data_T* data0, __gm__ const data_T* data1,
        __gm__ data_T* ws0, __gm__ data_T* ws1, __gm__ float* ws_res,
        __gm__ float* res, uint64_t ffts_addr) {
    set_ffts_base_addr(ffts_addr);

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
        transpose_inline<data_T, typename CONFIG_T::tpose_inp0_config, C, K>(data0, ws0, lane, nlanes);

        // input1's contract dim is the per-batch row count. For K==C, or a single
        // batch, the K-C tail sits after the (one) valid block, so a plain
        // transpose into the pre-zeroed ws1 suffices. For K!=C with batching, each
        // batch's C valid rows must land at the front of its K-row slot (stride
        // K*L1) — a contiguous write would mis-stride the batches — so repack with
        // a per-batch K-row pad.
        if constexpr (K != C && I > 1) {
            static_assert(IsIdentityPerm<typename CONFIG_T::tpose_inp1_config>::value,
                          "Batched K!=C fusion assumes an identity input1 permutation.");
            batched_pad_copy_inline<data_T>(data1, ws1, I, C, L1, K * L1, lane, nlanes);
        } else {
            transpose_inline<data_T, typename CONFIG_T::tpose_inp1_config>(data1, ws1, lane, nlanes);
        }
    }

    SyncAll<false>();  // inputs transposed -> matmul

    if constexpr (DAV_CUBE) {
        batched_matmul_inline<data_T, CONFIG_T>(ws0, ws1, ws_res, get_block_idx(), get_block_num());
    }

    SyncAll<false>();  // matmul done -> output transpose

    if constexpr (DAV_VEC) {
        unsigned lane = get_block_idx() * 2 + get_subblockid();
        unsigned nlanes = get_block_num() * 2;
        transpose_inline<float, typename CONFIG_T::tpose_out_conf>(ws_res, res, lane, nlanes);
    }
}

// Multi-launch einsum (4 separate kernels). No longer on the default dispatch
// path — the fused kernel now covers every supported config — but retained as a
// reference / debugging fallback. It host-pads the contraction via batched_matmul.
template <typename data_T, typename CONFIG_T>
void einsum_multilaunch(const data_T* data0, const data_T* data1, float* res, void* stream) {
    size_t ws_size = sizeof(data_T) * CONFIG_T::tpose_inp0_config::N +
                     sizeof(data_T) * CONFIG_T::tpose_inp1_config::N +
                     sizeof(float) * CONFIG_T::tpose_out_conf::N;

    void* workspace = nullptr;
    aclrtMalloc(&workspace, ws_size, ACL_MEM_MALLOC_NORMAL_ONLY);

    data_T* ws0 = reinterpret_cast<data_T*>(workspace);
    data_T* ws1 = reinterpret_cast<data_T*>(ws0 + CONFIG_T::tpose_inp0_config::N);
    float* ws_res = reinterpret_cast<float*>(ws1 + CONFIG_T::tpose_inp1_config::N);

    transpose<data_T, typename CONFIG_T::tpose_inp0_config>(data0, ws0, stream);
    transpose<data_T, typename CONFIG_T::tpose_inp1_config>(data1, ws1, stream);
    batched_matmul<data_T, CONFIG_T>(ws0, ws1, ws_res, stream);
    transpose<float, typename CONFIG_T::tpose_out_conf>(ws_res, res, stream);

    aclrtSynchronizeStream(stream);
    aclrtFree(workspace);
}

// The whole transpose -> matmul -> transpose pipeline runs as a single fused
// mix-kernel launch (einsum_fused_kernel). Covers every supported config: K==C
// (any batch) and K!=C (any batch, with the input transposes writing the K-padded
// layout). The workspace is K-padded so the Cube reads a full K-wide A / K-row B,
// and only the pad regions need zeroing.
template <typename data_T, typename CONFIG_T>
void einsum(const data_T data0[CONFIG_T::tpose_inp0_config::N], const data_T data1[CONFIG_T::tpose_inp1_config::N],
            float res[CONFIG_T::tpose_out_conf::N], void* stream = nullptr) {
    int32_t device_id = 0;
    aclrtGetDevice(&device_id);
    int64_t core_num = 1;
    aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_AICORE_CORE_NUM, &core_num);

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C  = CONFIG_T::n_contract;
    constexpr unsigned I  = CONFIG_T::n_inplace;
    constexpr unsigned K  = (C + 15) / 16 * 16;  // fractal-aligned contraction width

    // K-padded operand buffers (== the C-packed transpose output when K==C).
    const size_t ws0_elems = size_t(I) * L0 * K;
    const size_t ws1_elems = size_t(I) * K * L1;
    const size_t ws0_bytes = sizeof(data_T) * ws0_elems;
    const size_t ws1_bytes = sizeof(data_T) * ws1_elems;
    const size_t wsr_bytes = sizeof(float) * CONFIG_T::tpose_out_conf::N;

    void* workspace = nullptr;
    aclrtMalloc(&workspace, ws0_bytes + ws1_bytes + wsr_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);

    data_T* ws0 = reinterpret_cast<data_T*>(workspace);
    data_T* ws1 = reinterpret_cast<data_T*>(ws0 + ws0_elems);
    float* ws_res = reinterpret_cast<float*>(ws1 + ws1_elems);

    // Zero the contraction-pad regions (rows/cols K..C are not written by the
    // transposes and must contribute 0 to the matmul). Only needed when K != C.
    if constexpr (K != C) {
        aclrtMemsetAsync(ws0, ws0_bytes, 0, ws0_bytes, stream);
        aclrtMemsetAsync(ws1, ws1_bytes, 0, ws1_bytes, stream);
    }

    uint32_t ffts_len = 0;
    uint64_t ffts_addr = 0;
    rtGetC2cCtrlAddr(&ffts_addr, &ffts_len);

    einsum_fused_kernel<data_T, CONFIG_T><<<core_num, nullptr, stream>>>(
        data0, data1, ws0, ws1, ws_res, res, ffts_addr);

    aclrtSynchronizeStream(stream);
    aclrtFree(workspace);
}

} // namespace pto_einsum
