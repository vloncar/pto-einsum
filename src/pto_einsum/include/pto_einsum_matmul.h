#pragma once
#include "pto_einsum_utils.h"

namespace pto_einsum {

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
// sizes are min(config, padded-dim); a boundary tile whose padded dim is not a
// whole tile multiple is handled by padding the operand buffer up to a whole tile
// (Ma rows / Na cols, zero-filled — see MatmulGeom), so every loaded tile is full.
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

// Fused output permutation gate. When the einsum output's free0/free1 are each a
// single axis and free1 is the innermost output axis (res col stride 1), the Cube's
// fixpipe store can land each validM x validN tile straight into `res` at its permuted
// position — dropping Phase C, the second SyncAll and the ws_res buffer. The host
// emits the decode data (out_fusible, out_row_stride = res free0 stride, the inplace
// axes' sizes + res strides) into config_einsum; this struct surfaces it to the kernel
// (a struct of static constexpr, not a constexpr function, so device code can read it).
// Mutually exclusive with split-K, which stays gated on the identity-output path.
template <typename CONFIG_T>
struct FusibleOutputPerm {
    static constexpr bool value = (CONFIG_T::out_fusible != 0u);
    static constexpr unsigned row_stride = CONFIG_T::out_row_stride;  // res stride of free0
    static constexpr unsigned n_batch = CONFIG_T::out_n_batch;        // # inplace axes
};

// NT strided-input gate. When the host marks the config NT-eligible (contraction axis
// innermost+contiguous on both inputs, single free0/free1 axes — see builder.parse_recipe),
// the matmul reads both operands straight from the raw tensors in their natural strided
// layout, dropping Phase A: input0 as a row-strided ND tile (free0 row stride
// `row_stride0`, contraction contiguous), input1 transposed via a DN read (contraction
// contiguous == K innermost, free1 col stride `col_stride1`). The flat batch index decodes
// into each operand's source base via the inplace-axis sizes + per-operand source strides.
// Re-confirms the geometry preconditions the strided read needs (K==C so no contraction
// pad; full tiles so no zero-pad row/col to synthesize) so a host/device mismatch is a
// compile-time gate, never a silent misread. Composes with FusibleOutputPerm (output side).
template <typename CONFIG_T>
struct NtInput {
    using G = MatmulGeom<CONFIG_T>;
    static constexpr bool value = (CONFIG_T::in_nt != 0u)
                                  && (G::K == G::C) && !G::A_PADDED && !G::B_PADDED;
    // in_nt == 1 -> NT (B read transposed via Layout::DN); == 2 -> NN-strided (B read
    // as Layout::ND with a strided K row). Both read A as a row-strided ND tile.
    static constexpr bool dn = (CONFIG_T::in_nt == 1u);
    static constexpr unsigned n_batch = CONFIG_T::in_n_batch;
    static constexpr unsigned row_stride0 = CONFIG_T::in0_row_stride;  // src stride of free0 (input0)
    static constexpr unsigned col_stride1 = CONFIG_T::in1_col_stride;  // src stride of free1 (input1)
    static constexpr unsigned k_stride1 = CONFIG_T::in1_k_stride;      // src stride of contraction (input1, NN-strided)
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

// Cross-tile pipelining predicate (+ the granularities it keys on). The deep
// per-tile schedule (matmul_one_tile_deep) hides the big GM->L1 load behind compute
// only via its *within-tile* chunk prefetch (the `c+1 < nKd` arm). When nKd==1 —
// small K, e.g. the 128x128x128 GEMMs a batched attention / GDN contraction
// decomposes into — that prefetch never fires, so every tile's GM->L1 load (MTE2)
// and fixpipe store (FIX) run fully exposed and the cube idles (~15% of HBM,
// latency-bound, not bandwidth-bound). matmul_tile_loop_pipelined recovers this by
// overlapping ACROSS the tile boundary instead. This gates that path on:
//   * single-chunk tiles (nKd==1) — the only regime with no within-tile pipeline;
//   * all tiles full (!A_PADDED && !B_PADDED) — so the valid extents are
//     loop-invariant and can be set once (partial-tile nKd==1 keeps the per-tile
//     deep path; it is uncommon and not the batched-tiny target);
//   * the two ping-pong accumulators fit L0C (128 KB);
//   * enough tiles to amortize the prologue/epilogue.
template <typename data_T, typename CONFIG_T>
struct MatmulPipeline {
    using G = MatmulGeom<CONFIG_T>;
    static constexpr unsigned ds  = sizeof(data_T);
    static constexpr unsigned Kq  = deep_pick_kq(G::Mt, G::Nt, G::K, ds);
    static constexpr unsigned Kd  = deep_pick_kd(G::Mt, G::Nt, G::K, Kq, ds);
    static constexpr unsigned nKd = G::K / Kd;
    static constexpr unsigned MIN_TILES = 32u;
    // The L0C term below bounds the accumulator ping-pong; the matmul_tile_loop_pipelined
    // body's L1/L0A/L0B capacity static_asserts are implied by the same deep_pick_kq/kd
    // (Kq fits 64 KB L0, Kd fits 512 KB CBUF by construction), so this gate is a
    // sufficient precondition for the routed function -- a future granularity change that
    // broke an invariant would be a loud compile error there, never a silent miscompute.
    static constexpr bool PIPELINE_TILES =
        (nKd == 1u) && !G::A_PADDED && !G::B_PADDED &&
        (2u * G::Mt * G::Nt * sizeof(float) <= 128u * 1024u) &&
        (G::total_tiles >= MIN_TILES);
};

// Compile-time gate self-test (Phase 2 hardening). The cross-tile loop's win is
// invisible to a correctness check (the deep per-tile path is also correct), so a
// future change to deep_pick_kq/kd, MatmulGeom, or the predicate could silently stop
// routing the batched-tiny regime through it with no test failing. These static_asserts
// pin the gate's decision on canonical shapes so any such drift breaks the build loudly.
// Cost is nil: pure constexpr, evaluated once per translation unit, emits no code.
namespace gate_selftest {
template <unsigned F0, unsigned F1, unsigned Cn, unsigned I, unsigned T>
struct Cfg {
    static constexpr unsigned n_free0 = F0;
    static constexpr unsigned n_free1 = F1;
    static constexpr unsigned n_contract = Cn;
    static constexpr unsigned n_inplace = I;
    static constexpr unsigned tile_m = T;
    static constexpr unsigned tile_n = T;
    static constexpr unsigned tile_k = T;
};
// Target regime: full 128^3 tiles, many batches -> route to the cross-tile loop.
static_assert(MatmulPipeline<float, Cfg<128, 128, 128, 2048, 128>>::PIPELINE_TILES,
              "gate regression: fp32 128^3 batched-tiny must pipeline (nKd==1, full tiles, >=32 tiles)");
static_assert(MatmulPipeline<half, Cfg<128, 128, 128, 2048, 128>>::PIPELINE_TILES,
              "gate regression: fp16 128^3 batched-tiny must pipeline");
// Just past the tile floor (32 == MIN_TILES) still fires; one below does not.
static_assert(MatmulPipeline<float, Cfg<128, 128, 64, 32, 128>>::PIPELINE_TILES,
              "gate regression: total_tiles==MIN_TILES must pipeline");
static_assert(!MatmulPipeline<float, Cfg<128, 128, 128, 1, 128>>::PIPELINE_TILES,
              "gate regression: a single tile must NOT pipeline (prologue/epilogue unamortized)");
// Large K -> the deep tile's within-tile prefetch already fires (nKd>1); stay on it.
static_assert(!MatmulPipeline<float, Cfg<128, 128, 512, 2048, 128>>::PIPELINE_TILES,
              "gate regression: large-K (nKd>1) must NOT take the cross-tile loop");
// Partial tile (free dim not a whole-tile multiple) -> per-tile path keeps the clamps.
static_assert(!MatmulPipeline<float, Cfg<200, 128, 128, 64, 128>>::PIPELINE_TILES,
              "gate regression: partial-tile (A_PADDED) must NOT pipeline");
} // namespace gate_selftest

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
template <typename data_T, typename CONFIG_T, bool FUSE_OUT = false, bool NT_IN = false>
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
        // Fused output store: the destination is `res` (the true out_shape), so the tile
        // lands at its permuted position with the free0 axis's res stride as the row
        // stride (free1 is innermost -> col stride 1). Default (FUSE_OUT=false) keeps the
        // natural ws_res layout (row stride L1) byte-for-byte unchanged.
        using FOut = FusibleOutputPerm<CONFIG_T>;
        constexpr unsigned ROW_STRIDE = FUSE_OUT ? FOut::row_stride : L1;
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

        // Direct strided-input: read both operands straight from the raw tensors (no
        // Phase A). A is always a row-strided ND tile (free0 row stride NtIn::row_stride0,
        // contraction contiguous). B has two read modes:
        //   NT (NtIn::dn): transposed via Layout::DN (contraction contiguous == K
        //       innermost, B_K_STRIDE=1, free1 col stride NtIn::col_stride1, ColMajor tile).
        //   NN-strided (NT_IN && !dn): Layout::ND with a strided K row (B_K_STRIDE=
        //       NtIn::k_stride1, free1 contiguous so B_N_STRIDE=col_stride1==1), i.e. the
        //       canonical NN tile orientation with a non-contiguous K row stride.
        // The default (NT_IN=false) path is the contiguous ws0/ws1 read, byte-for-byte
        // unchanged (B_K_STRIDE=Bcols == k_stride uniform below).
        using NtIn = NtInput<CONFIG_T>;
        constexpr bool NT_DN = NT_IN && NtIn::dn;
        constexpr unsigned A_ROW_STRIDE = NT_IN ? NtIn::row_stride0 : K;
        constexpr unsigned B_K_STRIDE   = NT_DN ? 1u : (NT_IN ? NtIn::k_stride1 : Bcols);
        constexpr unsigned B_N_STRIDE   = NT_IN ? NtIn::col_stride1 : 1u;
        constexpr pto::Layout B_LAYOUT  = NT_DN ? pto::Layout::DN : pto::Layout::ND;
        constexpr pto::BLayout B_BLAYOUT = NT_DN ? pto::BLayout::RowMajor : pto::BLayout::ColMajor;
        constexpr pto::SLayout B_SLAYOUT = NT_DN ? pto::SLayout::ColMajor : pto::SLayout::RowMajor;

        using ShapeA = pto::Shape<1, 1, 1, -1, Kd>;
        using StrideA = pto::Stride<1, 1, 1, A_ROW_STRIDE, 1>;
        using ShapeB = pto::Shape<1, 1, 1, Kd, -1>;
        using StrideB = pto::Stride<1, 1, 1, B_K_STRIDE, B_N_STRIDE>;
        using ShapeC = pto::Shape<1, 1, 1, -1, -1>;
        using StrideC = pto::Stride<1, 1, 1, ROW_STRIDE, 1>;
        using GlobalB = pto::GlobalTensor<data_T, ShapeB, StrideB, B_LAYOUT>;
        using MatTileA = pto::Tile<pto::TileType::Mat, data_T, Mt, Kd, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, 512>;
        using MatTileB = pto::Tile<pto::TileType::Mat, data_T, Kd, Nt, B_BLAYOUT, -1, -1, B_SLAYOUT, 512>;
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

        // Operand source bases. NT: decode the flat batch index `i` into each operand's
        // source base (Σ idx_k·src_stride_k over the inplace axes), then add the tile's
        // row0/col0 corner with the strided free dims (contraction is contiguous, so the
        // K offset added at load time is +k0 for both). Default: the canonical contiguous
        // ws0/ws1 layout (i·Arows·K + row0·K  /  i·K·Bcols + col0), byte-for-byte unchanged.
        const __gm__ data_T* aBase;
        const __gm__ data_T* bBase;
        if constexpr (NT_IN) {
            unsigned baseA = 0, baseB = 0, rr = i;
            #pragma unroll
            for (int k = int(NtIn::n_batch) - 1; k >= 0; --k) {
                unsigned idx = rr % CONFIG_T::in_batch_sizes[k];
                baseA += idx * CONFIG_T::in0_batch_strides[k];
                baseB += idx * CONFIG_T::in1_batch_strides[k];
                rr /= CONFIG_T::in_batch_sizes[k];
            }
            aBase = ws0 + baseA + size_t(row0) * A_ROW_STRIDE;
            bBase = ws1 + baseB + size_t(col0) * B_N_STRIDE;
        } else {
            aBase = ws0 + i * Arows * K + row0 * K;
            bBase = ws1 + i * K * Bcols + col0;
        }

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
            GlobalB bg(const_cast<__gm__ data_T*>(bBase), ShapeB(loadN));
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
                // Contraction K-offset: contiguous for NT (stride 1 -> +k0), row-strided
                // for the canonical layout (B's K stride is Bcols -> +k0*Bcols).
                const __gm__ data_T* bChunk = bBase + size_t(k0) * B_K_STRIDE;
                pto::GlobalTensor<data_T, ShapeA, StrideA> ag(const_cast<__gm__ data_T*>(aBase + k0), ShapeA(loadM));
                GlobalB bg(const_cast<__gm__ data_T*>(bChunk), ShapeB(loadN));
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
        // Base offset of this tile's (row0,col0) corner in the destination. Natural
        // layout: i*L0*L1 + row0*L1 + col0. Fused: decode the flat batch index `i`
        // row-major over the inplace axes into a res base (Σ idx_k·res_stride_k), then
        // add row0·ROW_STRIDE + col0 (free1 innermost -> unit col stride).
        unsigned base;
        if constexpr (FUSE_OUT) {
            base = row0 * ROW_STRIDE + col0;
            unsigned rem = i;
            #pragma unroll
            for (int k = int(FOut::n_batch) - 1; k >= 0; --k) {
                base += (rem % CONFIG_T::out_batch_sizes[k]) * CONFIG_T::out_batch_strides[k];
                rem /= CONFIG_T::out_batch_sizes[k];
            }
        } else {
            base = i * L0 * L1 + row0 * L1 + col0;
        }
        pto::GlobalTensor<float, ShapeC, StrideC> cg(ws_res + base, ShapeC(validM, validN));
        TSTORE(cg, c_l0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
}

// Cross-tile software-pipelined schedule for the nKd==1 regime (gated by
// MatmulPipeline::PIPELINE_TILES). Each tile is a single GM->L1 chunk, so
// matmul_one_tile_deep's within-tile prefetch never fires and its GM->L1 load and
// fixpipe store are fully exposed per tile. This loop pipelines across the tile
// boundary instead: it prefetches tile t+1's GM->L1 load (MTE2) and lets tile t-1's
// fixpipe store (FIX) drain under tile t's compute (M). L1 A/B operands ping-pong
// across two buffers (a_l1/b_l1[2]) and the L0C accumulator ping-pongs across two
// (c_l0[2], the two that fit L0C); flags are drained once at loop exit instead of
// per tile. All tiles are full (PIPELINE_TILES requires !A_PADDED && !B_PADDED), so
// validM==Mt / validN==Nt are loop-invariant and the valid extents are set once.
//
// Event namespaces (each (producer,consumer) pipe pair is an independent ID space),
// keyed by L1-buffer parity q (A->ID0/ID1, B->ID2/ID3) and accumulator parity:
//   MTE1->MTE2  L1 buffer free (WAR: a load waits the extract that drained it)
//   MTE2->MTE1  GM->L1 load done (RAW: compute waits its load)
//   M->MTE1     L0 ping-pong free (WAR: an extract waits the matmul that used it)
//   MTE1->M     extract done (RAW: matmul waits its extract) -- ID0, set/wait adjacent
//   M->FIX      matmul done (RAW: the store waits this tile's matmuls)
//   FIX->M      accumulator free (WAR: tile t+2 reuses c_l0[q] after t's store)
template <typename data_T, typename CONFIG_T, bool FUSE_OUT = false, bool NT_IN = false>
AICORE inline void matmul_tile_loop_pipelined(__gm__ const data_T* ws0, __gm__ const data_T* ws1,
                                              __gm__ float* ws_res, unsigned start, unsigned end) {
        using G = MatmulGeom<CONFIG_T>;
        using P = MatmulPipeline<data_T, CONFIG_T>;
        constexpr unsigned L0 = G::L0;
        constexpr unsigned L1 = G::L1;
        constexpr unsigned K  = G::K;
        constexpr unsigned Mt = G::Mt;
        constexpr unsigned Nt = G::Nt;
        constexpr unsigned nN = G::nN;
        constexpr unsigned tiles_per_batch = G::tiles_per_batch;
        constexpr unsigned Arows = G::Arows;
        constexpr unsigned Bcols = G::Bcols;

        using FOut = FusibleOutputPerm<CONFIG_T>;
        constexpr unsigned ROW_STRIDE = FUSE_OUT ? FOut::row_stride : L1;

        constexpr unsigned ds  = sizeof(data_T);
        constexpr unsigned Kq  = P::Kq;
        constexpr unsigned Kd  = P::Kd;     // == K (nKd==1)
        constexpr unsigned nP  = Kd / Kq;   // L0 compute phases per tile
        static_assert(P::nKd == 1u, "pipelined tile loop requires single-chunk (nKd==1) tiles");

        if (start >= end) return;

        // Direct strided-input (see matmul_one_tile_deep): A row-strided ND; B either NT
        // (DN-transposed, NtIn::dn) or NN-strided (ND with a strided K row). nKd==1 here so
        // there is no within-tile K-chunk offset. Default path (NT_IN=false) is the
        // canonical contiguous ws0/ws1 read, byte-for-byte unchanged.
        using NtIn = NtInput<CONFIG_T>;
        constexpr bool NT_DN = NT_IN && NtIn::dn;
        constexpr unsigned A_ROW_STRIDE = NT_IN ? NtIn::row_stride0 : K;
        constexpr unsigned B_K_STRIDE   = NT_DN ? 1u : (NT_IN ? NtIn::k_stride1 : Bcols);
        constexpr unsigned B_N_STRIDE   = NT_IN ? NtIn::col_stride1 : 1u;
        constexpr pto::Layout B_LAYOUT  = NT_DN ? pto::Layout::DN : pto::Layout::ND;
        constexpr pto::BLayout B_BLAYOUT = NT_DN ? pto::BLayout::RowMajor : pto::BLayout::ColMajor;
        constexpr pto::SLayout B_SLAYOUT = NT_DN ? pto::SLayout::ColMajor : pto::SLayout::RowMajor;

        using ShapeA = pto::Shape<1, 1, 1, -1, Kd>;
        using StrideA = pto::Stride<1, 1, 1, A_ROW_STRIDE, 1>;
        using ShapeB = pto::Shape<1, 1, 1, Kd, -1>;
        using StrideB = pto::Stride<1, 1, 1, B_K_STRIDE, B_N_STRIDE>;
        using ShapeC = pto::Shape<1, 1, 1, -1, -1>;
        using StrideC = pto::Stride<1, 1, 1, ROW_STRIDE, 1>;
        using GlobalB = pto::GlobalTensor<data_T, ShapeB, StrideB, B_LAYOUT>;
        using MatTileA = pto::Tile<pto::TileType::Mat, data_T, Mt, Kd, pto::BLayout::ColMajor, -1, -1, pto::SLayout::RowMajor, 512>;
        using MatTileB = pto::Tile<pto::TileType::Mat, data_T, Kd, Nt, B_BLAYOUT, -1, -1, B_SLAYOUT, 512>;
        using LeftTileA = pto::TileLeft<data_T, Mt, Kq, -1, -1>;
        using RightTileB = pto::TileRight<data_T, Kq, Nt, -1, -1>;
        using AccTileC = pto::TileAcc<float, Mt, Nt, -1, -1>;

        constexpr unsigned matBytesA = Mt * Kd * ds;
        constexpr unsigned matBytesB = Kd * Nt * ds;
        static_assert(2 * matBytesA + 2 * matBytesB <= 512u * 1024u,
                      "Pipelined L1 A/B ping-pong exceeds the 512 KB CBUF.");
        constexpr unsigned l0aBytes = Mt * Kq * ds;
        constexpr unsigned l0bBytes = Kq * Nt * ds;
        static_assert(2 * l0aBytes <= 64u * 1024u, "Pipelined L0A ping-pong exceeds L0A.");
        static_assert(2 * l0bBytes <= 64u * 1024u, "Pipelined L0B ping-pong exceeds L0B.");
        constexpr unsigned accBytes = Mt * Nt * sizeof(float);
        static_assert(2 * accBytes <= 128u * 1024u, "Pipelined L0C accumulator ping-pong exceeds L0C.");

        MatTileA a_l1[2];
        MatTileB b_l1[2];
        LeftTileA a_l0[2];
        RightTileB b_l0[2];
        AccTileC c_l0[2];
        TASSIGN(a_l1[0], 0);
        TASSIGN(a_l1[1], matBytesA);
        TASSIGN(b_l1[0], 2 * matBytesA);
        TASSIGN(b_l1[1], 2 * matBytesA + matBytesB);
        TASSIGN(a_l0[0], 0);
        TASSIGN(a_l0[1], l0aBytes);
        TASSIGN(b_l0[0], 0);
        TASSIGN(b_l0[1], l0bBytes);
        TASSIGN(c_l0[0], 0);
        TASSIGN(c_l0[1], accBytes);

        // Full tiles (PIPELINE_TILES gate): every tile's valid extents are identical,
        // so set them once. Operands load the full tile so the ND->NZ packing has no
        // empty fractal block.
        a_l1[0].SetValidRow(Mt); a_l1[0].SetValidCol(Kd);
        a_l1[1].SetValidRow(Mt); a_l1[1].SetValidCol(Kd);
        b_l1[0].SetValidRow(Kd); b_l1[0].SetValidCol(Nt);
        b_l1[1].SetValidRow(Kd); b_l1[1].SetValidCol(Nt);
        a_l0[0].SetValidRow(Mt); a_l0[0].SetValidCol(Kq);
        a_l0[1].SetValidRow(Mt); a_l0[1].SetValidCol(Kq);
        b_l0[0].SetValidRow(Kq); b_l0[0].SetValidCol(Nt);
        b_l0[1].SetValidRow(Kq); b_l0[1].SetValidCol(Nt);
        c_l0[0].SetValidRow(Mt); c_l0[0].SetValidCol(Nt);
        c_l0[1].SetValidRow(Mt); c_l0[1].SetValidCol(Nt);

        // Init: both L1 A/B buffers free, both L0 ping-pong buffers free, both
        // accumulators free (primes the WAR flags the first iterations wait on).
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);   // a_l1[0] free
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);   // a_l1[1] free
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);   // b_l1[0] free
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);   // b_l1[1] free
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);      // a_l0[0]/b_l0[0] free
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);      // a_l0[1]/b_l0[1] free
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);       // c_l0[0] free
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);       // c_l0[1] free

        // Prologue: load tile `start` into its L1 buffer (parity start&1).
        {
            unsigned t = start, q = t & 1;
            unsigned i = t / tiles_per_batch, rem = t % tiles_per_batch;
            unsigned row0 = (rem / nN) * Mt, col0 = (rem % nN) * Nt;
            const __gm__ data_T* aBase;
            const __gm__ data_T* bBase;
            if constexpr (NT_IN) {
                unsigned baseA = 0, baseB = 0, rr = i;
                #pragma unroll
                for (int k = int(NtIn::n_batch) - 1; k >= 0; --k) {
                    unsigned idx = rr % CONFIG_T::in_batch_sizes[k];
                    baseA += idx * CONFIG_T::in0_batch_strides[k];
                    baseB += idx * CONFIG_T::in1_batch_strides[k];
                    rr /= CONFIG_T::in_batch_sizes[k];
                }
                aBase = ws0 + baseA + size_t(row0) * A_ROW_STRIDE;
                bBase = ws1 + baseB + size_t(col0) * B_N_STRIDE;
            } else {
                aBase = ws0 + i * Arows * K + row0 * K;
                bBase = ws1 + i * K * Bcols + col0;
            }
            event_t aF = q ? EVENT_ID1 : EVENT_ID0;
            event_t bF = q ? EVENT_ID3 : EVENT_ID2;
            pto::GlobalTensor<data_T, ShapeA, StrideA> ag(const_cast<__gm__ data_T*>(aBase), ShapeA(Mt));
            GlobalB bg(const_cast<__gm__ data_T*>(bBase), ShapeB(Nt));
            wait_flag(PIPE_MTE1, PIPE_MTE2, aF);
            TLOAD(a_l1[q], ag);
            set_flag(PIPE_MTE2, PIPE_MTE1, aF);
            wait_flag(PIPE_MTE1, PIPE_MTE2, bF);
            TLOAD(b_l1[q], bg);
            set_flag(PIPE_MTE2, PIPE_MTE1, bF);
        }

        unsigned gphase = 0;
        for (unsigned t = start; t < end; t++) {
            unsigned q = t & 1;
            // Prefetch tile t+1 into the other L1 buffer (overlaps this tile's compute).
            if (t + 1 < end) {
                unsigned tn = t + 1, qn = tn & 1;
                unsigned i = tn / tiles_per_batch, rem = tn % tiles_per_batch;
                unsigned row0 = (rem / nN) * Mt, col0 = (rem % nN) * Nt;
                const __gm__ data_T* aBase;
                const __gm__ data_T* bBase;
                if constexpr (NT_IN) {
                    unsigned baseA = 0, baseB = 0, rr = i;
                    #pragma unroll
                    for (int k = int(NtIn::n_batch) - 1; k >= 0; --k) {
                        unsigned idx = rr % CONFIG_T::in_batch_sizes[k];
                        baseA += idx * CONFIG_T::in0_batch_strides[k];
                        baseB += idx * CONFIG_T::in1_batch_strides[k];
                        rr /= CONFIG_T::in_batch_sizes[k];
                    }
                    aBase = ws0 + baseA + size_t(row0) * A_ROW_STRIDE;
                    bBase = ws1 + baseB + size_t(col0) * B_N_STRIDE;
                } else {
                    aBase = ws0 + i * Arows * K + row0 * K;
                    bBase = ws1 + i * K * Bcols + col0;
                }
                event_t aF = qn ? EVENT_ID1 : EVENT_ID0;
                event_t bF = qn ? EVENT_ID3 : EVENT_ID2;
                pto::GlobalTensor<data_T, ShapeA, StrideA> ag(const_cast<__gm__ data_T*>(aBase), ShapeA(Mt));
                GlobalB bg(const_cast<__gm__ data_T*>(bBase), ShapeB(Nt));
                wait_flag(PIPE_MTE1, PIPE_MTE2, aF);
                TLOAD(a_l1[qn], ag);
                set_flag(PIPE_MTE2, PIPE_MTE1, aF);
                wait_flag(PIPE_MTE1, PIPE_MTE2, bF);
                TLOAD(b_l1[qn], bg);
                set_flag(PIPE_MTE2, PIPE_MTE1, bF);
            }

            // Compute tile t from L1 buffer q into accumulator c_l0[q].
            event_t aLd = q ? EVENT_ID1 : EVENT_ID0;
            event_t bLd = q ? EVENT_ID3 : EVENT_ID2;
            event_t acc = q ? EVENT_ID1 : EVENT_ID0;
            wait_flag(PIPE_MTE2, PIPE_MTE1, aLd);   // this tile's GM->L1 load done
            wait_flag(PIPE_MTE2, PIPE_MTE1, bLd);
            wait_flag(PIPE_FIX, PIPE_M, acc);       // c_l0[q] free (tile t-2 stored)

            for (unsigned p = 0; p < nP; p++) {
                unsigned pp = gphase & 1;
                event_t l0f = pp ? EVENT_ID1 : EVENT_ID0;
                wait_flag(PIPE_M, PIPE_MTE1, l0f);          // a_l0[pp]/b_l0[pp] free
                TEXTRACT(a_l0[pp], a_l1[q], 0, p * Kq);
                TEXTRACT(b_l0[pp], b_l1[q], p * Kq, 0);
                if (p == nP - 1) {                           // last extract frees this L1 buffer
                    set_flag(PIPE_MTE1, PIPE_MTE2, aLd);
                    set_flag(PIPE_MTE1, PIPE_MTE2, bLd);
                }
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);      // RAW: matmul waits its extract
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
                if (p == 0)
                    TMATMUL(c_l0[q], a_l0[pp], b_l0[pp]);
                else
                    TMATMUL_ACC(c_l0[q], a_l0[pp], b_l0[pp]);
                set_flag(PIPE_M, PIPE_MTE1, l0f);            // a_l0[pp]/b_l0[pp] free
                gphase++;
            }

            // Store accumulator c_l0[q] -> GM. The FIX store is issued but NOT waited
            // here; it drains under the next tiles' compute, and tile t+2 (or the
            // epilogue drain) waits the FIX->M free before reusing c_l0[q].
            set_flag(PIPE_M, PIPE_FIX, acc);    // RAW: store waits this tile's matmuls
            wait_flag(PIPE_M, PIPE_FIX, acc);
            unsigned i = t / tiles_per_batch, rem = t % tiles_per_batch;
            unsigned row0 = (rem / nN) * Mt, col0 = (rem % nN) * Nt;
            unsigned base;
            if constexpr (FUSE_OUT) {
                base = row0 * ROW_STRIDE + col0;
                unsigned rr = i;
                #pragma unroll
                for (int k = int(FOut::n_batch) - 1; k >= 0; --k) {
                    base += (rr % CONFIG_T::out_batch_sizes[k]) * CONFIG_T::out_batch_strides[k];
                    rr /= CONFIG_T::out_batch_sizes[k];
                }
            } else {
                base = i * L0 * L1 + row0 * L1 + col0;
            }
            pto::GlobalTensor<float, ShapeC, StrideC> cg(ws_res + base, ShapeC(Mt, Nt));
            TSTORE(cg, c_l0[q]);
            set_flag(PIPE_FIX, PIPE_M, acc);    // WAR: c_l0[q] free for tile t+2 / drain
        }

        // Drain the WAR-free flags the tail iterations set but nothing consumed (both
        // L1 A/B buffers, both L0 buffers, both accumulators) so no flag state leaks
        // and both in-flight fixpipe stores complete before returning.
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
}

// Factored out so the standalone kernel and the fused kernel share it; the caller
// guards with `if constexpr (DAV_CUBE)` and supplies the block index/count.
// SPLITK enables the split-K schedule (see batched_matmul_inline body); it is off
// for the standalone matmul and on only for fused configs with idle cores.
template <typename data_T, typename CONFIG_T, bool SPLITK = false, bool FUSE_OUT = false, bool NT_IN = false>
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
        // core), full K per tile. (An L2 tile-swizzle was tried here and removed —
        // measured a no-op on square grids, HBM-bound, and a regression on non-square
        // ones.) For the nKd==1 batched-tiny regime the deep tile's within-tile
        // prefetch is dead (load/store fully exposed, latency-bound), so route to the
        // cross-tile pipelined loop; everything else keeps the per-tile deep path.
        unsigned chunk = (total_tiles + block_num - 1) / block_num;
        unsigned start = block_idx * chunk;
        unsigned end = (start + chunk) < total_tiles ? (start + chunk) : total_tiles;
        if constexpr (MatmulPipeline<data_T, CONFIG_T>::PIPELINE_TILES) {
            matmul_tile_loop_pipelined<data_T, CONFIG_T, FUSE_OUT, NT_IN>(ws0, ws1, ws_res, start, end);
        } else {
            for (unsigned t = start; t < end; t++) {
                matmul_one_tile_deep<data_T, CONFIG_T, FUSE_OUT, NT_IN>(ws0, ws1, ws_res, t);
            }
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

} // namespace pto_einsum
