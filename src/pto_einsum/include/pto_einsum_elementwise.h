#pragma once
#include "pto_einsum_utils.h"

namespace pto_einsum {

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
// intra-V hazard.

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

} // namespace pto_einsum
