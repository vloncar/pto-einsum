#pragma once
// Fused einsum pipeline entrypoint. The implementation is split across phase
// headers; this file wires them together and defines the single fused kernel
// (transpose -> matmul -> transpose) plus the host setup/exec/teardown ABI.
#include "pto_einsum_utils.h"
#include "pto_einsum_transpose.h"
#include "pto_einsum_matmul.h"
#include "pto_einsum_elementwise.h"

// FFTS control-address query used by the fused kernel's host launcher. We
// forward-declare it instead of including <runtime/rt_ffts.h>, which drags in a
// profiling header absent from some CANN toolkits. Provided by libruntime
// (link -lruntime). The kernel-side set_ffts_base_addr comes from pto-inst.hpp.
extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

namespace pto_einsum {

// Phase-profiling gate (Plan 0, transpose↔matmul overlap study). EINSUM_PHASE_STOP
// caps which fused phases execute their *compute* body, while the cross-core
// SyncAll barriers always run, so the launch/barrier structure is identical across
// stop levels and the per-phase wall-time isolates by subtraction:
//   0 = barriers only (fixed launch+barrier overhead F)
//   1 = + Phase A (input transposes)        -> t(1)-t(0) = T_A
//   2 = + Phase B (Cube matmul)             -> t(2)-t(1) = T_B
//   3 = + Phase C (output transpose) [full] -> t(3)-t(2) = T_C
// Default 3 reproduces the production kernel exactly (every body enabled), so this
// is a no-op unless a profiler build passes -DEINSUM_PHASE_STOP=N.
#ifndef EINSUM_PHASE_STOP
#define EINSUM_PHASE_STOP 3
#endif

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
    // Fused output permutation: the Cube store lands each tile straight into res in
    // permuted order, so (like identity output) Phase C and the second barrier drop and
    // ws_res is not allocated. OUT_DIRECT covers both "matmul writes res directly" cases.
    constexpr bool FUSE_OUT = FusibleOutputPerm<CONFIG_T>::value && !OUT_IDENTITY;
    constexpr bool OUT_DIRECT = OUT_IDENTITY || FUSE_OUT;
    // Inputs whose copy into the workspace is redundant (identity perm + K==C) are
    // read straight from data0/data1; their Phase A transpose is skipped below.
    constexpr bool SKIP0 = EinsumSkip<CONFIG_T>::inp0;
    constexpr bool SKIP1 = EinsumSkip<CONFIG_T>::inp1;
    // NT strided-input: the contraction is innermost+contiguous on both inputs, so the
    // matmul reads them straight from data0/data1 in their natural strided layout (A
    // row-strided, B transposed via DN) — dropping BOTH input transposes. RD0/RD1 are
    // the read-direct flags (skip Phase A and read the raw tensor) for each operand,
    // covering the identity-skip and NT cases uniformly.
    constexpr bool NT_IN = NtInput<CONFIG_T>::value;
    constexpr bool RD0 = SKIP0 || NT_IN;
    constexpr bool RD1 = SKIP1 || NT_IN;
    // Split-K is enabled only when the output is identity (so the matmul target is
    // `res`, which the host pre-zeros for these configs — see builder._splitk so the
    // cores can atomic-add into it), there is K to split, and the output-tile grid
    // is small enough to leave cores idle on the plain schedule. The runtime
    // ksplit (>=2) gate inside collapses it to the plain path on small parts.
    constexpr bool SPLITK_ELIGIBLE =
        OUT_IDENTITY && (MatmulGeom<CONFIG_T>::nK >= 2) && (MatmulGeom<CONFIG_T>::total_tiles < 16)
        && !NT_IN;

    if constexpr (DAV_VEC) {

        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        constexpr unsigned C = CONFIG_T::n_contract;
        constexpr unsigned K = (C + 15) / 16 * 16;
        constexpr unsigned I = CONFIG_T::n_inplace;
        constexpr unsigned L1 = CONFIG_T::n_free1;
        unsigned lane = get_block_idx() * 2 + get_subblockid();
        unsigned nlanes = get_block_num() * 2;

#if EINSUM_PHASE_STOP >= 1
        // input0's transpose output is [.., contract]; pad its innermost contract
        // dim C up to the fractal width K so the Cube reads a full K-wide A. The
        // pad is uniform across all I*L0 rows, so any batch count works directly.
        // RD0: ws0 would be a byte-identical copy (SKIP0) or the NT path reads data0
        // strided — either way Phase A's transpose is redundant, read data0 directly.
        if constexpr (!RD0)
            transpose_inline<data_T, typename CONFIG_T::tpose_inp0_config, C, K>(data0, ws0, lane, nlanes);

        // input1's contract dim is the per-batch row count. For K==C, or a single
        // batch, the K-C tail sits after the (one) valid block, so a plain
        // transpose into the pre-zeroed ws1 suffices. For K!=C with batching, each
        // batch's C valid rows must land at the front of its K-row slot (stride
        // K*L1) — a contiguous write would mis-stride the batches. An identity input1
        // is the contiguous repack (batched_pad_copy_inline); a non-identity input1
        // transposes *and* row-pads (transpose_inline_rowpad). RD1: ws1 == data1 (skip)
        // or the NT path reads data1 transposed (DN) straight from GM — no Phase A.
        if constexpr (!RD1) {
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
#endif  // EINSUM_PHASE_STOP >= 1
    }

    SyncAll<false>();  // inputs transposed -> matmul

#if EINSUM_PHASE_STOP >= 2
    if constexpr (DAV_CUBE) {
        // Identity output: the matmul's natural layout already is res, so write it
        // directly and skip the whole output-transpose pass below. SPLITK_ELIGIBLE
        // configs (identity, thin grid) hand idle cores K-slices that atomic-add
        // into the pre-zeroed res. The matmul reads each input from the workspace, or
        // straight from the raw tensor when its copy was skipped (SKIP0/SKIP1).
        __gm__ float* mm_out = OUT_DIRECT ? res : ws_res;
        __gm__ const data_T* mmA = RD0 ? data0 : static_cast<__gm__ const data_T*>(ws0);
        __gm__ const data_T* mmB = RD1 ? data1 : static_cast<__gm__ const data_T*>(ws1);
        batched_matmul_inline<data_T, CONFIG_T, SPLITK_ELIGIBLE, FUSE_OUT, NT_IN>(mmA, mmB, mm_out, get_block_idx(), get_block_num());
    }
#endif  // EINSUM_PHASE_STOP >= 2

    if constexpr (!OUT_DIRECT) {
        SyncAll<false>();  // matmul done -> output transpose

#if EINSUM_PHASE_STOP >= 3
        if constexpr (DAV_VEC) {
            unsigned lane = get_block_idx() * 2 + get_subblockid();
            unsigned nlanes = get_block_num() * 2;
            transpose_inline<float, typename CONFIG_T::tpose_out_conf>(ws_res, res, lane, nlanes);
        }
#endif  // EINSUM_PHASE_STOP >= 3
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
    // Inputs read straight from data0/data1 — identity-skip (perm + K==C) or NT (strided
    // read) — occupy no workspace region. Note K!=C forces every read-direct flag false,
    // so the pad-zeroing path below always has its buffers.
    constexpr bool RD0 = EinsumSkip<CONFIG_T>::inp0 || NtInput<CONFIG_T>::value;
    constexpr bool RD1 = EinsumSkip<CONFIG_T>::inp1 || NtInput<CONFIG_T>::value;
    const size_t ws0_elems = RD0 ? 0 : einsum_ws0_elems<data_T, CONFIG_T>();
    const size_t ws1_elems = RD1 ? 0 : einsum_ws1_elems<data_T, CONFIG_T>();
    const size_t ws0_bytes = sizeof(data_T) * ws0_elems;
    const size_t ws1_bytes = sizeof(data_T) * ws1_elems;
    // Identity or fused output writes straight to res, so ws_res is never read — don't
    // alloc it. FusibleOutputPerm carries the same permuted-store gate as the kernel.
    constexpr bool OUT_IDENTITY = IsIdentityPerm<typename CONFIG_T::tpose_out_conf>::value;
    constexpr bool OUT_DIRECT = OUT_IDENTITY || FusibleOutputPerm<CONFIG_T>::value;
    const size_t wsr_bytes = OUT_DIRECT ? 0 : sizeof(float) * CONFIG_T::tpose_out_conf::N;

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

    // Match einsum_setup's layout: read-direct inputs (identity-skip or NT) occupy no
    // workspace region.
    constexpr bool RD0 = EinsumSkip<CONFIG_T>::inp0 || NtInput<CONFIG_T>::value;
    constexpr bool RD1 = EinsumSkip<CONFIG_T>::inp1 || NtInput<CONFIG_T>::value;
    const size_t ws0_elems = RD0 ? 0 : einsum_ws0_elems<data_T, CONFIG_T>();
    const size_t ws1_elems = RD1 ? 0 : einsum_ws1_elems<data_T, CONFIG_T>();

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
