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

// Align `v` up to the next multiple of `m` (m a power-of-two factor of the dims
// we care about), used to pad UB tile row-strides for the TTRANS hardware path.
AICORE inline constexpr unsigned align_up(unsigned v, unsigned m) { return (v + m - 1) / m * m; }

// Mixed-radix odometer over the leading NAX output axes (CONFIG_T::to_shape /
// perm_strides). The N-D transpose loops walk a flat outer index `g` whose source
// base offset is Σ_k digit_k · perm_strides[k]; the natural decode is a per-step
// div/mod over the (constexpr) extents. `init()` pays that decompose exactly once,
// then `advance()` carries the digits with add/compare only — no per-iteration
// div/mod in the hot loop (carries are rare, so the common case is a single add).
// MAX_AX bounds the local digit array (einsum operand rank is small).
template <typename CONFIG_T, unsigned NAX>
struct PermOdometer {
    static constexpr unsigned MAX_AX = 8;
    static_assert(NAX <= MAX_AX, "PermOdometer: operand rank exceeds MAX_AX");
    unsigned digit[MAX_AX];
    unsigned base;

    AICORE inline void init(unsigned flat) {
        base = 0;
        for (int k = int(NAX) - 1; k >= 0; --k) {
            unsigned d = flat % CONFIG_T::to_shape[k];
            digit[k] = d;
            base += d * CONFIG_T::perm_strides[k];
            flat /= CONFIG_T::to_shape[k];
        }
    }

    AICORE inline void advance() {
        for (int k = int(NAX) - 1; k >= 0; --k) {
            base += CONFIG_T::perm_strides[k];
            if (++digit[k] < CONFIG_T::to_shape[k]) return;       // no carry: done
            digit[k] = 0;
            base -= CONFIG_T::to_shape[k] * CONFIG_T::perm_strides[k];  // wrap this axis, carry up
        }
    }
};

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

} // namespace pto_einsum
