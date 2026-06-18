#pragma once

#include <vector>

namespace cpu {

/* 
Transpose operation expects a config like this

struct transpose_config {
    static constexpr unsigned dims = 2;
    static constexpr unsigned N = 2048;
    static constexpr unsigned from_shape[2] = {64, 32};
    static constexpr unsigned to_shape[2] = {64, 32};
    static constexpr unsigned perm[2] = {0, 1};
    static constexpr unsigned perm_strides[2] = {32, 1};
};

Matrix-multiply operation expects a config like this:

struct einsum_config {
    typedef config_tpose_inp0 tpose_inp0_config;
    typedef config_tpose_inp1 tpose_inp1_config;
    typedef config_tpose_out tpose_out_conf;

    // Layer Sizes
    static const unsigned n_free0 = 32;
    static const unsigned n_free1 = 128;
    static const unsigned n_contract = 64;
    static const unsigned n_inplace = 1;
};

*/

template <typename CONFIG_T> unsigned transfer_idx(int index) {
    // Given output idx in c-order flat array, return input idx
    int idx = 0;
    for (int i = CONFIG_T::dims - 1; i >= 0; i--) {
        idx += (index % CONFIG_T::to_shape[i]) * CONFIG_T::perm_strides[i];
        index /= CONFIG_T::to_shape[i];
    }
    return idx;
}

template <typename data_T, typename CONFIG_T>
void transpose(const data_T data[CONFIG_T::N], data_T res[CONFIG_T::N]) {
    for (int i = 0; i < CONFIG_T::N; i++) {
        int idx = transfer_idx<CONFIG_T>(i);
        res[i] = data[idx];
    }
}

template <typename data_T, typename CONFIG_T>
void einsum(const data_T data0[CONFIG_T::tpose_inp0_config::N], const data_T data1[CONFIG_T::tpose_inp1_config::N],
            float res[CONFIG_T::tpose_out_conf::N]) {
    std::vector<data_T> tpose_i0(CONFIG_T::tpose_inp0_config::N);
    std::vector<data_T> tpose_i1(CONFIG_T::tpose_inp1_config::N);
    std::vector<float> tpose_o(CONFIG_T::tpose_out_conf::N);
    cpu::transpose<data_T, typename CONFIG_T::tpose_inp0_config>(data0, tpose_i0.data());
    cpu::transpose<data_T, typename CONFIG_T::tpose_inp1_config>(data1, tpose_i1.data());

    constexpr unsigned L0 = CONFIG_T::n_free0;
    constexpr unsigned L1 = CONFIG_T::n_free1;
    constexpr unsigned C = CONFIG_T::n_contract;
    constexpr unsigned I = CONFIG_T::n_inplace;

    float accum_buf;
    for (unsigned i = 0; i < I; i++) {
        for (unsigned l0 = 0; l0 < L0; l0++) {
            for (unsigned l1 = 0; l1 < L1; l1++) {
                accum_buf = 0;
                for (unsigned c = 0; c < C; c++) {
                    data_T a = tpose_i0[(i * L0 + l0) * C + c];
                    data_T b = tpose_i1[i * L1 * C + c * L1 + l1];
                    accum_buf += static_cast<float>(a) * static_cast<float>(b);
                }
                tpose_o[(i * L0 + l0) * L1 + l1] = accum_buf;
            }
        }
    }

    cpu::transpose<float, typename CONFIG_T::tpose_out_conf>(tpose_o.data(), res);
}

// Elementwise (Hadamard) multiply: res[i] = in0[i] * in1[i]. The no-contraction
// einsum case (e.g. `TS,TS->TS`), routed here instead of the matmul path.
template <typename data_T, typename CONFIG_T>
void elementwise_mul(const data_T* in0, const data_T* in1, float* res) {
    for (unsigned i = 0; i < CONFIG_T::N; i++) {
        res[i] = static_cast<float>(in0[i]) * static_cast<float>(in1[i]);
    }
}

} // namespace cpu
