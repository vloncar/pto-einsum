transpose_config_template = """struct {config_name} {{
    static constexpr unsigned dims = {dims};
    static constexpr unsigned N = {N};
    static constexpr unsigned from_shape[{dims}] = {{{from_shape}}};
    static constexpr unsigned to_shape[{dims}] = {{{to_shape}}};
    static constexpr unsigned perm[{dims}] = {{{perm}}};
    static constexpr unsigned perm_strides[{dims}] = {{{perm_strides}}};
}};
"""

einsum_config_template = """struct config_einsum {{
    typedef {tpose_inp0_name} tpose_inp0_config;
    typedef {tpose_inp1_name} tpose_inp1_config;
    typedef {tpose_out_name} tpose_out_conf;

    static const unsigned n_free0 = {n_free0};
    static const unsigned n_free1 = {n_free1};
    static const unsigned n_contract = {n_contract};
    static const unsigned n_inplace = {n_inplace};

    // Cube matmul tile sizes (padded-dim granularity, multiples of 16). Each is
    // clamped to its padded dim in the kernel; Stage 1 requires the padded dim to
    // be divisible by the (clamped) tile size.
    static const unsigned tile_m = {tile_m};
    static const unsigned tile_n = {tile_n};
    static const unsigned tile_k = {tile_k};
}};
"""
shared_lib_template = """#include "pto_einsum.h"

{tpose_inp0_code}
{tpose_inp1_code}
{tpose_out_code}
{einsum_code}

extern "C" {{
    void run_transpose_inp0(const {data_t}* input, {data_t}* output, void* stream) {{
        pto_einsum::transpose<{data_t}, {tpose_inp0_name}>(input, output, stream);
    }}
    void run_transpose_inp1(const {data_t}* input, {data_t}* output, void* stream) {{
        pto_einsum::transpose<{data_t}, {tpose_inp1_name}>(input, output, stream);
    }}
    void run_transpose_out(const float* input, float* output, void* stream) {{
        pto_einsum::transpose<float, {tpose_out_name}>(input, output, stream);
    }}
    void run_batched_matmul(const {data_t}* ws0, const {data_t}* ws1, float* ws_res, void* stream) {{
        pto_einsum::batched_matmul<{data_t}, config_einsum>(ws0, ws1, ws_res, stream);
    }}
    void run_einsum(const {data_t}* input0, const {data_t}* input1, float* output, void* stream) {{
        pto_einsum::einsum<{data_t}, config_einsum>(input0, input1, output, stream);
    }}
}}
"""
