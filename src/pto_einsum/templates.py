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
elementwise_config_template = """struct config_elementwise {{
    static const unsigned N = {N};
}};
"""

# Elementwise (Hadamard) variant. Exposes the SAME extern "C" entry-point names as
# the matmul path (run_einsum / run_einsum_setup / exec / teardown) so builder.py's
# dispatch is unchanged; only the bodies call the elementwise kernels. The
# transpose / batched_matmul entry points are intentionally absent (there is no
# transpose or matmul), so builder.load_library skips their argtypes for this path.
shared_lib_elementwise_template = """#include "pto_einsum.h"

{config_code}

extern "C" {{
    void run_einsum(const {data_t}* input0, const {data_t}* input1, float* output, void* stream) {{
        pto_einsum::elementwise_mul<{data_t}, config_elementwise>(input0, input1, output, stream);
    }}
    void* run_einsum_setup(void* stream) {{
        return pto_einsum::elementwise_setup(stream);
    }}
    void run_einsum_exec(const {data_t}* input0, const {data_t}* input1, float* output, void* workspace, void* stream) {{
        pto_einsum::elementwise_exec<{data_t}, config_elementwise>(input0, input1, output, workspace, stream);
    }}
    void run_einsum_teardown(void* workspace) {{
        pto_einsum::einsum_teardown(workspace);
    }}
}}
"""

cpu_lib_elementwise_template = """#include "cpu_einsum.h"
#include <stdio.h>

{config_code}

extern "C" {{
    void run_einsum(const {data_t}* input0, const {data_t}* input1, float* output, void* stream) {{
        cpu::elementwise_mul<{data_t}, config_elementwise>(input0, input1, output);
    }}
}}
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
    void* run_einsum_setup(void* stream) {{
        return pto_einsum::einsum_setup<{data_t}, config_einsum>(stream);
    }}
    void run_einsum_exec(const {data_t}* input0, const {data_t}* input1, float* output, void* workspace, void* stream) {{
        pto_einsum::einsum_exec<{data_t}, config_einsum>(input0, input1, output, workspace, stream);
    }}
    void run_einsum_teardown(void* workspace) {{
        pto_einsum::einsum_teardown(workspace);
    }}
}}
"""
