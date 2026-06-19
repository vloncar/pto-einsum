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

    // Fused output permutation (drop Phase C). When the output's free0/free1 are each
    // a single axis and free1 is the innermost output axis (res col stride 1), the
    // Cube store lands each tile straight into res in permuted order. out_fusible is
    // 0/1; out_row_stride is res's free0 (row) stride; the batch arrays decode the
    // flat batch index into a res base offset (res strides of the inplace axes, in
    // out-natural order). All zero/neutral when not fusible.
    static constexpr unsigned out_fusible = {out_fusible};
    static constexpr unsigned out_row_stride = {out_row_stride};
    static constexpr unsigned out_n_batch = {out_n_batch};
    static constexpr unsigned out_batch_sizes[{NB}] = {{{out_batch_sizes}}};
    static constexpr unsigned out_batch_strides[{NB}] = {{{out_batch_strides}}};
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

broadcast_config_template = """struct config_broadcast {{
    static constexpr int mode = {mode};
    static constexpr unsigned N = {N};
    static constexpr unsigned Cc = {Cc};
    static constexpr unsigned Rr = {Rr};
    static constexpr unsigned Inner = {Inner};
    static constexpr unsigned Outer = {Outer};
    static constexpr unsigned sizeB = {sizeB};
    static constexpr unsigned outer_rank = {outer_rank};
    static constexpr unsigned outer_dims[{OR}] = {{{outer_dims}}};
    static constexpr unsigned outer_bstride[{OR}] = {{{outer_bstride}}};
}};
"""

# Broadcast / scaling variant. Like the elementwise path, it reuses the run_einsum* entry
# names so builder dispatch is unchanged. `full`/`bcast` are input0/input1 in whichever
# order puts the full (contiguous) operand first (the broadcast kernel's first argument).
shared_lib_broadcast_template = """#include "pto_einsum.h"

{config_code}

extern "C" {{
    void run_einsum(const {data_t}* input0, const {data_t}* input1, float* output, void* stream) {{
        pto_einsum::broadcast_mul<{data_t}, config_broadcast>({full}, {bcast}, output, stream);
    }}
    void* run_einsum_setup(void* stream) {{
        return pto_einsum::broadcast_setup(stream);
    }}
    void run_einsum_exec(const {data_t}* input0, const {data_t}* input1, float* output, void* workspace, void* stream) {{
        pto_einsum::broadcast_exec<{data_t}, config_broadcast>({full}, {bcast}, output, workspace, stream);
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

cpu_lib_template = """#include "cpu_einsum.h"
#include <stdio.h>

{tpose_inp0_code}
{tpose_inp1_code}
{tpose_out_code}
{einsum_code}

extern "C" {{
    void run_einsum(const {data_t}* input0, const {data_t}* input1, float* output, void* stream) {{
        cpu::einsum<{data_t}, config_einsum>(input0, input1, output);
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
