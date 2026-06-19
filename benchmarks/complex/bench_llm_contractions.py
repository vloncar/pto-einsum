"""Benchmark LLM contraction patterns: pto-einsum vs torch.einsum (PyTorch).

Each operation is defined once as a function parameterised by an ``es`` einsum
callable, then evaluated with two backends:

    * ``torch.einsum``  -- the native PyTorch baseline
    * pto-einsum        -- the JIT-compiled NPU kernel

pto-einsum accepts exactly two operands, so every contraction is expressed as a
chain of two-operand einsums. Both backends evaluate the same expression, which
isolates the einsum primitive itself rather than opt_einsum's contraction
ordering. Patterns that are pure broadcast/scaling (a repeated output index with
no summation, e.g. ``bsd,d->bsd``) are not matmuls; pto-einsum reports them as
unsupported. They are retained to mark the boundary of kernel coverage.

The pattern set merges the LLM-contraction examples with additional ops from
``llm_einsum_ops.js``. Ops from the JS config that duplicate an existing
operation in a different index layout (Q.Kᵀ scores, attention-weighted values,
head-merge projection, GLU gate/value, contrastive similarity) are omitted.

Run on an Ascend NPU host with ``PTO_LIB_PATH`` exported:

    python benchmarks/complex/bench_llm_contractions.py
"""

import re
import statistics
import time

import torch
import torch_npu  # noqa: F401  (registers the NPU backend / .npu())
import torch.nn.functional as F

from pto_einsum import EinsumBuilder

# ── Timing configuration ──────────────────────────────────────────────────────
WARMUP = 5
RUNS = 30

# ── Dimensions ────────────────────────────────────────────────────────────────
B, S, T = 2, 16, 16    # batch, query seq len, key seq len
H, D    = 8, 64        # attention heads, head dim
I, O    = 512, 512     # input / output (model) features
F_dim   = 1024         # FFN hidden dim
R       = 8            # LoRA rank
G       = 2            # GQA key groups
E       = 8            # number of experts
K_moe   = 512          # MoE expert dim
V       = 1024         # vocab size (logits)
C       = 1024         # corpus size (retrieval) -- 1024 so the kernel's output dim is 128-aligned
K_rot   = 64           # rotary projection out dim
P, Q_d  = 128, 128     # generic small matrix dims (outer product / quadratic form)


# ── Operations (each takes an `es` einsum callable) ───────────────────────────
# Original LLM-contraction set --------------------------------------------------

def batched_linear(x, W, *, es):
    """Dense projection of every token by a shared weight (x @ W.T)."""
    return es('bsi,oi->bso', x, W)


def attention_scores(Q, K, *, es):
    """Scaled dot-product attention scores, per head."""
    d = Q.shape[-1]
    return es('bshd,bthd->bsht', Q, K) / d ** 0.5


def attention_context(A, V_, *, es):
    """Attention context: weighted sum of value vectors."""
    return es('bsht,bthd->bshd', A, V_)


def output_projection(ctx, W_o, *, es):
    """Multi-head output projection (contracts head + head-dim axes)."""
    return es('bshd,hdo->bso', ctx, W_o)


def rope(Q, cos, sin, *, es):
    """Rotary positional embedding -- pure broadcast multiply (not a matmul)."""
    d = Q.shape[-1]
    Q1, Q2 = Q[..., :d // 2], Q[..., d // 2:]
    Q_rot = torch.cat([-Q2, Q1], dim=-1)
    return es('bshd,sd->bshd', Q, cos) + es('bshd,sd->bshd', Q_rot, sin)


def swiglu_ffn(x, W_gate, W_up, W_down, *, es):
    """SwiGLU gated FFN: SiLU(gate) * up, projected back down."""
    gate = es('bsi,fi->bsf', x, W_gate)
    up = es('bsi,fi->bsf', x, W_up)
    mid = F.silu(gate) * up
    return es('bsf,if->bsi', mid, W_down)


def gqa_scores(Q, K_gqa, *, es):
    """Grouped-query attention scores (expand G key groups to H heads)."""
    d = Q.shape[-1]
    H_, G_ = Q.shape[2], K_gqa.shape[2]
    K_exp = K_gqa.repeat_interleave(H_ // G_, dim=2)
    return es('bshd,bthd->bsht', Q, K_exp) / d ** 0.5


def lora_forward(x, W, A, B_mat, *, es):
    """Frozen base projection + low-rank LoRA residual (x@A@B)."""
    base = es('bsi,oi->bso', x, W)
    lora = es('bsr,ro->bso', es('bsi,ir->bsr', x, A), B_mat)
    return base + lora


def pairwise_similarity(queries, corpus, *, es):
    """Dot-product similarity of every query against every corpus embedding."""
    return es('bd,cd->bc', queries, corpus)


def moe_routing(x, W_route, W_expert, *, es):
    """Soft MoE: router softmax blends per-expert projections of the input."""
    weights = torch.softmax(es('bsi,ei->bse', x, W_route), dim=-1)
    expert_out = es('bsi,eio->bseo', x, W_expert)
    return es('bse,bseo->bso', weights, expert_out)


# Additional ops from llm_einsum_ops.js ----------------------------------------

def rotary_matrix(x, rot, *, es):
    """RoPE applied as a learned per-head rotation matrix (a real contraction)."""
    return es('bshd,hdk->bshk', x, rot)


def linear_dense(x, W, *, es):
    """Standard linear layer over the sequence (untransposed weight)."""
    return es('bsd,de->bse', x, W)


def batched_outer(a, b, *, es):
    """Per-batch outer product (low-rank / adapter factorisation)."""
    return es('bi,bj->bij', a, b)


def headwise_qkv(x, W, *, es):
    """Fused per-head Q/K/V projection in a single einsum."""
    return es('bsd,hde->bshe', x, W)


def lora_update(x_A, B_mat, *, es):
    """LoRA up-projection step on its own (B-matrix apply)."""
    return es('bsr,rd->bsd', x_A, B_mat)


def moe_dispatch(gates, expert_out, *, es):
    """Weighted combination of precomputed expert outputs."""
    return es('bse,bek->bsk', gates, expert_out)


def rms_norm(x, gamma, *, es):
    """Per-feature scaling -- broadcast multiply, not a matmul."""
    return es('bsd,d->bsd', x, gamma)


def diagonal_quadform(x, W, diag, *, es):
    """Quadratic form with a diagonal weight: xᵀ W diag, reduced to a scalar."""
    return es('bj,j->b', es('bi,ij->bj', x, W), diag)


def token_pos_embed(tok, pos, *, es):
    """Multiplicative token x position embedding -- broadcast multiply."""
    return es('bsd,sd->bsd', tok, pos)


def alibi_bias(bias, slopes, *, es):
    """ALiBi relative-bias scaling by per-head slope -- broadcast multiply."""
    return es('hqk,h->hqk', bias, slopes)


def logits(h, emb, *, es):
    """Project hidden states to vocabulary logits (weight tying)."""
    return es('bsd,vd->bsv', h, emb)


# ── pto-einsum callable with a persistent per-shape runner cache ──────────────
class PtoEinsum:
    """An ``es``-compatible callable backed by cached, pre-compiled NPU runners.

    A kernel is built and compiled once per (equation, shapes, dtype). The first
    call (correctness check and warmup) populates the cache, so the timed loop
    reuses the compiled runner. This mirrors the persistence provided by the
    reusable ``EinsumBuilder.build()`` API.
    """

    def __init__(self):
        self._cache = {}  # key -> (builder, runner)

    def __call__(self, equation, a, b):
        key = (equation, tuple(a.shape), tuple(b.shape), a.dtype)
        entry = self._cache.get(key)
        if entry is None:
            builder = EinsumBuilder(equation, [tuple(a.shape), tuple(b.shape)],
                                    a.dtype, device="npu")
            entry = (builder, builder.build())
            self._cache[key] = entry
        return entry[1](a, b)

    def cleanup(self):
        for builder, _ in self._cache.values():
            builder.cleanup()
        self._cache.clear()


def g(*shape):
    """Random fp32 NPU tensor of the given shape."""
    return torch.randn(*shape, dtype=torch.float32).npu()


# ── Registry: (label, displayed einsum, fn, input tensors) ────────────────────
def build_ops():
    return [
        ("batched_linear",       "bsi,oi->bso",       batched_linear,
         (g(B, S, I), g(O, I))),
        ("attention_scores",     "bshd,bthd->bsht",   attention_scores,
         (g(B, S, H, D), g(B, T, H, D))),
        ("attention_context",    "bsht,bthd->bshd",   attention_context,
         (g(B, S, H, T), g(B, T, H, D))),
        ("output_projection",    "bshd,hdo->bso",     output_projection,
         (g(B, S, H, D), g(H, D, O))),
        ("rope",                 "bshd,sd->bshd",     rope,
         (g(B, S, H, D), g(S, D), g(S, D))),
        ("swiglu_ffn",           "bsi,fi->bsf (x3)",  swiglu_ffn,
         (g(B, S, I), g(F_dim, I), g(F_dim, I), g(I, F_dim))),
        ("gqa_scores",           "bshd,bthd->bsht",   gqa_scores,
         (g(B, S, H, D), g(B, T, G, D))),
        ("lora_forward",         "bsi,ir->bsr,..",    lora_forward,
         (g(B, S, I), g(O, I), g(I, R), g(R, O))),
        ("pairwise_similarity",  "bd,cd->bc",         pairwise_similarity,
         (g(B, I), g(C, I))),
        ("moe_routing",          "bsi,eio->bseo,..",  moe_routing,
         (g(B, S, I), g(E, I), g(E, I, O))),
        ("rotary_matrix",        "bshd,hdk->bshk",    rotary_matrix,
         (g(B, S, H, D), g(H, D, K_rot))),
        ("linear_dense",         "bsd,de->bse",       linear_dense,
         (g(B, S, I), g(I, O))),
        ("batched_outer",        "bi,bj->bij",        batched_outer,
         (g(B, P), g(B, Q_d))),
        ("headwise_qkv",         "bsd,hde->bshe",     headwise_qkv,
         (g(B, S, I), g(H, I, D))),
        ("lora_update",          "bsr,rd->bsd",       lora_update,
         (g(B, S, R), g(R, O))),
        ("moe_dispatch",         "bse,bek->bsk",      moe_dispatch,
         (g(B, S, E), g(B, E, K_moe))),
        ("rms_norm",             "bsd,d->bsd",        rms_norm,
         (g(B, S, I), g(I,))),
        ("diagonal_quadform",    "bi,ij->bj; bj,j->b", diagonal_quadform,
         (g(B, P), g(P, Q_d), g(Q_d,))),
        ("token_pos_embed",      "bsd,sd->bsd",       token_pos_embed,
         (g(B, S, I), g(S, I))),
        ("alibi_bias",           "hqk,h->hqk",        alibi_bias,
         (g(H, S, T), g(H,))),
        ("logits",               "bsd,vd->bsv",       logits,
         (g(B, S, I), g(V, I))),
    ]


def _reason(exc):
    """A concise, human-readable cause from a build/compile exception."""
    text = str(exc)
    # Prefer the message of a failed C++ static_assert (the kernel's own
    # description of what it does not support), then any first 'error:' line.
    m = (re.search(r"requirement\s+'[^']*':\s*([^\n]+)", text)
         or re.search(r"error:\s*([^\n]+)", text))
    msg = m.group(1).strip() if m else text.splitlines()[0]
    return msg[:48]


def _bench(call):
    """Median wall-clock (ms) of `call` over RUNS, syncing the NPU each time."""
    samples = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        call()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main():
    pto = PtoEinsum()
    ops = build_ops()

    print("=" * 110)
    print(f"{'LLM CONTRACTION BENCHMARK -- pto-einsum vs torch.einsum (PyTorch), NPU fp32':^110}")
    print("=" * 110)
    print(f"Warmup: {WARMUP} | Runs: {RUNS} | Ops: {len(ops)}")
    print("-" * 110)
    print(f"{'op':<22}{'einsum':<22}{'out shape':<18}"
          f"{'rel diff':<11}{'pto ms':<10}{'torch ms':<10}{'speedup':<9}status")
    print("-" * 110)

    speedups = []
    n_ok = 0
    for label, eq, fn, inputs in ops:
        ref = fn(*inputs, es=torch.einsum)
        torch.npu.synchronize()

        try:
            out = fn(*inputs, es=pto)
            torch.npu.synchronize()
        except Exception as exc:  # build/codegen rejected the pattern
            print(f"{label:<22}{eq:<22}{'-':<18}{'-':<11}{'-':<10}{'-':<10}"
                  f"{'-':<9}unsupported ({_reason(exc)})")
            continue

        diff = (out.float() - ref.float()).abs().max().item()
        denom = ref.float().abs().max().item() + 1e-12
        rel = diff / denom
        ok = rel < 2e-2

        for _ in range(WARMUP):
            fn(*inputs, es=torch.einsum)
            fn(*inputs, es=pto)
        torch.npu.synchronize()

        t_pto = _bench(lambda: fn(*inputs, es=pto))
        t_torch = _bench(lambda: fn(*inputs, es=torch.einsum))
        speedup = t_torch / t_pto if t_pto > 0 else 0.0

        status = "OK" if ok else f"MISMATCH (rel={rel:.1e})"
        print(f"{label:<22}{eq:<22}{str(tuple(out.shape)):<18}"
              f"{rel:<11.2e}{t_pto:<10.4f}{t_torch:<10.4f}{speedup:<8.2f}x {status}")

        if ok:
            n_ok += 1
            speedups.append(speedup)

    print("-" * 110)
    if speedups:
        gmean = statistics.geometric_mean(speedups)
        print(f"Correct: {n_ok}/{len(ops)} supported & matching | "
              f"speedup vs PyTorch: geomean {gmean:.2f}x, "
              f"range [{min(speedups):.2f}x - {max(speedups):.2f}x]")
    print("=" * 110)

    pto.cleanup()


if __name__ == "__main__":
    main()
