# P6 preparation: controlled custom model and mock relay

DIAGNOSTIC / ENGINEERING INTERPRETATION — CPU-testable P6 preparation, not a
completed phase. P5's controlled Qwen2.5-0.5B quality and P4 method/human review
remain pending. Keep P5 publications on trusted systems. The separate GPU
experiment is a controlled synthetic test, never untrusted-server deployment.

## Model and loader

`aloeforge_runtime` shares P5's strict publication validator, but its PyTorch
numerical graph calls neither reference forward nor offline worker transforms.
It loads no source weights, tokenizer, original IDs, private maps or Client Secret.
Expected artifact/binding/tokenizer hashes must come from an external trusted
channel. Unknown identities, modified publication and inconsistent engine geometry
fail. The public HF engine descriptor stays **outside** publication; it is not a
vanilla HF checkpoint. All existing P1 filesystem/immutability assumptions apply.

Only P5's named noise-off sparse-isometry method and controlled diagnostic opt-in
are admitted. This does not resolve AloePri V2 ambiguities or admit security or
production profiles. Parameter bytes are capped at 4 GiB before module allocation;
mmap, temporary copies, framework buffers and cache are additional memory, so
this is not an RSS/GPU-memory guarantee. Missing/extra/duplicate/nonfinite/wrong
shape/dtype weights fail. Partially loaded graphs cannot execute.

The independent FP32 graph implements explicit logical head_dim separately from
expanded physical residual width; coefficient/epsilon-corrected norm, folded
gamma, QKV biases, split-half RoPE, GQA and SiLU follow P5. Physical IDs accept
int32 (vLLM's input buffer) or int64; positions are int64. Bounds, nonfinite logits,
embedding/pipeline intermediate inputs and unknown semantics fail.

`CPUAttention` is a single-session causal GQA mock with per-layer physical KV
state, contiguous positions and explicit reset. Discard/reset failed sessions;
it is not a concurrent server or PagedAttention. CPU linear/head operators use
fixed row order, <=64 output rows per block and balanced FP32 product sum trees;
each attention query uses its causal prefix. This avoids chunk-size-dependent
GEMM/GEMV reduction drift. Cross-reference cache checks still failed for wider
FP32 graphs under fresh diagnostic transforms. CPU mock admission is therefore
limited to head_dim=4/logical_hidden_size<=16; wider geometry fails explicitly.
This limits the mock, not P5's FP64 reference. The GPU candidate is opt-in and
unvalidated; projections remain batched PyTorch operators and must pass their
own hardware gate. CPU tests preserve full-logit atol=rtol=3e-4,
cache/split-logit atol=rtol=2e-6 and exact physical greedy.

## Candidate vLLM integration

The lazy, process-idempotent general-plugin entry point registers
`AloeForgeDenseForCausalLM` and `aloe_dense_diagnostic`. The custom
`BaseModelLoader.download_model/load_weights` verifies local artifacts only,
with no HF weight/download fallback. Public config scheduling geometry is checked
against publication. Rotated physical Q/K and V enter `vllm.attention.Attention`;
the scheduler forward context owns paged slots, causal metadata and request/cache
lifetime. The NumPy/mock cache is never substituted for a paged cache.

Candidate pins: vLLM **0.11.0**, PyTorch **2.8.0**, CUDA runtime **12.8**,
Python **3.12.14**, NumPy **2.3.3**. Registration/loader/Attention, explicit head_dim,
int32 input, Triton FP32 support and metric APIs were inspected at upstream tag
v0.11.0. These pins are **not a tested GPU stack**. Only PyTorch **2.8.0+cpu**
executed here. requirements-p6-gpu.txt is separate from CPU/base installation.

TP/PP != 1, non-FP32, quantization, LoRA, speculation, prefix caching, chunked
prefill, non-eager execution and non-auto cache dtype fail. Physical CUDA12.8 is
required by the candidate model. The provided runner selects V1/TRITON_ATTN
(head_dim >=32), skips tokenizer initialization/detokenization, remote code,
downloads and usage stats. It submits temperature-zero physical token requests.
The plugin is a graph/loader, not a hardened public vLLM/text service. API mocks
explicitly replace vLLM/CUDA and cannot establish engine/kernel correctness.

## Go mock relay

`gateway.New` preserves P3's fake script and fault timing. `NewRelay` adds a
trusted in-process `Backend.Runtime/Open(ctx,physicalRequest)` and pulled
`TokenStream.Next(ctx)/Close()`. Complete public runtime metadata must match.
Only **p3-fake-token-only** wire binding/profile is admitted. No Python IPC,
dense binding, real model or untrusted-server weight connection is implemented.

TLS/auth/schema/identity/ledger/eight slots/deadline/SSE limits remain P3's.
The relay copies mutable request fields, forwards physical prompt/stops and bound
once, validates returned IDs and never retries/prefetches. EOS wins over explicit
stop, then length. No extra pull occurs after stop/length. Close releases resources
on completion, error, invalid ID, timeout, disconnect and write failure (including
when Open returns a stream plus an error). Private error text never enters SSE.
Open/Next/Close are trusted synchronous code and MUST honor cancellation, return
promptly and never log requests. The gateway cannot terminate a misbehaving adapter;
it does not abandon one in an extra goroutine. A real bridge still needs reviewed
dense protocol/IPC, verified TLS client binding and engine abort mapping.

## External synthetic gate

`p6_validate_gpu.py` requires external trusted identities and diagnostic opt-in.
It bounds vocab<=512/head_dim>=32 and workload JSON<=64 KiB: 1–8 cases, 2–64
physical prompt IDs, 2–16 output bound, valid context and no unknown fields.
It requires complete prefill/decode log-probability parity at atol=rtol=3e-4,
exact physical greedy, physical stop termination, repeated-batch cache isolation
and actual FP32 CUDA per-layer cache geometry
`[blocks,2,block_size,kv_heads,head_dim]`. Log-probability parity cannot detect an
additive raw-logit offset; raw-logit parity remains a separate full-P6 gate.

After warmup, three identical workload batches measure engine TTFT/inter-token
TPOT, wall-clock output throughput and worker CUDA allocated/peak/reserved bytes.
The independent FP64 CPU reference uses the same prompts/bounds sequentially;
batching/device differences are explicit and do not establish matched-GPU speedup.
Missing metric/decode evidence fails. The report includes GPU device/capability;
record driver/package/workload provenance separately on the controlled runner.

The script suppresses Python/child stdout/stderr and prints only aggregates/cache
geometry or a static failure. Success distinguishes `synthetic_gpu_gate=passed`
from `gpu_gate=pending_real_model` and production blocked. CPU tests/--help cannot
pass a GPU gate. Full P6 remains pending P5 quality, actual GPU execution, real
model/cache-pressure/cancellation tests and real gateway/backend integration.
