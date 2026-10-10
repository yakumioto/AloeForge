# Internal package boundaries

ENGINEERING INTERPRETATION — `cli` dispatches version, structural inventory and
diagnostic verification; `version` owns build identification. `cmd/aloeforge`
wires process I/O and exit status. P1 adds `strictjson` (bounded JSON),
`safetensors` (header validation), `model` (HF inventory and private source binding)
and `artifact` (staging, recovery, verification and atomic Linux publication).

P2 adds `tokenbaseline` for strictly admitted synthetic F32 embedding/head row
mapping, external Client Secret lifecycle and diagnostic client binding checks.
`cli` adds `compile-token` / `verify-token`; `artifact` adds optional public P2
binding metadata and finite-F32 verification, preserving P1 encodings without it.

See `docs/ARTIFACT_CONTRACT.md` and `docs/TOKEN_BASELINE_CONTRACT.md`.
P3 adds `runtimeprotocol` (strict public wire types), `clientruntime` (trusted Go
library owning mapping and local UTF-8/stop handling), and `gateway` (TLS/SSE
handler with a fixed physical-ID fake backend and no Secret/tokenizer/weight access).
See `docs/RUNTIME_PROTOCOL.md`. P5 adds `dense` for the registered strict Qwen2.5
diagnostic graph, private plan/secret, bounded offline worker files and P1
publication, plus `compile-dense` / `verify-dense`. `artifact` admits a separate
strict finite-F32 dense contract. Follow `docs/DENSE_COMPILER_CONTRACT.md`;
production and the physical GPU/vLLM backend remain unimplemented.
The independent Python reference must not import the production worker.

P6 preparation adds gateway Backend/TokenStream and NewRelay, restricted to P3
mock runtime metadata. No dense wire binding or Python IPC is admitted. Backend
calls must honor context cancellation and never log physical requests; the gateway
pulls synchronously and closes stream resources on stop/failure/disconnect.
See docs/VLLM_BACKEND_CONTRACT.md for this boundary and unvalidated GPU candidate.
