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
See `docs/RUNTIME_PROTOCOL.md`. Full transformer compiler and real inference
remain unimplemented; follow `docs/ARCHITECTURE.md` for their separate boundaries.
The independent Python reference must not import the production worker.
