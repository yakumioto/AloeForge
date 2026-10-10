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
Full transformer compiler, client runtime and gateway remain unimplemented. They must
have separate entry points and dependencies following `docs/ARCHITECTURE.md`.
The independent Python reference must not import the production worker.
