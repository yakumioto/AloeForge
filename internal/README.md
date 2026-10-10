# Internal package boundaries

ENGINEERING INTERPRETATION — `cli` dispatches version, structural inventory and
diagnostic verification; `version` owns build identification. `cmd/aloeforge`
wires process I/O and exit status. P1 adds `strictjson` (bounded JSON),
`safetensors` (header validation), `model` (HF inventory and private source binding)
and `artifact` (staging, recovery, verification and atomic Linux publication).

See `docs/ARTIFACT_CONTRACT.md`. Transform compiler, client and gateway remain
unimplemented. They must
have separate entry points and dependencies following `docs/ARCHITECTURE.md`.
The independent Python reference must not import the production worker.
