# Internal package boundaries

ENGINEERING INTERPRETATION — P0 implements `cli` (argument dispatch and help)
and `version` (build identification) only. `cmd/aloeforge` wires process I/O
and exit status; tests call the CLI without starting a process.

Future compiler, client and gateway packages remain unimplemented. They must
have separate entry points and dependencies following `docs/ARCHITECTURE.md`.
The independent Python reference must not import the production worker.
