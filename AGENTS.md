# AloeForge — repository instructions

- First-party implementation languages: **Go and Python**. Do not add Rust source or Rust build steps. Third-party binary wheels may internally use Rust.
- Read `docs/ARCHITECTURE.md`, `docs/METHOD_CONTRACT.md`, and the relevant module's contract before changes.
- Keep **trusted offline compiler**, **trusted client**, and **untrusted remote server** strictly separated. Never include source weights, Client Secret, original input IDs or plaintext prompts in server artifacts/logs.
- Token permutation is **not encryption**. No security claim without an explicit adversary model and reproducible evaluation.
- Treat AloePri V2 as the primary algorithm source. Label each specification PAPER, ENGINEERING INTERPRETATION, or DIAGNOSTIC. Never silently fill in ambiguous paper formulae.
- Unknown model architectures, Tensor shapes, quantization, layouts and attention variants MUST fail closed with explicit diagnostics.
- Full-V2 artifacts are custom runtime artifacts, not automatically vanilla Hugging Face or vanilla vLLM compatible.
- Exchange small Go/Python control messages; use validated files or bounded streams for Tensor data.
- All changes need focused tests, negative tests and exact commands/results. GPU tests cannot be reported passed without GPU evidence.
- One scoped Codex task per PR; do not automatically merge into main or modify the legacy Rust repository.
- Do not weaken acceptance thresholds to pass a test. Each transform requires independent math and model-level regression tests.
- Go: `go test ./...`, `go vet ./...`; Python: `pytest -q`, `ruff check`; CI enforces `make check`.
