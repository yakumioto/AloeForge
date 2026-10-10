# Codex Cloud playbook

1. Select the already configured Codex Cloud environment for `yakumioto/AloeForge`.
2. Checkout this Starter Kit PR once it is merged (or use its branch deliberately).
3. Give Codex the contents of `codex/tasks/00-bootstrap.md`. Complete exactly one task in one PR, never directly on main.
4. Initial environment should provide pinned Go, Python 3.12, pytest and Ruff. P0 must be CPU-only without model downloads, CUDA or vLLM.
5. Require actual `make check` output in the PR, with skipped tests named explicitly.
6. Run real model, vLLM and CUDA integration on a separate explicitly configured GPU runner.
7. Never upload credentials, source-model IP or real Client Secrets to public code, CI logs, or cloud task prompts.
8. After P0 merge, continue P1–P8 in dependency order in `docs/ROADMAP.md`.

## P0 reproducible CPU setup

ENGINEERING INTERPRETATION — use official Go 1.25.3 (`go version` must report
`go1.25.3`, not an unrelated executable named `go`), Python 3.12.14 and GNU Make.
Put the official Go `bin` directory on `PATH` to provide both `go` and `gofmt`.
Use the same versions as `.github/workflows/cpu.yml`.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install pip==25.2
.venv/bin/python -m pip install -e './python[dev]'
make check
```

No additional credential, model download, GPU, CUDA, PyTorch, vLLM or Rust
toolchain is required. `make check` performs no network operations once the
development tools are installed. The Python runtime packages have no dependencies.
Set `GO`, `GOFMT` and `PYTHON` when tool executables are outside `PATH` or when
using an alternate virtual environment. Tests import installed packages rather
than relying on a `PYTHONPATH` override.

When PR #1 is unmerged, P0 uses `codex/p0-bootstrap` derived from
`codex/starter-kit` with a separate PR targeting `codex/starter-kit`. Retarget to
`main` after the Starter Kit merges. Do not change or merge the documentation PR
as part of P0. Record actual results in the P0 PR; no GPU gate is claimed.
