# Codex Cloud playbook

1. Select the already configured Codex Cloud environment for `yakumioto/AloeForge`.
2. Checkout current `main`; P0 is merged. Create a new task branch.
3. Give Codex the next authorized task from `codex/tasks/`. Complete exactly one task in one PR, never directly on main.
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
development tools are installed. Base Python runtime packages have no dependencies; P4 reference/dev extras pin
NumPy 2.3.3.
Set `GO`, `GOFMT` and `PYTHON` when tool executables are outside `PATH` or when
using an alternate virtual environment. Tests import installed packages rather
than relying on a `PYTHONPATH` override.

P1 uses pinned `golang.org/x/sys` v0.37.0 for Linux filesystem primitives. Run
`go mod download` during setup if the module cache is empty; checks are offline
once tools/dependencies are installed. Read `docs/ARTIFACT_CONTRACT.md` before
changes to inventory, staging or publication. Generate synthetic sparse fixtures
with `python/scripts/p1_synthetic_inventory.py` in a new `work/` directory and
measure them with `python/scripts/p1_measure_rss.py`; no models are downloaded.
Real models, GPU and P2–P8 tests remain outside P1. Record exact commands,
limitations and RSS evidence in the PR.

P2 adds standard-library-only synthetic token/logit oracle tests and real Go CLI
integration in Python. Put the pinned Go executable on `PATH` for those tests,
including when setting Make's `GO`/`GOFMT` overrides. Read
`docs/TOKEN_BASELINE_CONTRACT.md`; CPU CI also runs bounded permutation fuzzing.
P2 uses no model downloads or GPU/transformer runtimes. Keep generated model,
output and external Client Secret directories disjoint and private where required.

P3 adds verified localhost TLS tests using ephemeral test certificates, a fixed
fake-ID script and local byte-piece fixtures. No credentials/models are fetched.
Read `docs/RUNTIME_PROTOCOL.md`; CPU CI includes SSE and incremental UTF-8 fuzzing
alongside full race/header/permutation checks. Keep the fake gateway free of
P1/P2 diagnostic weight artifacts, plaintext tokenization and Client Secret access.

P4 reference/dev extras pin NumPy 2.3.3; setup's existing editable dev install
supplies it automatically. Base/worker runtime dependency lists remain empty.
Read `docs/V2_MATH_AUDIT.md` before mathematical changes. `make check` includes
independent scalar-baseline P4 tests; CPU CI also runs
`python/scripts/p4_measure_math.py` to record only seeds, counts and scalar errors.
No weights/models/keys are printed or downloaded, and no Torch/CUDA/GPU toolchain
is installed. Preserve declared tolerances; unresolved method choices block
production even when diagnostic covariance tests pass.
