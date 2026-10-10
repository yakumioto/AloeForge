# P0 CPU validation

ENGINEERING INTERPRETATION — executed on Linux amd64 in Codex Cloud on
2026-10-10. Official Go 1.25.3, Python 3.12.14, pip 25.2, pytest 8.4.2 and
Ruff 0.14.0. This records bootstrap checks, not model or security acceptance.

## Commands and results

Development environment setup (all commands exited 0):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install pip==25.2
.venv/bin/python -m pip install -e './python[dev]'
```

The cloud image had an unrelated executable named `go`. The official Go
toolchain was installed under `/workspace/work/tools/go`; the check command was:

```sh
PATH="/workspace/work/tools/go/bin:$PATH" make check
```

Exit status: **0**. Final output:

```text
go vet ./...
.venv/bin/python -m ruff check python
All checks passed!
.venv/bin/python -m ruff format --check python
5 files already formatted
go test ./...
?       github.com/yakumioto/AloeForge/cmd/aloeforge      [no test files]
ok      github.com/yakumioto/AloeForge/internal/cli      (cached)
?       github.com/yakumioto/AloeForge/internal/version  [no test files]
.venv/bin/python -m pytest -q -c python/pyproject.toml python/tests
..................................                                       [100%]
34 passed in 0.06s
```

Go formatting validation also passed silently as the first `make lint` step.
The initial uncached `make check` passed: Go CLI tests in 0.002s, Python
34 passed in 0.08s. The final check followed the CI pinning and Makefile review.

Additional commands (each exited 0):

```sh
PATH="/workspace/work/tools/go/bin:$PATH" go list ./...
PATH="/workspace/work/tools/go/bin:$PATH" go mod tidy
PATH="/workspace/work/tools/go/bin:$PATH" go test -v ./internal/cli
.venv/bin/python -m pytest -q python/tests/reference
.venv/bin/python -m pytest -q python/tests/worker
PATH="/workspace/work/tools/go/bin:$PATH" go build -o /workspace/work/aloeforge ./cmd/aloeforge
/workspace/work/aloeforge version
/workspace/work/aloeforge inspect --help
git diff --check
```

Go CLI: 11 table cases passed, including invalid commands, extra arguments,
unimplemented inspection and static diagnostics without argument echo.
Reference and worker: 2 tests passed separately in each suite (0.02s each),
including isolated installed imports without the other module or PyTorch.
Version output: `aloeforge 0.1.0-dev`. Inspection help explicitly reports P0
is unimplemented and no model files are opened.

Additional local subprocess audits passed: four compiled CLI cases (including
expected exits 1 and 2), 23 ignored sensitive/generated paths and three permitted
source/documentation paths via `git check-ignore --no-index -q`, and no Rust
source in the checkout. No production schema or artifact validation is implied.

## Coverage and limitations

No CPU test was skipped. The entrypoint and version constant have no standalone
unit test files; the CLI unit tests and compiled-process smoke audit cover them.
Contract tests cover exact fields, unknown versions (including boolean version),
unknown/missing fields, false implementation status and populated placeholders.

GPU/CUDA, model downloads, numerical transforms, runtime protocol integration,
model-level regressions and security evaluation were not attempted and have no
implemented tests in P0. CI is configured to run the same `make check`; its
GitHub run result is reported separately in the PR. P1–P8 remain unimplemented.
No real credential, Client Secret, source weights, prompt or original input IDs
were used in these checks.
