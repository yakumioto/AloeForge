# AloeForge

AloeForge is a **Go + Python** project for researching and implementing AloePri V2-style privacy-preserving LLM compilation and inference.

**Status:** P0 CPU-only Go/Python bootstrap. CLI help/version and importable package boundaries are implemented. Model inspection, transformations, IPC and inference remain unimplemented. No complete AloePri V2 implementation or cryptographic privacy claim.

## Design

- **Go:** compiler orchestration, model discovery, transform planning, artifact verification, secret lifecycle, client SDK and HTTP gateway.
- **Python/PyTorch:** trusted offline tensor transform worker, independent numerical reference and vLLM inference adapter.
- **Protocol:** client-side tokenization and permutation; transport obfuscated token IDs; server executes an obfuscated model; client restores output.

## Local development (Linux, CPU only)

Use Go **1.25.3**, Python **3.12.14** and GNU Make. Python packaging pins
Hatchling 1.27.0, pytest 8.4.2 and Ruff 0.14.0; CI pins pip 25.2.
Go uses only the standard library, and the Python packages have no runtime
dependencies. Setup downloads development tools only, never models.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install pip==25.2
.venv/bin/python -m pip install -e './python[dev]'
make check
go run ./cmd/aloeforge version
go run ./cmd/aloeforge inspect --help
```

`make lint` checks Go formatting/vet and Python Ruff lint/format.
`make test` runs Go unit tests and Python pytest; `make check` runs both.
The independent reference and worker smoke tests can also run separately:

```sh
.venv/bin/python -m pytest -q python/tests/reference
.venv/bin/python -m pytest -q python/tests/worker
```

`inspect` without exactly `--help` or `-h` returns exit 1 with a static
unimplemented diagnostic and opens no model files. Invalid CLI usage returns
exit 2. Arguments are not echoed into diagnostics.

## Codex Cloud

Read [`AGENTS.md`](AGENTS.md), all engineering documents under `docs/`, and
[`codex/tasks/00-bootstrap.md`](codex/tasks/00-bootstrap.md). Select the configured
environment and checkout the intended branch; the Starter Kit documents already
live in the repository and need no archive extraction. Follow
[`docs/CODEX_CLOUD.md`](docs/CODEX_CLOUD.md) for pinned tool setup. Run the same
`make check` locally and in CPU CI. Submit one scoped PR with exact commands,
exit statuses, changed files, skipped tests and limitations. When Starter Kit
PR #1 is pending, use a separate P0 branch based on `codex/starter-kit` and target
that branch so the implementation diff stays separate. After #1 merges, retarget
the P0 PR to `main`. Never merge automatically. P1–P8 begin only after P0 review.

## Package and trust boundaries

`cmd/aloeforge` delegates to `internal/cli`; `internal/version` owns build
identification. `python/src/aloeforge_reference` is the independent numerical
oracle boundary, and `python/src/aloeforge_worker` is the trusted offline transform
boundary. They are importable and separately testable but implement no math and
do not import each other. There is no remote runtime package in P0.

The trusted offline compiler/worker may access source weights and private
material. The trusted client owns Client Secret, tokenizer, plaintext prompts
and original input IDs. The untrusted remote server may receive only published
obfuscated artifacts and public binding metadata; source weights, Client Secret,
plaintext prompts and original input IDs must never appear in its artifacts or
logs. `.gitignore` protects common private inputs and generated paths; developers
must still keep sensitive data outside tracked files and cloud prompts.

Go/Python control messages must be small, versioned and size bounded. Tensors
move through validated files, mmap or bounded streams, never JSON payloads.
Path/shape/dtype/layout validation and all IPC remain future work. See
[`docs/BOOTSTRAP_CONTRACTS.md`](docs/BOOTSTRAP_CONTRACTS.md) for the four versioned,
explicitly unimplemented contract drafts and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the intended boundaries.

**Security:** Do not upload production keys, private source-model weights or plaintext prompts to an untrusted server or public repo. Token permutation is not encryption.

## References

- AloePri V2: https://arxiv.org/abs/2603.01499v2
- Previous Rust research code (reference only): https://github.com/yakumioto/AloePri-Model-Compiler
- Community implementation (verify method revision): https://github.com/sheng1feng/Aloepri
