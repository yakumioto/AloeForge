# AloeForge

AloeForge is a **Go + Python** project for researching and implementing AloePri V2-style privacy-preserving LLM compilation and inference.

**Status:** P5 diagnostic Qwen2.5 dense Go/Python compiler and independent CPU
artifact runtime, including strict tensor roles, bounded offline worker files,
external secrets, expanded residual coordinates, prefill/decode and greedy.
P1–P4 inventory, mapping, fake TLS/SSE and independent math remain available.
The Qwen2.5-0.5B real-model quality gate and P4 method review remain pending;
production/full AloePri V2, privacy guarantees and GPU/vLLM are not admitted.

## Design

- **Go:** compiler orchestration, model discovery, transform planning, artifact verification, secret lifecycle, client SDK and HTTP gateway.
- **Python/PyTorch:** trusted offline tensor transform worker, independent numerical reference and vLLM inference adapter.
- **Protocol:** client-side tokenization and permutation; transport obfuscated token IDs; server executes an obfuscated model; client restores output.

## Local development (Linux, CPU only)

Use Go **1.25.3**, Python **3.12.14** and GNU Make. Python packaging pins
Hatchling 1.27.0, pytest 8.4.2 and Ruff 0.14.0; CI pins pip 25.2.
Go uses the standard library plus pinned `golang.org/x/sys` v0.37.0 for Linux
locking and no-replace rename. Base Python packages have no runtime dependencies;
P4/P5 reference/worker/dev extras pin NumPy 2.3.3. PyTorch is not required for CPU CI.
Setup downloads development dependencies/tools only, never models.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install pip==25.2
.venv/bin/python -m pip install -e './python[dev]'
make check
go run ./cmd/aloeforge version
go run ./cmd/aloeforge inspect --help
go run ./cmd/aloeforge verify --help
go run ./cmd/aloeforge compile-token --help
go run ./cmd/aloeforge verify-token --help
go run ./cmd/aloeforge compile-dense --help
go run ./cmd/aloeforge verify-dense --help
```

`make lint` checks Go formatting/vet and Python Ruff lint/format.
`make test` runs Go unit tests and Python pytest; `make check` runs both.
The independent reference and worker smoke tests can also run separately:

```sh
.venv/bin/python -m pytest -q python/tests/reference
.venv/bin/python -m pytest -q python/tests/worker
```

`inspect --model-root PATH` checks local HF/safetensors metadata without reading
payloads and prints counts, known architecture identity and inventory hash.
`verify --artifact-root PATH` streams hashes of a P1/P2/P5 diagnostic artifact and prints
its kind/count/hash. Verification/I/O failures return exit 1; invalid CLI usage
returns exit 2. Paths, tensor names and raw metadata are not echoed.
See [`docs/ARTIFACT_CONTRACT.md`](docs/ARTIFACT_CONTRACT.md) for supported identities,
limits, source-content binding, stage API, recovery and Linux filesystem needs.
P1 artifacts are diagnostic-only and must not be sent to an untrusted server.

`compile-token` accepts only the explicit synthetic F32 embedding/head fixture
schema in [`docs/TOKEN_BASELINE_CONTRACT.md`](docs/TOKEN_BASELINE_CONTRACT.md).
Source, output and Client Secret directories must be disjoint; output/secret
parents must be private (0700). Secret material is never printed or published.
Use `--resume` to reload the saved secret and verified checkpoint.
`verify-token` additionally requires the expected artifact ID and tokenizer hash
from the trusted compiler/client channel. P2 artifacts also remain diagnostic-only.
See [`docs/P2_VALIDATION.md`](docs/P2_VALIDATION.md) for commands and actual evidence.

`compile-dense` accepts the exact dense Qwen2 schema and only the explicit
noise-off isometry diagnostic revision. It invokes a selected offline Python
interpreter, validates worker output in Go and publishes through P1's atomic path.
`verify-dense` checks external artifact/tokenizer identities and Client Secret
binding. The independent runtime loads publication alone without source or secret.
Read [`docs/DENSE_COMPILER_CONTRACT.md`](docs/DENSE_COMPILER_CONTRACT.md) and
[`docs/P5_VALIDATION.md`](docs/P5_VALIDATION.md) for usage, fixed gates and pending
controlled Qwen2.5-0.5B evaluation. Keep P5 diagnostic artifacts on trusted systems.

P3 libraries and exact wire/API/limits are documented in
[`docs/RUNTIME_PROTOCOL.md`](docs/RUNTIME_PROTOCOL.md). Applications supply locally
tokenized IDs and exact byte pieces; no text round-trip fallback or real model
backend exists. Configure the fake gateway with public metadata and physical IDs
only; keep all P1/P2 diagnostic weights and Client Secrets on the trusted side.
Use the gateway's TLS server helper and a client with verified certificates.
Exercise the complete local CPU slice without downloading a model:

```sh
go test -count=1 ./internal/clientruntime ./internal/gateway ./internal/runtimeprotocol
```

See [`docs/P3_VALIDATION.md`](docs/P3_VALIDATION.md) for actual end-to-end,
negative, race and fuzz evidence. Request replay protection is bounded and
process-local; partial streams are explicit and never automatically retried.

P4 formula audit, proofs, ambiguities and fixed numeric gates are in
[`docs/V2_MATH_AUDIT.md`](docs/V2_MATH_AUDIT.md). The oracle's `production` and
unqualified `paper-reference` profiles reject unresolved choices. Noise-off and
explicit supplied-noise modes are separate; empirical approximation drift is not
reported as floating-point covariance. Applications can install only the math
extra with `pip install -e './python[reference]'`. Run its independent CPU gates:

```sh
.venv/bin/python -m pytest -q python/tests/reference/test_v2_math.py
.venv/bin/python python/scripts/p4_measure_math.py
```

See [`docs/P4_VALIDATION.md`](docs/P4_VALIDATION.md) for measured errors, commands,
review status and exclusions. BF16 is software conversion with an FP32 carrier,
not a BF16 compute kernel or GPU validation.

## Codex Cloud

Read [`AGENTS.md`](AGENTS.md), all engineering documents under `docs/`, and
[`codex/tasks/00-bootstrap.md`](codex/tasks/00-bootstrap.md). Select the configured
environment and checkout the intended branch; the Starter Kit documents already
live in the repository and need no archive extraction. Follow
[`docs/CODEX_CLOUD.md`](docs/CODEX_CLOUD.md) for pinned tool setup. Run the same
`make check` locally and in CPU CI. Submit one scoped PR with exact commands,
exit statuses, changed files, skipped tests and limitations. P0 is merged into
`main`; start each authorized task from current `main` in a new branch and submit
one scoped PR. Never merge automatically or advance beyond the authorized task.

## Package and trust boundaries

`cmd/aloeforge` delegates to `internal/cli`; `internal/version` owns build
identification. `python/src/aloeforge_reference` is the independent numerical
oracle boundary, and `python/src/aloeforge_worker` is the trusted offline transform
boundary. They are importable and separately testable and do not import each other.
The reference package includes the independent P2 synthetic mapping/logit oracle;
the worker still implements no transforms. The separate P4 `keymat`/`v2` modules
use NumPy, never import the worker, and accept only bounded diagnostic toy arrays.
P3 adds `internal/clientruntime`,
`internal/gateway` and `internal/runtimeprotocol` with a fake backend only.
P2 adds `internal/tokenbaseline` atop the P1 inventory/artifact packages.

The trusted offline compiler/worker may access source weights and private
material. The trusted client owns Client Secret, tokenizer, plaintext prompts
and original input IDs. The untrusted remote server may receive only published
obfuscated artifacts and public binding metadata; source weights, Client Secret,
plaintext prompts and original input IDs must never appear in its artifacts or
logs. `.gitignore` protects common private inputs and generated paths; developers
must still keep sensitive data outside tracked files and cloud prompts.

Go/Python control messages must be small, versioned and size bounded. Tensors
move through validated files, mmap or bounded streams, never JSON payloads.
P1 validates paths and structural shapes/dtypes; semantic layout admission and all
IPC remain future work. See
[`docs/BOOTSTRAP_CONTRACTS.md`](docs/BOOTSTRAP_CONTRACTS.md) for the four versioned,
explicitly unimplemented contract drafts and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the intended boundaries.

**Security:** Do not upload production keys, private source-model weights or plaintext prompts to an untrusted server or public repo. Token permutation is not encryption.

## References

- AloePri V2: https://arxiv.org/abs/2603.01499v2
- Previous Rust research code (reference only): https://github.com/yakumioto/AloePri-Model-Compiler
- Community implementation (verify method revision): https://github.com/sheng1feng/Aloepri
