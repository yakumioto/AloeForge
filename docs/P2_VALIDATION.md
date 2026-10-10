# P2 CPU validation

DIAGNOSTIC — executed on Linux amd64 in Codex Cloud on 2026-10-10, based on
merged P1 PR #5 (`f136a28`). Go 1.25.3, Python 3.12.14, pytest 8.4.2,
Ruff 0.14.0; no dependency changes, models, credentials or GPU were used.
Generated Client Secrets belong only to synthetic test fixtures and are never
included in this record or PR. Public binding/artifact hashes are recorded below.

## Required checks

All these commands actually ran and exited **0**:

```sh
PATH="/workspace/work/tools/go/bin:$PATH" make check
PATH="/workspace/work/tools/go/bin:$PATH" go test -race -count=1 ./...
PATH="/workspace/work/tools/go/bin:$PATH" go test ./internal/tokenbaseline -run '^$' -fuzz '^FuzzPermutationInverse$' -fuzztime=3s -parallel=2
PATH="/workspace/work/tools/go/bin:$PATH" go test ./internal/safetensors -run '^$' -fuzz '^FuzzHeader$' -fuzztime=3s -parallel=2
git diff --check
```

Final `make check`: Go formatting/vet and all Go package tests passed. Ruff lint
passed; **10 Python files** formatted. **76 Python tests passed in 0.58s**, no
skips (47 prior tests plus 29 new oracle/integration cases). Main/version have no
standalone test files; CLI tests exercise their behavior.

Final uncached race results: artifact 1.166s, CLI 1.014s, model 1.043s,
safetensors 1.017s, strictjson 1.009s, tokenbaseline 1.215s; all passed.

Bounded fuzz campaigns: permutation inverse **83,744 executions**, 22 new
interesting inputs, PASS (3.597s); safetensors **178,821 executions**, 26 new
interesting inputs, PASS (3.113s). These are smoke campaigns, not exhaustive
validation. CPU CI keeps `make check` and race checks and adds the permutation
campaign alongside the existing header campaign. The final GitHub run is linked
in the PR.

## Independent oracle and negative gates

The Python reference imports no Go/worker implementation or tensor framework.
It independently implements inverse lookup, physical row mapping, special-ID
mapping, canonical binding commitment and FP64 embedding/head dot products.
Integration tests execute the real Go CLI on independently generated tiny
safetensors, then check public artifact/descriptor/file hashes and private
binding commitment. For **all four input tokens**, tied and untied output logits
equal source logits only after restoring vocabulary coordinates.

Known untied source logits for original input 1 are `[7, -3, -12.5, 2]`;
with fixed test permutation `[2, 0, 3, 1]`, physical logits are
`[-3, 2, 7, -12.5]`. Tied source logits are `[1, 10, -2.5, -6.25]`.
These exact binary-representable fixture values use equality, not loosened
tolerances. An independently computed JSON/hashlib golden commitment is
`f7cd207dc66f23b4e7e160a9327b974f90b2f962c2a6dc6ba2d88ebfc6432d0c`.
Golden material is deliberately synthetic; production generation uses Go
`crypto/rand` and persisted state rather than a test permutation or seeded RNG.

Negative cases cover duplicate/missing/out-of-range permutations, token/logit
bounds, malformed nonce and private JSON, version/method/RNG mismatch, missing or
unknown special roles, multiple EOS and BOS/PAD aliases, unsupported fixture
kind/shape/dtype/extra tensors, duplicate/oversized tokenizer entries, NaN/Inf,
overlapping/symlink-aliased directories, secret symlinks/FIFOs/permissions/length,
and secret/source/tokenizer/checkpoint changes. Fresh compilation never replaces
an existing secret or stage. Invalid fixtures save no secret and publish nothing.
Client verification rejects wrong externally expected identities, a different
valid secret and altered mapped special IDs. P2 artifact verification rejects
unknown kinds/binding fields and nonfinite payloads, including output written
through the lower-level P1 staging API. P1 descriptor/hash goldens still pass.

Three actual child-process exits at code 77 (without closing kernel locks)
exercise recovery after secret persistence, stage creation and a completed shard
checkpoint. Recovery retains the exact secret bytes and client/artifact binding.
Repeated completed resume skips shard rewriting and returns the same artifact ID.
Existing P1 partial/orphan/no-overwrite/crash tests also run. Concurrent secret
creation permits exactly one successful atomic write; race checks pass.

## Actual compiled CLI exercise

The build and fixture creation below exited 0. Use a fresh parent when repeating:

```sh
PATH="/workspace/work/tools/go/bin:$PATH" go build -o /workspace/work/aloeforge-p2 ./cmd/aloeforge
.venv/bin/python - <<'PY'
import runpy
from pathlib import Path
fixtures = runpy.run_path('python/tests/reference/test_token_baseline.py')
parent = Path('/workspace/work/p2-cli-demo')
parent.mkdir(mode=0o700, exist_ok=False)
fixtures['fixture'](parent, False)
PY
/workspace/work/aloeforge-p2 compile-token --model-root /workspace/work/p2-cli-demo/source --output-parent /workspace/work/p2-cli-demo/output --secret-root /workspace/work/p2-cli-demo/secret --stage .stage-demo --target visible
/workspace/work/aloeforge-p2 verify --artifact-root /workspace/work/p2-cli-demo/output/visible
```

Both CLI commands exited **0**, kind `p2-token-permutation-diagnostic-only`,
vocabulary 4, one shard with two `[4,2]` F32 matrices. Actual public identity:

- Artifact: `febb529d41d6acb64f92abe9230eb77d7d29904dc36dc68f0ca8a743b24e1037`
- Binding: `8ee47981d773e69154854d5252a3d5684fdb102b2887fd92cf9588fe2b97e82b`
- Tokenizer: `bfc412ab8dad0110efd12ab85a968385b3707643c16ba99a8742fc4bcfdea583`

Fresh runs deliberately create different random secrets and artifact/binding IDs;
use the new compiler summary for the externally expected artifact ID. The
tokenizer bytes/fingerprint remain stable for this fixture.

These checks on the recorded artifact exited **0**; rebuilding after the final
durability change and repeating resume/client verification also exited 0:

```sh
/workspace/work/aloeforge-p2 verify-token --artifact-root /workspace/work/p2-cli-demo/output/visible --secret-root /workspace/work/p2-cli-demo/secret --artifact-id febb529d41d6acb64f92abe9230eb77d7d29904dc36dc68f0ca8a743b24e1037 --tokenizer-sha256 bfc412ab8dad0110efd12ab85a968385b3707643c16ba99a8742fc4bcfdea583
/workspace/work/aloeforge-p2 compile-token --model-root /workspace/work/p2-cli-demo/source --output-parent /workspace/work/p2-cli-demo/output --secret-root /workspace/work/p2-cli-demo/secret --stage .stage-demo --target visible --resume
```

Replacing expected artifact ID with 64 zeroes exited **1**, stdout empty and
stderr `token baseline failed: client artifact identity mismatch`. No raw paths,
tokenizer strings, original IDs, permutation or nonce were printed. Public output
contains only `manifest.json` and `model.safetensors`; secret/private controls
remain in separate directories.

## Skips and limits

No CPU tests were skipped. Real SmolLM2/HF tokenizer/model regressions, transformer
attention/generation, model quality, GPU/CUDA/vLLM and attack/privacy evaluation
were not attempted. This task uses synthetic/tiny fixtures as allowed by task 02;
it does not admit real-model semantic compatibility. P3–P8 remain unimplemented.
No Rust, model downloads or legacy-repository edits were introduced.

Token permutation is not encryption or full AloePri V2. Diagnostic weights must
never be sent to an untrusted server. Expected artifact/tokenizer identities must
come from a trusted channel; hashes do not prove an external signer's identity.
Private directories, immutable inputs and trusted Linux/local filesystem support
are required. Process-death recovery does not certify physical power-loss or
hostile filesystem tolerance. See `TOKEN_BASELINE_CONTRACT.md` for the schema,
limits, commitment encoding and exact coordinate convention.
