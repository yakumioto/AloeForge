# P1 CPU validation

ENGINEERING INTERPRETATION — executed on Linux amd64 in Codex Cloud on
2026-10-10. Go 1.25.3, Python 3.12.14, pytest 8.4.2, Ruff 0.14.0,
`golang.org/x/sys` v0.37.0. No models, credentials, Client Secrets or GPU were used.

## Exact checks and results

Dependency setup and normalization (exit 0):

```sh
PATH="/workspace/work/tools/go/bin:$PATH" go get golang.org/x/sys@v0.37.0
PATH="/workspace/work/tools/go/bin:$PATH" go mod tidy
```

Final required check (exit **0**):

```sh
PATH="/workspace/work/tools/go/bin:$PATH" make check
```

```text
go vet ./...
.venv/bin/python -m ruff check python
All checks passed!
.venv/bin/python -m ruff format --check python
7 files already formatted
go test ./...
?       github.com/yakumioto/AloeForge/cmd/aloeforge       [no test files]
ok      github.com/yakumioto/AloeForge/internal/artifact   0.029s
ok      github.com/yakumioto/AloeForge/internal/cli        0.003s
ok      github.com/yakumioto/AloeForge/internal/model      (cached)
ok      github.com/yakumioto/AloeForge/internal/safetensors (cached)
ok      github.com/yakumioto/AloeForge/internal/strictjson  (cached)
?       github.com/yakumioto/AloeForge/internal/version    [no test files]
.venv/bin/python -m pytest -q -c python/pyproject.toml python/tests
..................................                                       [100%]
34 passed in 0.05s
```

Go formatting passed silently. Full uncached race check and bounded fuzzing
also exited **0**:

```sh
PATH="/workspace/work/tools/go/bin:$PATH" go test -race ./...
PATH="/workspace/work/tools/go/bin:$PATH" go test ./internal/safetensors -run '^$' -fuzz '^FuzzHeader$' -fuzztime=3s -parallel=2
```

Race results: artifact 1.141s, CLI 1.014s, model 1.029s, safetensors 1.016s,
strictjson 1.012s; all passed. Fuzzing: 190,346 executions in 3 seconds,
63 new interesting inputs, PASS (3.025s). This is a bounded smoke campaign,
not exhaustive proof. CPU CI runs `make check`, the race check and the same
bounded fuzz command; its GitHub run is linked in the PR.

```sh
PATH="/workspace/work/tools/go/bin:$PATH" go test -json ./... > /workspace/work/p1-go-tests.jsonl
PATH="/workspace/work/tools/go/bin:$PATH" go list ./...
git diff --check
```

All exited 0. JSON events recorded 27 passing top-level tests/targets and 100
passing nested cases (including default fuzz seeds). Negative cases include
malformed JSON/Unicode, duplicate keys, unsupported identities/config variants,
traversal/escaping symlinks/FIFOs, invalid offsets/shapes/dtypes/lengths, overlaps,
holes, missing/extra/contradictory shards/index entries, wrong content hashes,
unexpected metadata, mismatched source-content/method/RNG/checkpoint bindings,
corrupt restart states, concurrent writes and targets that must never be replaced.
Source-content tests change payload bytes while preserving metadata: inventory
identity stays unchanged, content binding changes. Process-death/restart tests
cover six states; orphan/partial-copy states are deliberately synthesized.

## Sparse synthetic inventory and peak RSS

Build and fixture creation commands (all exit 0; roots must not already exist):

```sh
PATH="/workspace/work/tools/go/bin:$PATH" go build -o /workspace/work/aloeforge-p1 ./cmd/aloeforge
.venv/bin/python python/scripts/p1_synthetic_inventory.py --root /workspace/work/p1-multishard --shards 8 --tensors-per-shard 4096 --payload-bytes-per-shard 536870912
.venv/bin/python python/scripts/p1_synthetic_inventory.py --root /workspace/work/p1-large-header --shards 1 --tensors-per-shard 30000 --payload-bytes-per-shard 30720000 --name-padding 96
```

The fixtures contain zero-valued sparse payloads, not downloaded models. The
multishard case declares 4,294,967,296 payload bytes and 3,066,624 header bytes.
The large-header case declares 30,720,000 payload bytes and a 5,558,304-byte header.

```sh
.venv/bin/python python/scripts/p1_measure_rss.py --binary /workspace/work/aloeforge-p1 --root /workspace/work/p1-multishard
.venv/bin/python python/scripts/p1_measure_rss.py --binary /workspace/work/aloeforge-p1 --root /workspace/work/p1-large-header
```

Each measurement starts a fresh Python process with exactly one inspection child;
Linux `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` reports that child's peak in
KiB. Generation/build are excluded. Both final measurements exited 0:

| Fixture | Shards / tensors | Final elapsed | Final peak RSS | Earlier observation |
| --- | --- | --- | --- | --- |
| 4 GiB sparse multishard | 8 / 32,768 | 0.502s | 34,092 KiB | 35,256 KiB, 0.480s |
| Large header | 1 / 30,000 | 0.633s | 52,332 KiB | 78,244 KiB, 0.568s |

RSS varies with GC and scheduling; these are measurements, not hard limits or
real-model RSS guarantees. Inspection reads headers only; a focused ReaderAt
test panics on any payload access and passes with a declared 1 GiB payload.
`BindSource`/artifact verification additionally stream payload hashes; those
operations are not represented by the header-only RSS measurements.

## Actual compiled verification command

A synthetic one-shard artifact was created through the Go stage API using a
scratch program (`work/p1_demo.go`, excluded from version control). It supplied
two bytes `ab`, one U8 output of shape `[2]`, expected SHA-256, and synthetic
all-zero source fingerprints; these mocks are not real source bindings.

```sh
PATH="/workspace/work/tools/go/bin:$PATH" go run ./work/p1_demo.go
/workspace/work/aloeforge-p1 verify --artifact-root /workspace/work/p1-demo/visible
/workspace/work/aloeforge-p1 verify --artifact-root /workspace/work/nonexistent-p1-root
```

Creation and valid verification exited 0. Output:

```json
{"kind":"p1-diagnostic-only","files":1,"manifest_sha256":"b9e1437261d5b2360b399694fc8a7c8006a5549fed643a15657604b134625c5b"}
```

Missing-root verification exited **1**, printing only
`verification failed: cannot open artifact root`. Positive/negative inspect and
verify cases and redaction are also committed Go CLI tests.

The scratch program can be reproduced in a new checkout (use a fresh parent):

```go
//go:build ignore

package main

import (
    "bytes"
    "crypto/sha256"
    "encoding/hex"
    "io"
    "os"
    "strings"

    "github.com/yakumioto/AloeForge/internal/artifact"
)

func main() {
    parent := "/workspace/work/p1-demo"
    must(os.Mkdir(parent, 0700))
    payload := []byte("ab")
    sum := sha256.Sum256(payload)
    plan := artifact.Plan{SchemaVersion: 1,
        SourceInventoryHash: strings.Repeat("0", 64), SourceContentHash: strings.Repeat("0", 64),
        MethodRevision: "p1-diagnostic", RNGAlgorithm: "none", RNGRevision: "none",
        Outputs: []artifact.OutputTensor{{Name: "synthetic.output", Shard: "out.safetensors",
            DType: "U8", Shape: []int64{2}, PayloadHash: hex.EncodeToString(sum[:])}}}
    stage, err := artifact.Create(parent, ".stage-demo", plan)
    must(err)
    defer stage.Close()
    must(stage.WriteShard("out.safetensors", map[string]io.Reader{"synthetic.output": bytes.NewReader(payload)}))
    _, err = stage.Publish("visible")
    must(err)
}

func must(err error) { if err != nil { panic(err) } }
```

## Skips and limitations

No CPU tests were skipped. Main/version have no standalone unit-test files;
CLI tests cover their behavior. No real-model, GPU/CUDA, transform mathematics,
token permutation, protocol/inference, attack evaluation or P2–P8 tests were
attempted. Production model-semantic admission remains unimplemented.
P1 artifacts are diagnostic-only and must not be sent to an untrusted server.

Publication depends on Linux/local filesystem flock, fsync and atomic no-replace
rename under a trusted private parent. Process-crash tests do not certify power
loss or hostile filesystem behavior. Source files must remain immutable; callers
must recompute Go source-content bindings before resume. Stream readers must be
finite; IPC and deadline/cancellation handling remain future work. See
`ARTIFACT_CONTRACT.md` for the exact contract and adversary boundaries.

## Runner umask verification

The [first PR CPU run](https://github.com/yakumioto/AloeForge/actions/runs/38040340587)
failed because test fixtures assumed `testing.TempDir` children were private under
every umask. The production 0700 parent gate correctly rejected GitHub Runner's
wider default directory permissions. Fixtures now explicitly create a 0700 parent
(or tighten the owned CLI fixture directory); production checks are unchanged.

These additional exact commands reproduced the Runner umask and passed (exit 0):

```sh
umask 0022
PATH="/workspace/work/tools/go/bin:$PATH" go test -count=1 ./...
PATH="/workspace/work/tools/go/bin:$PATH" go test -race -count=1 ./...
PATH="/workspace/work/tools/go/bin:$PATH" make check
```

Uncached unit results: artifact 0.045s, CLI 0.003s, model 0.013s, safetensors
0.004s, strictjson 0.002s. Uncached race results: artifact 1.144s, CLI 1.016s,
model 1.026s, safetensors 1.015s, strictjson 1.010s. Final `make check` is shown
above (Python 34 passed in 0.05s). The failed initial CI skipped its later race/fuzz
step because `make check` failed; the corrected PR run is recorded in the PR body.

## PR #5 subprocess review audit

The two Sourcery/opengrep comments on `p1_measure_rss.py` name
`python.lang.security.audit.dangerous-subprocess-use-audit` and
`python.lang.security.audit.dangerous-subprocess-use-tainted-env-args`.
Both concern the same inspection child. The original call already used an argv
list and the default `shell=False`; it did not interpolate a shell command.
These are audited false positives for this trusted offline developer tool.

The call now explicitly sets `shell=False`. CLI parsing requires an existing
regular executable and an existing model directory, resolving both to absolute
paths before launching a child. Commands/flags remain fixed; spaces, quotes,
shell metacharacters and leading-dash directory names remain one literal path
argument. No PATH lookup or shell quoting is used. `shlex.quote` is inappropriate
for an argv list because it would alter the filenames.

`--binary` must identify a trusted locally built AloeForge executable. Validating
its path does **not** authenticate its contents or make an attacker-selected
program safe. This script must not be exposed as a remote executable-selection
service. Two rule-specific `nosemgrep` annotations document only this reviewed
call/argument; no file-wide or repository-wide security checks are disabled.

Executed after this change (exit 0):

```sh
.venv/bin/python -m pytest -q -c python/pyproject.toml python/tests/tooling/test_p1_measure_rss.py
PATH="/workspace/work/tools/go/bin:$PATH" make check
.venv/bin/python python/scripts/p1_measure_rss.py --binary /workspace/work/aloeforge-p1 --root /workspace/work/p1-multishard
.venv/bin/python python/scripts/p1_measure_rss.py --binary /workspace/work/aloeforge-p1 --root /workspace/work/p1-large-header
git diff --check
```

Focused tests: **13 passed** in 0.11s. They run a real synthetic child with special
filenames, assert exact argv and the absence of injected shell marker files,
reject missing/non-executable/directory/FIFO/NUL paths before child creation, and
propagate nonzero exit/invalid JSON rather than emitting a success report.
`make check`: Go formatting/vet/tests passed; Ruff lint/format passed (8 files);
**47 Python tests passed** in 0.16s, no skips. Fresh compiled inspection measurements
still passed: multishard **33,424 KiB / 0.503s**, large header
**52,792 KiB / 0.575s**, with unchanged inventory hashes.

The named public rules were also checked with local opengrep **1.30.2**:

```sh
curl -fsSL https://raw.githubusercontent.com/semgrep/semgrep-rules/develop/python/lang/security/audit/dangerous-subprocess-use-audit.yaml -o /workspace/work/subprocess-audit.yaml
curl -fsSL https://raw.githubusercontent.com/semgrep/semgrep-rules/develop/python/lang/security/audit/dangerous-subprocess-use-tainted-env-args.yaml -o /workspace/work/subprocess-tainted.yaml
/workspace/work/tools/opengrep scan --config /workspace/work/subprocess-audit.yaml --config /workspace/work/subprocess-tainted.yaml --disable-version-check --json --output /workspace/work/p1-rss-scan.json python/scripts/p1_measure_rss.py
/workspace/work/tools/opengrep scan --config /workspace/work/subprocess-audit.yaml --config /workspace/work/subprocess-tainted.yaml --disable-version-check --disable-nosem --json --output /workspace/work/p1-rss-unsuppressed.json python/scripts/p1_measure_rss.py
```

Both scans exited 0 with no scan errors: normal scanning had **0 findings**;
`--disable-nosem` had the **2 expected audit findings**. Suppression records the
manual audit, not proof that arbitrary executables are safe. The downloaded rule
SHA-256s, in command order, were
`50f69dc35f47e405633559f54fc2b3bd344707221742400e74b1c902c1f76faa` and
`0f3f3f8fafef5b828a15253585e4c6648c46c6fd1cca4b0ae8c70a9399def44d`;
the unpinned upstream branch may subsequently change. This is a local reproduction
of the named rules; hosted Sourcery review status is independent.
