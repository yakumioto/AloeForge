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
ok      github.com/yakumioto/AloeForge/internal/artifact   (cached)
ok      github.com/yakumioto/AloeForge/internal/cli        (cached)
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
