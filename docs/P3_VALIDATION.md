# P3 CPU protocol validation

DIAGNOSTIC — task 03 only, based on merged P2 PR #6 at main
`977bcefe747110955dfef4a72da1ce01b1f0aa7b`. All tests use synthetic byte pieces,
public fake runtime metadata and physical-ID scripts; no model weights are loaded,
no model is downloaded, and no GPU/privacy/generation-quality result is claimed.

## Commands actually executed

Linux amd64; official Go 1.25.3, Python 3.12.14, pytest 8.4.2, Ruff 0.14.0.
The official Go directory supplies both `go` and `gofmt`; Python uses the existing
editable package installation in `.venv`. These commands all completed with exit 0:

```sh
PATH="/workspace/work/tools/go/bin:$PATH" make check
PATH="/workspace/work/tools/go/bin:$PATH" go test -race -count=1 ./...
PATH="/workspace/work/tools/go/bin:$PATH" go test -count=1 ./internal/clientruntime ./internal/gateway ./internal/runtimeprotocol
PATH="/workspace/work/tools/go/bin:$PATH" go test -race -count=30 ./internal/clientruntime -run 'TestRejectUnsafeClientConfiguration|TestHandshakeMismatchPreventsGenerationAndRedirect'
PATH="/workspace/work/tools/go/bin:$PATH" go test -count=1 -cover ./internal/clientruntime ./internal/gateway ./internal/runtimeprotocol
PATH="/workspace/work/tools/go/bin:$PATH" go test ./internal/clientruntime -run '^$' -fuzz '^FuzzSSE$' -fuzztime=3s -parallel=2
PATH="/workspace/work/tools/go/bin:$PATH" go test ./internal/clientruntime -run '^$' -fuzz '^FuzzUTF8Pieces$' -fuzztime=3s -parallel=2
PATH="/workspace/work/tools/go/bin:$PATH" go test ./internal/safetensors -run '^$' -fuzz '^FuzzHeader$' -fuzztime=3s -parallel=2
PATH="/workspace/work/tools/go/bin:$PATH" go test ./internal/tokenbaseline -run '^$' -fuzz '^FuzzPermutationInverse$' -fuzztime=3s -parallel=2
git diff --check
```

## Observed results

- Final `make check`: Go formatting/vet/tests and Ruff lint/format passed;
  **76 Python tests passed in 0.55s**, 10 Python files already formatted. No skips.
- Final complete race check passed: clientruntime 2.098s, gateway 5.288s,
  runtimeprotocol 1.040s; existing P1/P2 packages passed as well.
- Focused verified-TLS protocol suite passed: clientruntime 0.572s, gateway 1.173s,
  runtimeprotocol 0.004s. A separate coverage run observed clientruntime **94.7%**,
  gateway **86.0%**, runtimeprotocol **74.0%** statement coverage; percentages are
  diagnostic evidence, not security or semantic-model admission thresholds.
- SSE fuzz: **20,136 executions**, PASS (4.013s); UTF-8 piece fuzz:
  **22,846 executions**, PASS (4.011s). The latter compares valid incremental
  decoding/stops against an independent complete UTF-8 string oracle and checks
  UTF-8/pending-suffix bounds. Two workers and three-second budgets per target.
- Existing header fuzz: **169,636 executions**, PASS (3.034s); permutation inverse
  fuzz: **98,891 executions**, PASS (3.100s). CPU CI retains both and adds the two
  runtime fuzz campaigns after `make check` and full race checks.
- Development rerun initially exposed a negative-test defect: replacing a random
  mapped EOS with constant 15 could accidentally leave EOS unchanged. The fixture
  mutation now chooses `(actual + 1) % vocab`, guaranteeing a different ID without
  weakening validation. Both configuration/handshake negative suites passed
  **30 repeated race runs** (3.926s); final `make check` and complete race rerun
  passed after that correction.

## Positive and negative evidence

The local end-to-end test uses an actual HTTPS server and a client trusting only
its ephemeral certificate, with hostname/certificate verification enabled. It
restores exact original IDs `[0,3,4,5,6,2,7,1]` and text `H中!`: original IDs 4/5/6
supply the individual UTF-8 bytes of 中, BOS/PAD remain in the ID result but not
text, and EOS ends the stream before trailing script IDs. Wire prompt IDs equal
the forward-mapped sequence; the API never reconstructs IDs by encoding text.

Tests cover original-coordinate stop IDs mapped on the wire, EOS/BOS/PAD role
aliases, EOS precedence, max-token length, backend exhaustion, overlapping and
Unicode text stops, prefix holdback/flush, and prompt-independent fake output.
Invalid/incomplete UTF-8 and oversized local byte pieces fail explicitly.
Local text stop and callback failure cancel the HTTP stream; a deliberately
private callback error is replaced by a static code.

Negative gates cover insecure HTTP/TLS verification, unknown trust roots, redirects,
wrong artifact/model/binding/tokenizer/method/specials/features/bounds/protocol,
missing/null/duplicate/unknown JSON fields, **null array IDs** (which ordinary
Go JSON could otherwise turn into zero), invalid types/ranges/stop duplicates,
unsupported plaintext controls, wrong auth/ambiguous multiple auth headers,
query/reconnect/content-encoding headers, oversized bodies and malformed SSE.
Bad sequence/request ID, premature EOS, extra token after EOS, missing terminal,
truncated boundary, oversized/infinite line, backend error/timeout and interrupted
stream all yield explicit errors/partial results with no automatic retry.

Forbidden marker prompts have local original-token byte pieces. Tests capture the
actual POST boundary and gateway logs and assert the prompt marker, permutation,
nonce, source, Client Secret and plaintext stop fields never appear there. Other
rejection tests deliberately place a forbidden marker in malformed bodies, unknown
fields, auth, paths and query strings; responses are static and server logs empty.
This covers first-party handler/server behavior, not third-party middleware.

Cancellation before the first token, after text output and after local text stop
releases fake generation. The real HTTP ledger test accepts 4096 distinct IDs,
rejects additional IDs at capacity and retains canceled IDs against replay. Eight
simultaneous slow fake streams are admitted, the ninth receives 429, and an active
duplicate receives 409. Separate concurrent streams pass under the race detector.
A **real TLS socket with constrained TCP buffers and an undrained response body**
blocks writes; a 25ms write deadline releases the handler/inference slot within
the two-second test bound, with fewer than 4096 successful token writes. A writer
without deadline support is rejected rather than silently running unbounded.

## Skips, scope and limitations

No CPU test skipped. Main/version packages have no standalone test files; existing
CLI tests cover their behavior. No real HF tokenizer/chat template, model, Python
runtime/IPC, full-V2 transform, GPU/CUDA/vLLM, distributed/persistent replay ledger,
cache/key rotation, performance benchmark or privacy/attack evaluation attempted.
No Rust or P4–P8 implementation. P0 unimplemented contract drafts are unchanged.

The application must verify expected identities and its actual local tokenizer
through a trusted channel; hashes are not signer authentication. The fake gateway
loads no P1/P2 diagnostic artifact weights, and those weights remain inadmissible
for an untrusted server. Authentication requires verified TLS. Request IDs are
retained only within this bounded gateway process; restarts/multiple instances have
no durable replay guarantee. Partial output must be handled explicitly, without
assuming a retry can resume or deduplicate it. Trusted callbacks must return
promptly. Per-request buffers/deadlines are not a total process RSS guarantee.
