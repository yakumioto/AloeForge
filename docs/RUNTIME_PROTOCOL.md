# P3 ClientRuntime / Gateway token-only protocol

DIAGNOSTIC — `codex/tasks/03-protocol.md` implements Go libraries and a deterministic
**fake token backend only**, on CPU with no model. ENGINEERING INTERPRETATION —
the wire schema, bounds, stop semantics and retry policy below are project choices,
not a claim about AloePri's security. Token permutation is **not encryption**.

## Trust and entry points

- `internal/clientruntime`: trusted client library. `New(Config)` requires an
  externally trusted expected runtime, Client Secret, actual local tokenizer
  fingerprint and `Piece(originalID)` returning exact local byte pieces.
  `Generate(ctx, originalIDs, Options)` maps input IDs once, restores output IDs
  once and incrementally decodes bytes. `Handshake(ctx)` is also explicit.
  The application owns local tokenization/chat templates; no HF tokenizer is
  implemented, and no text encode/decode round-trip fallback exists.
- `internal/gateway`: untrusted-server handler. `New(Config)` accepts public runtime
  metadata, bearer authentication and a fixed **physical-ID** fake script. Fake
  output is independent of the prompt. It receives no tokenizer, private mapping,
  source weights or model artifact directory. Optional bounded delay/fault
  injection exercises cancellation/errors, not real inference.
- `internal/runtimeprotocol`: shared bounded public structs and strict decoders.
  P0 draft contracts remain explicitly unimplemented; this slice has its own
  version/profile and does not imply a production or full-V2 runtime.

P1/P2 diagnostic weight artifacts must **not** be sent to an untrusted server.
P3 uses only their public identity/binding metadata. Expected artifact identity,
complete public binding and tokenizer identity must come from the trusted compiler
channel; the client compares its Secret's detached `PublicBinding()` against it.
Where a local P2 artifact is involved, use `VerifyClient` before trusting that
metadata. A hash/handshake alone is not signer authentication or proof of model
execution. The application must independently check its actual tokenizer bytes;
passing an unchecked fingerprint string does not validate a tokenizer.

Use `Gateway.Server(addr).ListenAndServeTLS(cert, key)` and `Server.Shutdown(ctx)`.
Direct verified HTTPS plus a bearer credential is mandatory. The client enforces
HTTPS, certificate validation and TLS >= 1.2, refuses userinfo/query/path URLs and
redirects, and never disables verification. The gateway rejects requests without
`Request.TLS`; reverse proxy TLS offload is not supported in P3. Its server helper
sets read/header/write/idle deadlines, an 8 KiB header setting, and discards HTTP
server diagnostics; the handler has no request logger. Do not add middleware
that logs authentication, bodies, token IDs, paths or plaintext query strings.

## Handshake

Authenticated `GET /v1/models/{model_id}/runtime`, with no query, returns required,
non-null fields:

```json
{
  "protocol_version": 1,
  "profile": "p3-fake-token-only",
  "model_id": "fake-demo",
  "artifact_id": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "token_binding": {
    "schema_version": 1,
    "profile": "token-permutation-diagnostic",
    "binding_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "tokenizer_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "method_revision": "token-permutation-v1",
    "rng_algorithm": "crypto-rand-fisher-yates",
    "rng_revision": "1",
    "vocabulary_size": 16,
    "hidden_size": 2,
    "tied_embeddings": true,
    "mapped_special_token_ids": {"bos": [7], "eos": [9], "pad": [5]}
  },
  "supported_features": ["greedy", "sse", "stop_token_ids"],
  "max_prompt_tokens": 32,
  "max_output_tokens": 32,
  "context_tokens": 64
}
```

Illustrative identities only. Protocol/profile/model/artifact identity and every
binding field, feature and bound must exactly match the trusted expectation.
`Generate` performs this check before **every** POST; capability discovery never
silently changes configuration. Model IDs use ASCII letters/digits/hyphen/underscore,
1–64 bytes. Artifact/binding/tokenizer hashes are 64 lowercase hex digits.
P2 special-ID validation applies, including sorted unique IDs and role aliases.
Features are exactly the ordered list above. Per-input/output bounds are 1–4096;
context bound is 2–8192 and also bounds prompt count + requested output count.

## Request

Authenticated `POST /v1/generate`; exact `Content-Type: application/json` and
`Accept: text/event-stream`; no query, content encoding or `Last-Event-ID`.
All fields, including empty arrays, are required and non-null:

```json
{
  "protocol_version": 1,
  "model_id": "fake-demo",
  "artifact_id": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "binding_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "request_id": "0123456789abcdef0123456789abcdef",
  "prompt_token_ids": [7, 8, 3],
  "stop_token_ids": [],
  "sampling": {"temperature": 0.0, "max_tokens": 16},
  "stream": true
}
```

All IDs on the wire are in **physical/permuted** vocabulary coordinates. Prompt
IDs must be nonempty and in range. Up to 16 unique stop-token IDs are allowed;
client options take original IDs and map them before transmission. The client
neither inserts BOS nor strips/pads prompt IDs. Request ID is exactly 32 lowercase
hex digits (128 random bits by default); callers may supply a fresh valid ID.
Only temperature zero/greedy and streaming are admitted. Unknown fields, duplicate
JSON keys, missing/null fields, bad UTF-8, invalid nesting, trailing JSON, unsupported
sampling and text-mode/forced-token/logit-bias/allowed-set fields fail closed.
There is no fallback tokenizer, cache, logit bias, forced token or Python backend.

Bodies and handshake responses are bounded at 64 KiB. Static JSON errors carry
only `error`: 401 unauthorized; 400 invalid/unsupported request or TLS required;
413 body limit; 404 unknown route/method; 409 binding mismatch/duplicate request;
429 capacity. No untrusted input is echoed. Ambiguous multiple authorization
headers are rejected. Bearer credentials are required on both endpoints and
compared without content-dependent early exits; they are never put in URLs.

## SSE and bounded execution

Each record is exactly:

```text
event: token
data: {"request_id":"0123456789abcdef0123456789abcdef","sequence":1,"output_token_ids":[3],"finish_reason":"","error":""}

```

All five JSON fields are required and non-null. Sequence starts at 1 and increments
by exactly one, including the terminal record. A token record has exactly one
physical ID and empty reason/error. A terminal record has an empty ID array and
one of `eos`, `stop_token`, `length`, `backend_done`, or `error`. Error terminals
carry only `backend_error` or `timeout`; all other terminal errors are empty.
No plaintext text, SSE reconnect ID, heartbeat, multiline data or resume extension.

The fake script is bounded at 4096 IDs and emits directly into the response, with
no producer goroutine, queue or prefetch buffer. EOS takes precedence over explicit
stop tokens; generation stops at EOS, a mapped stop ID, max_tokens, or script end.
At max_tokens, reason is length unless EOS/stop won on that last token. At most
8 accepted streams run at once. Server response headers flush before the first
token; cancellation interrupts a fake delay and all subsequent sends. Per-stream
execution timeout is at most 30 seconds, per-write timeout at most 5 seconds.
Each write uses the earlier execution/write deadline; a final timeout error is
best effort with at most one write-timeout grace period. Disconnect/write failure
can make that error undeliverable; the client still reports an incomplete stream.
The configured server also bounds body/header reads and non-stream writes.

Client SSE lines are bounded at 4096 bytes, aggregate reads at 16,912,416 bytes,
and accepted events at requested max_tokens + one terminal. Handshakes, bodies,
streams and tokenizer pieces have independent limits. This bounds application
buffers **per request**, not total process RSS or accepted TCP connections.
Client total HTTP timeout is 35 seconds per exchange; caller context can shorten it.
Client output retains at most 4096 original IDs and 1 MiB of decoded text.

## Local decoding, stops, cancellation and partial results

`Result.TokenIDs` contains the exact restored sequence received, including BOS,
PAD, EOS and explicit stop IDs. It is never reconstructed by re-encoding text.
EOS wins over aliased roles; EOS and stop IDs are omitted from text, while BOS/PAD
are suppressed in output text wherever present. Non-special tokens are passed to
the local `Piece` callback only after inverse mapping. A piece is at most 256 bytes,
may split a UTF-8 rune, and must contribute valid UTF-8 across the complete stream.
Decoder retains at most three unfinished UTF-8 bytes; invalid or truncated UTF-8
fails explicitly instead of emitting replacement characters.

Up to 8 nonempty local UTF-8 stop strings, each <= 256 bytes, never enter the wire.
The decoder retains the longest suffix that could become a stop string; callbacks
receive complete UTF-8 with no matching stop-prefix leakage. The earliest complete
match terminates local text with `text_stop`, drops the stop and the rest of that
piece, closes/cancels the HTTP stream and interrupts the fake backend. Received IDs
remain exact through that whole piece. Unmatched prefixes flush only at a valid
terminal. Error/partial streams do not flush undecided stop or UTF-8 tails.
`OnText` returning an error also cancels; its private error text is not propagated.
Callbacks/tokenizer functions are trusted synchronous application code and must
return promptly; caller-owned tokenizer concurrency safety is the caller's job.

The client validates request ID, exact sequence, ID range/count, output limit,
terminal shape and EOS/stop/length consistency. Missing terminal, malformed stream,
transport loss, invalid bytes, cancellation or error terminal returns `PartialError`
with a static code and accepted-token count, together with the accepted IDs/text
in `Result`. A POST transport error before response headers is `submission_unknown`:
zero observed tokens does not prove that the server rejected the request.
Explicit HTTP rejection is a static error, not an automatic retry.

Neither side retries/reconnects or appends resumed output. The gateway retains
accepted request IDs, including canceled/failed streams, for its process lifetime.
Duplicates return 409 before generation. At 4096 retained IDs, new requests return
429; records are never evicted silently. This diagnostic ledger is not persistent,
so it makes no replay guarantee across restarts or other gateway instances. Callers
must use fresh IDs, treat partial output explicitly and never assume a retry can
resume or safely deduplicate generation. Production persistence, idempotent replay,
KV/prefix caching and key rotation are future tasks, not implemented here.

## CPU evidence

Run `go test -count=1 ./internal/clientruntime ./internal/gateway ./internal/runtimeprotocol`
for real verified TLS + exact IDs, Unicode/stops, binding/auth/schema failures,
cancellation, partial streams, forbidden-marker logging checks, concurrent slots,
full lifetime ledger and real slow-reader write deadlines. See
[P3_VALIDATION.md](P3_VALIDATION.md) for actual commands/results. No model, GPU,
real HF tokenizer, Python runtime, semantic generation or privacy evaluation.

P6 preparation adds `gateway.NewRelay` with a trusted in-process physical-ID
Backend/TokenStream mock interface. It preserves this profile/binding and all
wire rules, pulls once per sent token, closes on stop/error/disconnect and
redacts backend errors. It does not accept dense identities/artifacts, add Python
IPC or admit real inference. See VLLM_BACKEND_CONTRACT.md and P6_VALIDATION.md.

A separate trusted-local `p6-local-dense-reference` profile now connects P5's
FP64 reference through bounded Go/Python IPC. Its dense binding, loopback-only
entry points and process lifetime are specified in LOCAL_REFERENCE_CONTRACT.md.
It does not change P3 remote admission or establish a GPU/production gate.
