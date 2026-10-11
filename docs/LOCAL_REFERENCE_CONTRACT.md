# P6 trusted local CPU reference bridge

DIAGNOSTIC / ENGINEERING INTERPRETATION — a Go/Python integration experiment
on a trusted machine. This connects P5's **FP64 NumPy reference**. P4 method
admission, P5 real-checkpoint quality and P6 physical GPU gates remain pending.
Keep these diagnostic publications trusted. Token permutation is not encryption.
The independent FP32 mock geometry guard and its numeric gates are unchanged.

## Entry points and trust

`runtimeprotocol.LocalDenseProfile` is `p6-local-dense-reference`, version1.
Its authenticated runtime response has P3's common fields/features/bounds, with
the complete P5 public **dense_contract** replacing `token_binding`. All fields,
including false/empty special arrays, are required/non-null. Mixed bindings,
unknown fields, duplicate keys, invalid UTF-8, unsupported geometry/method/cache
and context beyond publication fail closed. Request/SSE shapes are unchanged:
physical IDs, strict sequence, greedy, EOS/physical stop/length and no retries.

On the trusted client, first `dense.VerifyClient(publication, secretRoot,
expectedArtifactID, actualTokenizerHash)` against compiler-provided identities.
Read the verified public dense contract; load `dense.LoadSecret` only in the
client. `clientruntime.New(Config{DenseSecret: secret, Expected: capability,
...})` checks its commitment/vocabulary and local tokenizer fingerprint. P3 uses
`Secret`; mixing them fails. Map input/stop IDs once, restore output IDs once;
byte pieces/text stops stay in the client. The application must independently
verify actual tokenizer bytes; a supplied hash alone does not verify them.

`referencebackend.New(Config{Runtime: capability, ArtifactRoot: publication,
Python: absoluteTrustedInterpreter, ...})` verifies publication identity and the
complete contract. Installed Python needs `numpy==2.3.3`; no Torch, vLLM, download
or source-model path is required. `gateway.NewLocalDense` matches the complete
backend capability. `LocalServer("127.0.0.1:port")` (or literal `::1`) supplies
bounded HTTP configuration for `ListenAndServeTLS`. TLS certificate verification
and bearer authentication remain mandatory. The client requires literal loopback
and disables environment proxies; the handler rejects non-loopback peers. Server
helpers refuse wildcard/remote addresses and DNS names. `New`/`NewRelay` remain
P3-only. There is no new CLI/text service or HF tokenizer.

This is a trusted local library path. Do not forward/tunnel/proxy it or deploy
it untrusted. Loopback and Python isolated mode are not filesystem/container
isolation, signer authentication or proof of model execution.

## Framed pull IPC

Go starts `python -I -m aloeforge_runtime.reference_worker`, one process per
accepted session. Frames have a four-byte unsigned **big-endian** payload length
followed by strict UTF-8 JSON. Empty/oversized/truncated frames fail before
allocation. Initial `open` is at most 64 KiB; subsequent commands/responses are
at most 4096 bytes. Tensor data never crosses JSON.

`open` contains exactly `schema_version:1`, `operation:"open"`, absolute public
`artifact_root`, complete public `runtime` and validated physical `request`.
The worker independently validates control, reloads/verifies publication and
compares the **entire** contract. `ready` contains schema/event/request ID plus
artifact/binding/tokenizer identities, all checked by Go. No prefill runs yet.

`pull` contains schema1, operation, request ID and sequence starting at1.
First pull runs full causal prefill; later pulls decode the previous emitted
physical ID using private session cache. `token` contains exactly schema1, event,
request ID, sequence and `physical_token_id`. Both sides enforce range/sequence
and output bound. No producer queue/prefetch, extra pull after EOS/physical
stop/length, shared cache or resume/retry. A raw worker caller may pull once after
terminal for `done`; Go reports completion locally and closes the process.
Worker failure emits only `{schema_version:1,event:"error",error:"reference_error"}`
and exits1. Go converts process/pipe/schema/identity failures to static errors.

## Lifetime, memory and diagnostics

Backend capacity defaults to one child, with configurable maximum two and
immediate rejection. Gateway eight active slots/lifetime4096 request ledger remain
separate. Each child owns its mmap/cache and has default/maximum 30-second **total**
lifetime, including startup, verification, idle time and computation. Context
cancellation interrupts startup and pipe I/O. `Close` cancels/kills the Unix process
group, waits/reaps once, closes pipes and releases capacity. A context callback
also reaps abandoned idle sessions. No blocking graceful-close pipe write is used;
WaitDelay is1s. There is no Python worker pool or backend shutdown requirement.

Only fixed PATH and BLAS thread limits enter child environment: no inherited
credentials, proxy, PYTHONPATH or preload hooks. Child stderr is discarded; stdout
contains framed control. No prompts, original IDs, private maps, seed, Secret
directory or source path is sent to the worker. No request logs; HTTP/worker
exceptions never echo private data. Test oracle inputs stay in the trusted test
harness. Source files are deleted before real TLS tests.

P5 default allocation bounds remain: 4 GiB weights, 512 MiB cache including peak
replacement, 64 MiB attention, 128 MiB activations, 256 MiB logits. These are
individual buffer gates, not total RSS guarantees. Each child loads a publication;
startup/CPU cost make this a bounded reference experiment, not performance serving.
P3 HTTP/body/SSE/write deadlines remain. Local text stop/cancellation may leave one
computation in flight until disconnect reaches the parent, which kills/reaps it.
Physical stops/output bounds require no further decode.

## Validation

Base NumPy CI runs `python/tests/runtime/test_reference_bridge.py`. It compiles
fresh F32/F16/BF16, tied/untied, MHA/GQA and head_dim64 publications, computes the
independent scalar source-model oracle, deletes source files and invokes a Go
**race-enabled** real-process verified-TLS test. Exact restored greedy, repeat
cache isolation, physical/local text stops, actual cancellation/startup timeout,
full-contract handshake, loopback and Secret/tokenizer bindings are checked.
Go helper processes explicitly replace numerical execution to inject bad ready
metadata, bad IDs/sequences/null/unknown/duplicate/oversized/truncated frames,
hangs, capacity, environment markers and concurrent close. P5 numeric thresholds
are unchanged. See P6_VALIDATION.md for executed results and remaining gates.
