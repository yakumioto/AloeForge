# P2 token-permutation diagnostic contract

DIAGNOSTIC — task 02 implements a bounded synthetic embedding/LM-head baseline,
not a transformer adapter, full AloePri V2, encryption or a remote runtime.

## Coordinates and admitted fixtures

The bijection `p` maps original vocabulary IDs to physical IDs. Its inverse maps
physical IDs back. Output embedding/head row `p[i]` equals source row `i`.
For a hidden vector `h`, physical logits satisfy `L'[p[i]] = L[i]`; comparisons
must first restore original coordinates. A tied head aliases the single output
embedding tensor; an untied head is separately permuted. No hidden-space, RoPE,
attention, normalization, bias, noise or tokenizer-text transformation is applied.

The input must have a strict `token-baseline.json` v1 with kind
`synthetic-embedding-head-v1`, explicit vocabulary/hidden sizes, tied flag and
`special_token_ids` containing exactly `bos`, `eos`, `pad`. BOS/PAD have zero or
one ID; EOS has 1–16 IDs. Values are sorted, unique within each role and in range;
aliases across roles are allowed. `tokenizer.json` is a strict synthetic vocabulary
v1 (`kind: synthetic-vocabulary-v1`, unique UTF-8 `tokens`), not an HF tokenizer.
Its raw-byte SHA-256 is independently bound; vocabulary count must match.

P1 Go inspection and whole-source binding remain mandatory. There are exactly
one (tied) or two (untied) rank-2 row-major little-endian F32 tensors named
`embedding.weight` and, when untied, `lm_head.weight`, each shaped `[vocab, hidden]`.
All values must be finite. Unsupported tensors, dtypes/shapes or sidecar variants
fail closed. Limits: vocabulary 2–65,536, hidden size 1–256, at most 1,048,576
elements per matrix; token strings at most 64 bytes. Tensor rows are streamed;
permutation memory is O(vocab), and row memory is O(hidden), not O(weight bytes).
These fixtures are not admitted as real Llama/Qwen models by their HF identity.

## External Client Secret

Go creates an immutable, versioned `client-secret.json` in an existing private
0700 directory disjoint from source and output directories. File creation is
0600, fsynced and atomically renamed without replacement under a kernel lock.
An existing secret is never overwritten or silently regenerated. Loading rejects
symlinks, non-regular/widely readable/oversized files, duplicate or unknown fields,
unsupported versions/revisions, noncanonical JSON, malformed nonce, and
non-bijective permutations.

Private material includes source inventory/content hashes, tokenizer fingerprint,
dimensions/tied layout/original special IDs, method/RNG revisions, a random
32-byte nonce and the permutation. Production generation uses unbiased
Fisher–Yates with `crypto/rand.Int`, not a seed or `math/rand`.
Method: `token-permutation-v1`; RNG: `crypto-rand-fisher-yates`, revision `1`.

The binding ID is SHA-256 of ASCII `AloeForge/token-binding/v1\n` followed by Go
canonical typed JSON of the private material. Field order is context, nonce,
permutation; context order is source, tokenizer_sha256, vocabulary_size,
hidden_size, tied_embeddings, special_token_ids, method_revision, rng_algorithm,
rng_revision. Source order is inventory_sha256, content_sha256. Map keys sort
lexically. Nonce and digests are lowercase hex. This is a binding commitment,
not proof of origin or a security/privacy claim. The nonce does not turn
permutation into encryption.

## Public identity, verification and recovery

P2 reuses P1 bounded output-driven staging, hashes, checkpoint validation and
atomic publication. Its manifest kind is `p2-token-permutation-diagnostic-only`.
The public binding contains only schema/profile/revisions, binding ID, tokenizer
fingerprint, dimensions/tied layout and mapped special IDs. No original IDs,
permutation, nonce, private source fingerprints or secret files are published.
Public shape/name/byte hashes and binding are included in artifact identity.
P1 manifests without a binding retain their prior kind/encoding/identity.

The secret is durable before staging; resume also fsyncs its file and directory.
Resume reloads it, recomputes source and
tokenizer/layout bindings, regenerates the same output plan, and uses P1 recovery.
Completed shards are reverified and skipped; partials are discarded. If no stage
was created before a crash, explicit resume may create one using the already
persisted matching secret. Missing/corrupt/mismatched secrets fail without
generating a replacement. Retry after publication returns the same artifact ID.

Client verification requires an externally expected artifact ID and tokenizer
fingerprint, verifies the artifact in Go, recomputes the secret commitment and
compares the entire public binding including mapped special IDs. Hashes alone do
not authenticate attacker-rewritten artifacts; expected identities must come from
the trusted compiler/client channel. No P3 wire protocol is defined here.

Both source and generated files must remain immutable during use. Trusted local
filesystem and private-directory assumptions from P1 apply. Diagnostic outputs
must never be sent to an untrusted server; they are neither a complete transformed
model nor a privacy protection mechanism. P0 contract drafts remain unimplemented.

## Acceptance evidence

Go negative tests cover corrupted permutations, ID bounds, mismatched secret,
source/tokenizer/layout changes, persistence permissions/paths, mapped special
IDs and checkpoint recovery. An independent Python standard-library oracle
implements inverse/row mapping and FP64 dot products. CPU integration tests run
the real Go CLI on tied/untied tiny safetensors, independently verify binding
commitments and compare every token's logits only after coordinate restoration.
Tests include fixed known values and rejection/tamper cases; no model downloads,
GPU evidence or real-model quality/privacy claims are included.
