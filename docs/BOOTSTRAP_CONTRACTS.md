# P0 contract data structures

ENGINEERING INTERPRETATION — these are version 1 **documentation drafts**, not
runtime schemas, serialized artifacts or accepted model capabilities.
[`contracts/bootstrap.json`](contracts/bootstrap.json) defines the initial
field names. Test-only structures in `python/tests/contracts/test_drafts.py`
check the draft and reject altered versions, missing/unknown fields and populated
placeholders. No production code loads or validates these drafts.

Every structure has `schema_version: 1` and `status: "unimplemented"`.
**Every other field is unimplemented and must remain JSON `null` in P0.**
Null means unspecified, never a supported default or a successful validation.
Schema versions are independent of future protocol and method revisions. Future
tasks must define types, invariants, migration/rejection rules and negative tests
before accepting any values.

| Structure | Initial fields and intended meaning (all unimplemented) |
| --- | --- |
| `ModelIR` | `architecture_id`, `tokenizer_fingerprint`; distinct `source_tensor_inventory` / `output_tensor_inventory` and `logical_hidden_dimensions` / `physical_hidden_dimensions`; `attention_layout` (head/KV groups), `normalization` (epsilon), `rope_layout` (pair ordering), `ffn_layout` (activation), `layer_tensor_roles`, `moe_layout`, `mla_layout`, `sharding`, `quantization` |
| `MethodContract` | `method_revision`, `evidence_label`, `paper_reference`, `profile`, `rng_algorithm`, `rng_revision`, `transform_specification`, `acceptance_gates` |
| `RuntimeContract` | `protocol_revision`, `artifact_id`, `method_revision`, `binding_id`, `tokenizer_fingerprint`, `loader_revision`, `cache_layout`, `mapped_special_token_ids`, `context_bounds`, `supported_features` |
| `SecretBinding` | Public identity only: `artifact_id`, `binding_id`, `tokenizer_fingerprint`, `method_revision`, `protocol_revision`. No secret material, private paths, original token IDs or prompts. |

Method evidence must eventually distinguish PAPER, ENGINEERING INTERPRETATION
(called INTERPRETATION in the existing method contract) and DIAGNOSTIC, with
specific paper references for PAPER assertions. P0 adds no mathematical claims.
RNG algorithm and revision must be bound as well as any future private seed.

The compiler and trusted client will retain secret material separately from
public identity metadata. The untrusted server receives only the published
obfuscated artifact and public contract/binding identifiers. Fingerprint and
revision mismatches must fail closed. Full V2 requires a custom loader/cache
contract; this draft implies no vanilla Hugging Face or vLLM compatibility.

Cross-process I/O remains a design boundary: small, versioned, size-bounded
control messages between Go and Python; large tensors via validated files,
mmap or bounded streams, never inline JSON. Future implementations must check
paths, inventories, sizes, dtypes and shapes before use, reject unknown layouts,
bound diagnostics and avoid logging inputs or secrets. P0 implements no IPC,
parser, compiler, client, gateway, secret generation or remote runtime.
