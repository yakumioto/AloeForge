# P1 inventory and diagnostic artifact contract

ENGINEERING INTERPRETATION — Linux-only structural foundation for task 01. No
AloePri transform, tokenizer mapping, model-semantic adapter, inference or privacy
guarantee is implemented. All four P0 contract drafts remain unimplemented.

## Source inventory

`model.Inspect(root)` uses Go `os.Root` confinement, including symlink resolution.
Shard names are portable basenames, never absolute/traversal paths. Relative
source symlinks within the root are allowed; escaping symlinks are rejected.
Nonblocking opens reject FIFOs and other non-regular files without waiting.

Discovery accepts `model.safetensors` alone, or an explicit
`model.safetensors.index.json` exactly covering all discovered `.safetensors`
shards and their tensors. Duplicate/missing names, extra shards, contradictory
mappings and contradictory `metadata.total_size` fail. Total size, when present,
must be a nonnegative integer.

Architecture identity must be exactly `llama` / `LlamaForCausalLM` or `qwen2` /
`Qwen2ForCausalLM`, with one declared architecture. Unknown/multiple/mismatched
identities, non-null quantization config, remote `auto_map` and RoPE scaling fail.
This is structural identification, **not** admission of attention geometry,
weight roles, aliases, semantic layouts or runtime/transform compatibility. Other
HF config fields contribute to the config fingerprint but are not interpreted.
Semantic capability is unimplemented and must fail closed in future adapters;
it cannot be inferred from this inventory.

JSON is checked recursively for duplicate keys (including escaped aliases),
trailing data, malformed UTF-8 and unpaired Unicode surrogates. Typed tensor,
index, plan, checkpoint and manifest records reject unknown fields. HF config
fields and string-only safetensors metadata may have extra keys; their raw content
is never copied to diagnostics or published artifacts.

| Bound | Value |
| --- | --- |
| One config/index/control file or safetensors header | 8 MiB |
| JSON nesting / model directory entries | 32 / 4,096 |
| Shards / tensors per shard / aggregate tensors | 128 / 32,768 / 100,000 |
| Tensor rank / name bytes | 16 / 512 |
| Streaming hashing/copy buffer | 32 KiB |

Supported dtypes: BOOL, U8/I8, U16/I16, U32/I32, U64/I64, F16, BF16, F32, F64.
Shapes are non-null integer arrays with nonnegative dimensions. Scalars (`[]`)
and empty tensors are explicit. Size arithmetic, offsets and header lengths are
overflow checked. Tensor spans must cover the entire payload exactly: no overlaps,
holes or trailing bytes. `Inspect` never reads source payloads.

`ModelInventory` v1 records architecture identity, canonical-config SHA-256,
sorted shard file/data sizes, and name-sorted tensors with shard/dtype/shape/offsets.
Its SHA-256 binds metadata, **not weights**. Canonical encoding is Go
`encoding/json` over typed/sorted records and sorted config keys. It is not RFC
8785; numeric spellings remain significant. CLI exports counts and fingerprints,
not absolute paths, tensor names or arbitrary metadata.

`model.BindSource(root)` separately streams entire source shards into SHA-256
and combines their sorted digests with inventory identity. Its private
`SourceBinding` contains inventory and content fingerprints. Recompute it in Go
before `Resume`; identical headers cannot prove identical source weights.
The trusted compiler must keep source files immutable during binding/compilation.
Stat/metadata checks detect ordinary changes but do not authenticate a filesystem
controlled by a privileged concurrent writer.

## Stage-only output writer

Private `artifact.Plan` v1 binds both source fingerprints, method revision, RNG
algorithm/revision and explicit output descriptors. Revision identifiers are
opaque metadata, not implemented algorithms; P1 examples use `p1-diagnostic`,
`none`, `none`. There is no seed/Client Secret field. `OutputTensor` contains
output name/shard/dtype/physical shape/expected payload SHA-256, with no source
name, shape or offset defaults. Descriptor hashing sorts by shard/name and binds
every field. A golden test uses an independently calculated Python hashlib digest.

`Create(parent, ".stage-ID", plan)` requires an existing private 0700 parent and
creates a new 0700 stage. Bounded identifiers are required. Nonblocking kernel
`flock` serializes sessions per parent and releases on process death; a mutex
serializes in-process operations. Go owns control files. Python workers may supply
only finite tensor streams to `WriteShard`, never public paths or modified plans.
IPC framing, stream deadlines and cancellation remain future work.

`WriteShard` requires exactly the planned streams, computes offsets/header from
output descriptors, consumes exactly declared bytes and checks content hashes.
Short/extra/error streams fail. Only complete, validated, fsynced files are renamed
from `.part`, followed by an atomic/fsynced checkpoint containing file digests and
full plan/source binding. Source config/sidecar metadata is never copied.

## Recovery and atomic publication

`Resume(parent, stageName, expectedPlan)` checks saved plan/checkpoint against both
fresh source fingerprints, method/RNG revisions and output contract. It revalidates
all completed shards, adopts complete orphans only after header/content checks,
and discards partial shard/control temporaries. Unexpected files, symlinks, missing
completed shards, unsupported versions and wrong bindings fail closed. A crash
before initial plan/checkpoint durability leaves an unusable stage; create a new
stage rather than guessing its binding.

`Publish(target)` verifies all files, copies only output shards plus a public
manifest into a separate private directory, fsyncs and verifies that snapshot,
then atomically publishes using Linux `renameat2(RENAME_NOREPLACE)`. The parent
is fsynced before/after rename. Different existing files/directories/symlinks are
never replaced. Unsupported filesystems fail without a weaker fallback. This
requires trusted local filesystem support for flock, directory fsync and no-replace
rename. Hostile local writers, privileged mount changes/bind mounts/devices and
hostile filesystems are outside the P1 threat model.

The private stage/checkpoint remains available. Published files are independent
copies, not hardlinks. Retry after a lost acknowledgement succeeds only for an
identical fully verified artifact. Post-rename parent-sync failure reports visible
output with uncertain durability and requires verification/retry. Process-death
tests cover creation, interrupted writing, checkpoint progress and publication;
synthetic orphan/partial-copy states exercise restart boundaries. They do not
certify physical power-loss or filesystem fault tolerance.

## Public diagnostic manifest

Manifest v1 contains kind **`p1-diagnostic-only`**, canonical output descriptors,
descriptor SHA-256 and sorted whole-file SHA-256 digests. Artifact identity is
SHA-256 of canonical typed manifest JSON, without a self-referential ID field.
Source fingerprints, private plan/checkpoint, RNG/method control metadata,
Client Secret, prompts and original IDs are excluded.

`artifact.Verify(root)` / `aloeforge verify --artifact-root PATH` rejects extra
files, symlinks, unsupported kind/version, noncanonical headers/descriptors,
unexpected tensor metadata and content/file hash mismatches. Hashing is streamed.
Inputs must remain immutable during verification. Diagnostics are static bounded
strings; CLI summaries expose counts and known identities/hashes only.

**Never send P1 diagnostic artifacts to an untrusted server.** Structural/content
consistency does not prove obfuscation, semantic correctness, external signer
authenticity or privacy. No production artifact kind or remote loader is admitted.
P2–P8 remain unimplemented; see `P1_VALIDATION.md` for actual commands and RSS.
