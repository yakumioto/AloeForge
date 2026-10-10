# P5 Qwen2.5 dense diagnostic compiler and independent runtime

DIAGNOSTIC / ENGINEERING INTERPRETATION — complete dense decoder graph for the
explicit Qwen2 semantics below. This exact-isometry, noise-off ablation is
**not Algorithm 1**, full AloePri V2, encryption, production admission or a
privacy guarantee. The sparse coordinate map is unsuitable for privacy claims.
P4's unresolved choices and human review still block production/paper-reference.
Artifacts are custom, not vanilla HF/vLLM compatible; keep them on trusted systems.

## Adapter and private graph

The Go architecture registry admits `qwen2` / `Qwen2ForCausalLM` after P1
inspection. Capability follows exact semantics, not a checkpoint's marketing name.
Llama/Qwen3/MoE/MLA, quantization, remote code, scaled/partial RoPE, QK norms,
fused QKV, sliding attention and unknown config fields fail closed. Explicit
`use_sliding_window=false`, SiLU and tied flag are required. Absent dropout means
the Qwen2 inference default zero; nonzero/incorrectly typed dropout fails.
Known provenance/dormant fields are not copied into the artifact.

Logical d equals query heads times even full rotary head dimension; KV heads
divide query heads in contiguous groups. RoPE is HF split-half with theta-derived
frequencies, causal inference and positions starting at zero. Q/K/V require
biases; O and FFN are bias-free. Norm gammas are per-channel. HF matrices are
`[output,input]`, embedding/head `[vocab,d]`.

The exact source inventory is embedding, final norm, optional untied head and,
per layer, two norms, Q/K/V weights/biases, O and gate/up/down weights. Tied source
safetensors must omit the head alias; duplicate physical aliases are rejected.
Missing/extra/mismatched roles, shapes and unsupported dtypes fail. F32/F16/BF16
inputs convert explicitly to finite F32 outputs; software BF16 decoding is not
BF16 GEMM/GPU evidence. Source tokenizer JSON is validated and fingerprinted,
but never published. HF tokenization is outside this runtime's scope.

Private TransformPlan v1 contains inventory and explicit nodes binding operation
revision, source and norm dependencies, distinct output name/shard/physical shape
and compiler-selected raw payload basename. This is an acyclic
source→fold/transform→output graph. Source norms are consumed offline.

Limits: vocab 2–262144; d 2–8192; expansion 4–128 in multiples of four
(`D=d+2h`, even positive h); intermediate 2–32768; layers 1–64; query heads 1–64;
head dimension 2–256 even; context 1–131072; total F32 output ≤4 GiB; finite norm
epsilon `(0,1]`, theta `[1,1e9]`. P1 structural/8 MiB control bounds remain.
BOS/PAD may be absent; EOS has 1–16 sorted unique in-range IDs. Role aliases are
allowed. Unknown special-ID types/counts fail.

## Exact diagnostic math

Vectors are rows. Private P is `[d,D]` with one signed nonzero per physical
column; every logical coordinate occurs at least once. Columns for coordinate i
have magnitude `1/sqrt(count[i])`, giving `P P.T=I_d` and `Q=P.T`.
`Q P` is a rank-d projection, not `I_D`. Extra-coordinate assignments, signs and
physical column order are private diagnostic maps, not the general KeyMat sampler.

Embedding is `E P` with row p[i] representing original i. Input projection gamma
folds offline as `W'_in=W diag(gamma) P` in HF orientation. O/down outputs expand
as `P.T W_out`; final gamma folds into `H diag(gamma) P` before vocabulary row
permutation. A tied source therefore has an explicitly **untied** physical head.
Every residual branch uses the same P.

Runtime norm is `sqrt(d/D) y/sqrt(mean_D(y²)+(d/D) eps_d)`. P4's scaled-isometry
proof makes this the logical normalized row mapped by P. Ordinary physical
RMSNorm without these coefficient/epsilon changes is unsupported.

Private per-layer maps permute KV groups, the same within-query-group order in
every group, and corresponding O columns. Q/K use reciprocal paired split-half
scales `{±1/2,±1,±2}`, including their biases; paired scales commute with RoPE
at unequal positions. V/O use inverse channel scales per shared physical KV group.
FFN gate/up are jointly permuted; up receives diagonal scales, down their inverse.
The nonlinear gate is never scaled. Frequencies remain fixed; no BlockPerm,
gamma/window approximation or added noise is silently introduced.

Independent FP64/FP32 component tests and a scalar source-graph baseline exercise
these choices. Worker and runtime do not import each other or reuse the baseline.

## Secret, bounded worker I/O and recovery

Go requires existing disjoint roots, private 0700 output/secret directories and
creates 0600 canonical `client-secret.json` with flock/fsync/no-replace storage.
It saves a 32-byte crypto/rand seed and unbiased crypto/rand vocabulary permutation
before transformation. Fresh compilation never replaces a secret; resume needs
the original validated material. Source content/inventory, tokenizer, geometry,
roles, method and expansion bind the private context fingerprint.

Binding commitment: SHA-256 of `AloeForge/dense-secret/v1\n` plus Go canonical JSON
of the disk record with empty binding_id, in order schema_version,
context_sha256, seed, permutation, binding_id. This proves consistency, not origin
or privacy. RNG is `crypto-rand-sha256-pcg64` / `1`. Transform substreams use
NumPy 2.3.3 PCG64 seeded by the little-endian first 16 bytes of
`SHA256(seed_bytes || ASCII(label))`; labels `residual-v1`, `layer-N-v1`.
Saved material and bound revisions, not a regenerated seed, determine recovery.

Go invokes the explicitly selected Python interpreter with ≤8 MiB JSON plan on
stdin and private source/secret/scratch paths. The CPU NumPy worker maps source
payloads read-only and transforms ≤64 output rows per block into private raw F32
files. No PyTorch download is needed in CPU CI; the controlled real-model HF
baseline uses pinned PyTorch separately. Block heap usage is bounded by width;
mmap page residency, total RSS and scratch storage are separate concerns.

Stdout is bounded descriptor JSON; stderr is discarded. Go returns static errors,
checks exact file count/names/regular type, shape/dtype/length and recomputed hashes,
then rebinds source content and metadata. Go alone writes the P1 stage, checkpoint
and verified atomic publication. Worker cannot write published control files.
CLI deadline is one hour; package Compile takes a cancellable context.

Resume recomputes payloads in fresh scratch and uses P1 recovery to verify/skip
completed shards. A missing stage can be created using an already durable matching
secret; changed/corrupt/missing source/secret/method fails. Scratch is removed on
ordinary exit; abrupt process death can leave private `.dense-worker-*` directories
for operator cleanup. Source/published files must stay immutable. P1 trusted local
filesystem assumptions and exclusions for privileged concurrent writers still apply.

## Artifact and independent loader

Manifest v1 kind `p5-qwen25-dense-diagnostic-only` adds strict dense_contract,
complete output descriptors and file hashes. Public fields contain geometry,
mapped special IDs, binding/tokenizer identity and loader/cache/method/RNG revisions.
No seed, permutation, source fingerprints/config/tokenizer, original IDs,
private plans or Client Secret is copied. Existing P1/P2 encodings remain unchanged
through omitted extension fields. Diagnostic artifacts must not be deployed to
untrusted servers or connected to P3/P6 as a production backend.

`aloeforge_reference.dense_runtime.Runtime.load` requires external trusted artifact
ID, binding ID and tokenizer fingerprint. It reads only publication, rejects
extra/unknown/symlink files and metadata, independently reconstructs roles/shapes
and canonical headers, streams file/tensor hashes and finite checks, and maps the
same validated open files read-only. Runtime identity hashes exact canonical
manifest bytes from Go publication; reformatted JSON is rejected. Hashes without
trusted external identities cannot authenticate attacker-rewritten artifacts.
Go `verify-dense` also validates the external Client Secret commitment.
The trusted Go Secret exposes bounded MapIDs/RestoreIDs (≤4096 per call) and
finite RestoreLogits; its normal formatting/JSON never exposes private maps.

Only physical IDs enter runtime. `forward(ids,cache=None)` supports prefill/decode
with contiguous positions. Cache has `[kv_head,position,head_dim]` per layer and
one runtime owner; dtype/shape/length/nonfinite mismatches fail. `greedy` returns
bounded argmax physical IDs and stops at mapped EOS. Default diagnostic compute
is FP64; FP32 is explicit. Weight admission defaults to 4 GiB, cache allocation
512 MiB, attention buffers 64 MiB, activations 128 MiB and logits 256 MiB, checked
before inference allocation. Nonfinite residual norms fail closed. These
are specific buffer budgets, not a process RSS promise. No batching, padding,
sampling, beam search, sliding/scaled/partial RoPE or GPU/vLLM kernel is admitted.

Fixed gates: component FP64 atol=rtol=2e-11, FP32 3e-4; full-model restored logits
atol=rtol=3e-4; mean teacher-forced NLL delta ≤4e-4; synthetic split-cache
atol=rtol=2e-6; greedy exactly equal. Gates are not adjusted from results.
See P5_VALIDATION.md for measured evidence and pending real-model quality gates.

Greedy ties use the lowest **physical** ID, as NumPy argmax specifies. Equal source
maxima need not preserve the source's lowest-ID tie break under token permutation.
No source-rank side channel is published to repair this. The strict external
greedy equality gate remains in force and rejects any such divergent case.
