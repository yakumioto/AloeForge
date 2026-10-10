# P5 validation: Qwen2.5 diagnostic dense compiler/runtime

DIAGNOSTIC — CPU compiler/expanded artifact/inference slice implemented.
The complete P5 gate is **pending**: no real Qwen2.5-0.5B checkpoint/dataset,
physical GPU or reviewed production method was available in this cloud session.
No real-model quality, GPU, performance, privacy or full AloePri V2 claim is made.
See DENSE_COMPILER_CONTRACT.md and the unresolved P4 audit.

## Executed CPU checks

Go 1.25.3, Python 3.12.14, NumPy 2.3.3, pytest 8.4.2, Ruff 0.14.0.
The default `/usr/bin/go` was an unrelated executable; official Go was installed
in ignored `work/tools/go`. An ignored `work/go.mod` isolates toolchain regression
sources from the repository's `./...` package scan. An initial make/race attempt
encountered these toolchain sources; after isolation the checks below use only
repository packages. This was a local setup correction, not a weakened check.

```sh
PATH="$PWD/work/tools/go/bin:$PATH" make check
PATH="$PWD/work/tools/go/bin:$PATH" go test -race ./...
PATH="$PWD/work/tools/go/bin:$PATH" .venv/bin/python python/scripts/p5_measure_dense.py
git diff --check
```

Results (all exit 0):

| Check | Observed result |
| --- | --- |
| make check | Go formatting/vet/unit tests, Ruff lint/format; **210 Python tests passed**, 36.12 s |
| go test -race ./... | All repository packages passed |
| P5 numeric harness | Six independent whole-graph comparisons; greedy **6/6 exact** |
| Restored logits, worst absolute error | **7.716956547110243e-7** |
| Single-token decode vs prefill, worst absolute error | **1.3113021850585938e-6** |
| Mean teacher-forced NLL, worst absolute delta | **6.524090645498859e-8** |
| External runner --help / dataset validation | Help exit 0; seven bounded malformed-dataset cases in pytest |
| git diff --check | Clean |

The numeric table records one executed six-case run. Source fixture seed is 6;
each invocation draws fresh crypto/rand private material, so measured maxima vary
while gates stay fixed. Harness scratch/key fixtures are removed; normal compiler
outputs/secret persist for exact resume. No private RNG material is logged.

All gates retain
their predeclared thresholds: component FP64 2e-11, FP32 3e-4; restored model
logits atol=rtol=3e-4; mean NLL delta 4e-4; cache atol=rtol=2e-6; exact greedy.

Coverage includes tied/untied heads, F32/F16/BF16 source conversion, FP64/FP32
runtime, MHA/MQA/GQA with multi-KV and within-group permutations, nonzero QKV
biases, non-unit norm gamma, unequal rotary positions, two/three-layer residual
graphs, 64-row streaming boundaries, teacher-forced logits/NLL, greedy, split
prefill/decode, subprocess artifact-only loading, source/secret removal and stable
resume. Independent scalar source code and worker component tests provide separate
mathematical/model-level baselines; the runtime imports no worker or private key.

Negative coverage: unknown/missing/extra roles, shapes/dtypes, biased/fused/QK-norm
variants, sliding/partial/scaled rotary, quantization, source/tokenizer/expansion
changes, corrupt/missing/noncanonical secret, mixed artifact profiles, nonfinite
source/output, corrupted header/file/payload/descriptor, extra secret fields/files,
symlinks, wrong public/external identities, unknown revisions, malformed special
IDs, cache owner/geometry/dtype/nonfinite data, out-of-context/invalid IDs, bounded
allocation, worker stderr redaction and malformed bounded external evaluation data.
P1 crash/recovery tests remain part of make/race, and P5 resumes the same stage API.
P5-specific physical process-death/RSS fault experiments were not run.

## Local usage

Source root is an exact local HF Qwen2.5 dense safetensors checkpoint. Output and
Client Secret roots must already exist, be 0700, and be disjoint from source and
each other. Keep generated files under ignored private directories.

```sh
go run ./cmd/aloeforge compile-dense \
  --model-root work/source --output-parent work/output --secret-root work/secret \
  --stage .stage-qwen25 --target qwen25-diagnostic \
  --python "$PWD/.venv/bin/python" --expansion 4 > work/identity.json
# Repeat the identical command with --resume after interruption/lost acknowledgement.
go run ./cmd/aloeforge verify --artifact-root work/output/qwen25-diagnostic
go run ./cmd/aloeforge verify-dense \
  --artifact-root work/output/qwen25-diagnostic --secret-root work/secret \
  --artifact-id EXPECTED_SHA256 --tokenizer-sha256 EXPECTED_SHA256
```

Trusted caller obtains identities from compiler stdout and maps token IDs with
its separate Client Secret. An independent runtime receives publication and only
those expected public identities:

```python
from aloeforge_reference.dense_runtime import Runtime

runtime = Runtime.load(
    "work/output/qwen25-diagnostic",
    artifact_id=expected_artifact_id,
    binding_id=expected_binding_id,
    tokenizer_sha256=expected_tokenizer_sha256,
)
logits, cache = runtime.forward(physical_input_ids)
next_logits, cache = runtime.forward(physical_next_ids, cache)
generated_physical_ids = runtime.greedy(physical_input_ids, max_new_tokens=8)
```

## Pending controlled Qwen2.5-0.5B quality job

No external runner was attached/scheduled, so this is a reproducible job recipe,
not evidence of a run. Provision a controlled runner with the pinned official
Go/Python tools, repository commit, immutable `Qwen/Qwen2.5-0.5B` (or the explicitly
chosen dense variant) snapshot at an exact 40-hex HF revision and a fixed annotated
task dataset. Keep snapshot/secret/dataset on that trusted runner; do not log them
or upload diagnostic artifacts to an untrusted server.

The independent HF baseline uses pinned `torch==2.8.0`,
`transformers==4.57.1`, `tokenizers==0.22.1`, `safetensors==0.6.2`, NumPy 2.3.3
from `python/requirements-p5-external.txt`. This separate dependency set and actual
HF loading have **not** been exercised in the current CPU environment.
The runner installs locally/through its configured policy; scripts never download
checkpoint/data or execute remote model code.

Dataset v1 is bounded JSON with exactly schema_version=1, checkpoint_revision
matching the pinned revision, and 1–64 cases. Each case has trusted-client
`input_ids` and annotated `target_ids` (each length 1–128). Build it using the
pinned tokenizer/task definitions outside server scope. The caller's asserted
checkpoint revision is provenance, not independent proof that a local path came
from that revision; immutable snapshot provisioning plus compiler source binding
must supply that assurance.

```sh
.venv/bin/python -m pip install -r python/requirements-p5-external.txt
# Compile snapshot with compile-dense and retain work/identity.json privately.
.venv/bin/python python/scripts/p5_validate_qwen25.py \
  --model-root work/source --artifact-root work/output/qwen25-diagnostic \
  --secret-root work/secret --identity-file work/identity.json \
  --dataset work/evaluation.json --checkpoint-revision PINNED_40_HEX_REVISION \
  --cli work/aloeforge --device cpu --max-new-tokens 8
```

The runner first uses Go verify-dense, then loads artifact alone in NumPy FP32
and source independently in HF/PyTorch FP32 eager attention with remote code and
model downloads disabled. It verifies restored teacher-forced logits at fixed
3e-4 atol/rtol, exact greedy sequences, annotated target-token accuracy retention
and mean NLL difference ≤4e-4. It emits aggregate scalar counts/errors/NLL and
perplexity ratio, no inputs/logits/keys. Failures return static diagnostics and
exit 1, without lowering thresholds. `--device cuda` is allowed only on a physical
GPU runner; the report distinguishes GPU source baseline from CPU reference,
and is not a P6 vLLM/cache/performance certification.

Remaining work before full P5 admission: execute and archive pinned real-model
task evidence, review paper ambiguities/actual desired transform, and define any
non-diagnostic runtime/security acceptance. Sparse diagnostic isometry and the
omitted BlockPerm/noise/general KeyMat choices must not silently become full V2.
