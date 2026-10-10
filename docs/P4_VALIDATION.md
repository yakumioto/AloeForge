# P4 independent mathematical oracle validation

DIAGNOSTIC — task 04 only. Based on merged P3 PR #7, main
`ce42ae1e743fc432a397042acfdd3716e451fc79`. The V2 source/shape/algebra self-audit
and proofs are in [V2_MATH_AUDIT.md](V2_MATH_AUDIT.md); human method review is pending
in this PR. Production and unqualified paper-reference admission remain blocked.
No compiler/worker/real model/runtime implementation or privacy claim is supplied.

## Actual tools and commands

Linux amd64; official Go 1.25.3, Python 3.12.14, NumPy **2.3.3**, pytest 8.4.2,
Ruff 0.14.0. NumPy is pinned in `reference` and `dev` extras; base/worker runtime
dependencies remain empty. The existing editable dev install was updated, and
`pip check` found no broken requirements. No Torch/CUDA/model download.

Commands actually completed with exit 0:

```sh
.venv/bin/python -m pip install -e './python[dev]' numpy==2.3.3
.venv/bin/python -m pip check
.venv/bin/python -m ruff check python
.venv/bin/python -m pytest -q python/tests/reference/test_v2_math.py
.venv/bin/python python/scripts/p4_measure_math.py
PATH="/workspace/work/tools/go/bin:$PATH" make check
git diff --check
```

- Focused P4 tests: **70 passed in 0.34s**.
- Final `make check`: Go formatting/vet/tests and Ruff lint/format passed;
  **146 Python tests passed in 0.86s**, 15 Python files already formatted. No skips.
- Numerical harness: **100 exact comparisons per dtype**; observed worst absolute
  error FP64 **1.5543122344752192e-15**, FP32 **4.76837158203125e-7**.
- Fixed component/full-block tolerances were declared before results: FP64
  atol=rtol=2e-11; FP32 atol=rtol=3e-4. Key/null checks use 2e-12 / 3e-5;
  BF16-conversion toy rounding comparison uses atol=rtol=2e-2. None weakened.
- CPU CI runs `make check`, the scalar-error harness, and retains all existing
  complete Go race/header/permutation/SSE/UTF-8 fuzz checks. Local Go production
  code is unchanged; `make check` ran Go tests/vet, and reused their valid cached
  results. New mathematical gates are Python CPU tests, not a GPU/backend test.

The public-only measurement output was:

```json
{"exact_comparisons":{"float32":{"comparisons":100,"max_absolute_error":4.76837158203125e-7},"float64":{"comparisons":100,"max_absolute_error":1.5543122344752192e-15}},"fixture_seeds":[0,11,23,16,8,42,35,7,9,17],"numpy":"2.3.3","profile":"diagnostic"}
```

The harness prints no tensor entries, source weights, prompts or keys. Seeds are
public synthetic PCG64 fixture seeds, not production secret seeds. Test fixtures:
logical d=6, h=2/4, physical D=10/14; vocabulary 7, FFN width 8; head dimension 6,
sequence length 4 with positions [0,1,4,9]. MHA uses 3 query/3 KV heads, GQA 4/2,
MQA 4/1, in both interleaved and split-half layouts. Generic toy projection
geometry is not an admitted checkpoint architecture. KV groups use different
frequencies/rotations/scalings/value maps, with a shared map only within each group.
The key family uses seeds 0/11/23 and three distinct members per family.

## Independent and falsifiable evidence

`python/tests/reference/math_baseline.py` implements matrix multiplication,
RMSNorm, SiLU FFN, causal attention/RoPE and a complete toy decoder with Python
scalar arithmetic and `math.fsum`. It imports **neither NumPy nor the oracle nor
the worker**. The NumPy transformed path compares against that scalar baseline,
including all restored vocabulary logits and exact greedy IDs, not an unmapped
logit array or a reused worker transform. Isolated-process tests verify importing
the oracle does not import the worker or Torch.

P/Q tests verify shapes, right inverse, cross-family-member cancellation,
logical recovery and same-coordinate residual addition. They deliberately reject
a nonsymmetric transposed inverse, mismatched families, singular/ill-conditioned
matrices, nonfinite/mixed/unknown dtypes and invalid bounded parameters. QP is
shown to differ from the physical identity. Copies cannot alias caller key arrays.

Embedding/head tests cover tied/untied **source** heads, original/physical row and
column directions, exact restored logits in noise-off, and supplied-noise logits
against a separate noisy scalar baseline. Non-isometric transformation can break
a source tied alias; the oracle never silently retains it. Explicit noise differs
from clean weights and is not mistaken for rounding error or privacy validation.

Attention tests cover MHA/GQA/MQA, group/within-group permutation, intra-pair
rotations/scales, shared KV-group V/O cancellation, both RoPE layouts and causal
prefill/prefix equivalence. A three-cycle proves the literal opposite-Z Q/K formula
breaks raw scores at position zero. Relocating frequency coordinates makes a full
pair permutation exact; retaining the old frequencies has visible approximation
drift. KV-only shuffling and wrong pair layouts also fail equality. Explicit full
bounded windows reject missing tails and cross-window movement; the unresolved
paper BlockPerm sampler fails explicitly instead of inventing gamma or hanging.

FFN tests show correct gate permutation/up scaling/down cancellation and that
scaling a nonlinear gate fails. RMSNorm tests cover non-unit gamma, expanded
scaled-isometry correction, epsilon scaling, zero rows, explicit empirical L2/RMS
conventions and the counterexamples to raw-L2 kappa and to a universal scalar for
anisotropic P. These approximation tests deliberately assert nonzero drift;
they do not certify the paper's generic normalization accuracy.

BF16 uses a separate scalar `struct`-bit nearest-even baseline with halfway ties,
signed zero, negative values, subnormals and overflow/nonfinite rejection. A toy
attention run with converted inputs/weights has bounded rounding drift; operations
run in FP32. This does not certify BF16 GEMM, native BF16, GPU or real-model quality.

The first focused run had four counterexample-fixture failures: very small random
inputs made fixed-frequency output drift smaller than the declared 0.001 negative
gate. The counterexample now uses deliberately stronger rows (4×), preserving
that gate and every precision acceptance tolerance. All exact relocated-frequency
comparisons still pass. Unsupported-dtype/shape/scale/profile/noise/layout/MLA/MoE,
scalar coercion and transformed-allocation bound negatives all pass.

## Exclusions and remaining blockers

No CPU tests skipped; main/version have no standalone Go files and remain covered
by existing CLI tests. No real HF tokenizer/model/checkpoint, production transform
worker/compiler/artifact, Python IPC/remote loader, MoE/MLA/partial-or-scaled RoPE,
GPU/CUDA/Torch/vLLM, native BF16 computation, privacy/attack or performance evaluation.
No Rust, legacy-repository modifications or P5–P8 code. P0 draft contracts remain
explicitly unimplemented; P1/P2/P3 production code is unchanged.

Paper section/page/formula ambiguities are individually registered with their
engineering interpretation and falsifiable test. They remain production blockers,
even when noise-off mathematical diagnostic tests pass. Numerical correctness is
not encryption, model-quality admission or security evidence. QR/SVD/PCG64 and
bounded well-conditioned keys are reproducible diagnostic choices, not a production
key distribution. Human method review and independent real-model/privacy gates
remain outstanding for later authorized tasks.
