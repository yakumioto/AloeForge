# P6 preparation validation

Base: merged P5 main `1e31b46`. CPU/API preparation, not completed P6. No Qwen
checkpoint, vLLM installation, CUDA device or GPU metric was available. Task06's
real-model prerequisite remains pending. See VLLM_BACKEND_CONTRACT.md.

## Executed local checks

Go1.25.3, Python3.12.14, NumPy2.3.3, pytest8.4.2, Ruff0.14.0, PyTorch2.8.0+cpu.
Ignored work/tools/go supplies Go; work/go.mod isolates extracted toolchain tests.

```sh
.venv/bin/python -m pip install 'torch==2.8.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e './python[dev]'
PATH="$PWD/work/tools/go/bin:$PATH" make check
PATH="$PWD/work/tools/go/bin:$PATH" go test -race ./internal/gateway ./internal/clientruntime
PATH="$PWD/work/tools/go/bin:$PATH" go test -race ./...
PATH="$PWD/work/tools/go/bin:$PATH" .venv/bin/python -m pytest -q -c python/pyproject.toml python/tests/runtime
```

Focused runtime suite: **11 passed**. Four independent FP32 model cases cover
MHA/GQA, tied/untied, folded nonunit norms, QKV bias, F32/F16/BF16 sources and
head_dim4; a head_dim64 case verifies explicit mock rejection and continued P5
FP64 reference admission. Other cases cover strict weights/config/artifact
negatives, explicit
pinned-API mocks and bounded workload/probability/metric gates. Mocks replace CUDA
and vLLM modules deliberately; no hardware is inferred. Go relay/TLS client and
focused race tests passed. GPU CLI help and failure redaction execute without an
engine. Final `make check`: **223 passed in 26.76s**, Go unit/vet, Ruff lint and
format passed, exit0. Full `go test -race ./...` passed, exit0. No local CPU test
was skipped (Torch was installed). After moving the CPU geometry guard ahead of
parameter allocation, the focused runtime suite was rerun to verify that rejection.

The head_dim64 case exposed CPU FP32 GEMM/GEMV reduction drift, failing the
fixed 2e-6 split/cache checks. Fixed row order/balanced FP32 sum trees reduced
drift, but a fresh-transform full-suite run still failed the wider cross-reference
cache gate (222 passed, one failed). Wider CPU mock geometry is now rejected,
not numerically admitted. Thresholds remain unchanged. Wide FP32/cache behavior
and GPU batched arithmetic remain unresolved numeric gates.

One disposable two-case head_dim64 publication (vocab71, two layers, physical
width132, intermediate70) yielded max absolute logit error **7.7827e-6**, max
split-logit difference **2.3842e-6** and exact greedy **2/2** versus FP64 reference.
This exploratory run did not pass the complete cache gate and does not admit
head_dim64 CPU execution. These are maxima, not absolute-only thresholds;
gates use atol+rtol. Fresh transforms can change numbers. Only aggregates printed.

CI keeps the NumPy/base job and adds a separate runtime-cpu job requiring
torch==2.8.0+cpu before its 11 tests. Base pytest skips the Torch module when absent;
the separate job prevents a missing dependency from satisfying the CPU model gate.
Neither job installs vLLM/CUDA or downloads checkpoints.

## Separate controlled GPU recipe (not executed)

On the trusted offline checkout, create a fresh private scratch root:

```sh
go build -o work/aloeforge ./cmd/aloeforge
.venv/bin/python python/scripts/p6_prepare_synthetic.py \
  --root work/p6-controlled-fixture --cli work/aloeforge
```

The utility downloads no model, creates a head_dim64 publication plus external
identity/workload files, and prints no secret/IDs. Copy only `output/published/`,
`identity.json` and `workload.json` to a **trusted controlled** GPU runner. Never
copy source/secret directories or deploy these diagnostic weights untrusted.

Create a separate Python3.12.14 environment; install
`python/requirements-p6-gpu.txt` and the local package (`pip install -e ./python`).
Record physical NVIDIA GPU, driver, CUDA runtime and installed package versions.
The candidate versions have not executed together here. Read the three hashes
from the trusted external identity file and run:

```sh
python python/scripts/p6_validate_gpu.py \
  --artifact-root /controlled/publication \
  --artifact-id EXPECTED_ARTIFACT_SHA256 \
  --binding-id EXPECTED_BINDING_SHA256 \
  --tokenizer-sha256 EXPECTED_TOKENIZER_SHA256 \
  --workload /controlled/workload.json --allow-diagnostic
```

Only public path/hash arguments and physical synthetic IDs enter the runner.
Missing hardware/version, integrity, parity, greedy/cache/stop or metric failure
returns exit1 with static `gpu_gate=not_passed`. Success would pass only this
bounded synthetic experiment; full P6/real-model/production remain pending.

Skipped/unresolved: real Qwen2.5-0.5B quality; P4 method/human review; actual vLLM
import/boot/CUDA prefill/decode/PagedAttention/cache evidence; GPU raw-logit parity;
hardware cancellation/backpressure/abort and cache/memory pressure; matched GPU
reference performance; dense Go/Python IPC/TLS binding; batching stress, TP/PP,
LoRA/quantization/prefix caching/chunked prefill; production/security admission.
No GPU throughput, TTFT, TPOT or memory result is reported passed here.
