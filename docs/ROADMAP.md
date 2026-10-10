# Executable Codex Cloud roadmap

One scoped PR per task. PR must state files changed, actual tests run, tests skipped, remaining risk and gaps.

| Stage | Deliverable | Gate |
|---|---|---|
| P0 | Go/Python module skeleton, AGENTS, CPU CI, contracts | go test/vet, pytest, Ruff, make check |
| P1 | Strict HF/safetensors inventory, writer/manifest/checkpoint | malformed index/header/path and crash tests |
| P2 | Token-permutation baseline + secret binding | independent mapping/logit oracle |
| P3 | Go client/gateway + fake token backend, SSE | error/stop/UTF-8/cancel/security tests |
| P4 | Independent V2 math reference | FP64/FP32, GQA/RMSNorm/RoPE/FFN, ambiguity audit |
| P5 | Full Qwen2.5 dense Go/Python compiler + reference runtime | independent load/generation/quality gate |
| P6 | vLLM custom-model GPU backend | external GPU prefill/decode/cache/performance tests |
| P7 | Llama/Qwen3/MoE/MLA separate adapters | per-architecture exact schema/semantics gates |
| P8 | Attack evaluation, key rotation, multi-GPU, release | reproducible security/perf benchmark |

Cloud CPU: Go unit/vet, Python unit/Ruff, toy Tensor/Protocol tests. Controlled model runner: pinned HF checkpoint. GPU: pinned PyTorch/CUDA/vLLM with physical GPU. Security: explicit attacker/dataset/query budget. No phase marked complete without its gate.

P5 status: CPU diagnostic compiler/expanded artifact runtime and synthetic gates
implemented. Controlled Qwen2.5-0.5B quality and P4 human/method admission are
pending, so the full P5 gate is not marked complete. See P5_VALIDATION.md.

P6 status: CPU/API preparation only. Independent PyTorch graph, strict custom
loader/plugin candidate, P3-profile Go mock relay and separate synthetic GPU runner
implemented. Wide FP32 CPU cache parity is unresolved and rejected by the mock.
P5 real quality prerequisite, actual GPU/engine/cache/performance and dense wire
integration remain pending. Full P6 is not marked complete. See P6_VALIDATION.md.
