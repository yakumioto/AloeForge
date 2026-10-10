# Codex task 06 — vLLM model plugin and GPU integration

Prerequisite: 05 passed reference-runtime model-level gate.

Implement Python out-of-tree custom model loader/executor for AloePri physical dimensions. Pin exact tested vLLM/PyTorch/CUDA versions. Integrate obfuscated token-ID path, prefill/decode, attention group/RoPE contract, KV cache, generation stop semantics and Go gateway relay.

Codex Cloud tasks should implement and test pure CPU/mock interfaces; external GPU runner must execute actual vLLM tests. Explicitly compare reference backend parity, TTFT, TPOT, throughput and GPU memory with identical workload.

Do not claim working PagedAttention or GPU throughput without real hardware evidence.
