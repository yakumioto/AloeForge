# Codex task 07 — multi-model adapter expansion

Prerequisite: 05 stable, 06 reference GPU gate established.

Introduce new model families independently: Llama dense, Qwen3 dense Q/K Norm, Qwen3 MoE, then DeepSeek MLA/MoE. Each adapter must validate config and every expected tensor, physical and logical sizes, rotary/head packing, quantization and sharding restrictions. Unknown model types must fail closed.

For MoE test expert router permutation, top-k weights, expert placement and shared experts. For MLA test low-rank projections, decoupled RoPE and real KV-cache representation, with independent oracle. Distributed and quantized support requires separate contracts and benchmarks.

One model family per PR and model-specific correctness evidence. Do not infer support from the model's marketing name.
