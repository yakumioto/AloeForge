# Multi-model Adapter contract

Proposed Go interface (not implemented):
```go
type ArchitectureAdapter interface {
    ID() string
    Detect(inv ModelInventory) (CapabilityReport, error)
    BuildIR(inv ModelInventory) (ModelIR, error)
    BindTensors(ir ModelIR) ([]TensorBinding, error)
}
```

`ModelIR` records architecture ID, tokenizer fingerprint, original and physical Tensor inventories, physical/logical hidden dimensions, attention head/KV groups, norm eps, RoPE ordering, FFN activation, per-layer tensor roles, MoE router/expert semantics, MLA projections and supported sharding/quantization. `RuntimeContract` declares custom loader/cache expectations; `MethodContract` pins transform revision; `SecretBinding` ties public artifact ID to client-side secrets without exposing secrets.

Order: SmolLM2 token permutation regression → Qwen2.5-0.5B full dense → Llama dense → Qwen3 Dense/QK Norm → Qwen3 MoE → DeepSeek MoE/MLA → separately verified quantization and TP/PP/EP.

Reject unsupported/ambiguous layouts and fused QKV, unrecognized architecture, quantization, GQA/rotary mode, missing/duplicated weight aliases. Do not infer full capability from model name alone.
