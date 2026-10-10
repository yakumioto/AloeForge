# Client/Server token-only protocol — proposed

**Token IDs are obfuscated, not encrypted.** Transport MUST use authenticated TLS; avoid logging IDs or application plaintext.

## Handshake
`GET /v1/models/{model_id}/runtime` yields protocol revision, artifact ID, method revision, public binding ID, tokenizer fingerprint, supported features, mapped special token IDs and context bounds. Go ClientRuntime refuses any mismatch.

## Request
```json
{
  "protocol_version": 1,
  "model_id": "qwen25-v2-demo",
  "artifact_id": "sha256-or-blake3-identity",
  "binding_id": "public-binding-identifier",
  "request_id": "unique-id",
  "prompt_token_ids": [7,42,13],
  "sampling": {"temperature": 0.0, "max_tokens": 16},
  "stream": true
}
```

## Stream
SSE events contain `request_id`, monotonic `sequence`, obfuscated `output_token_ids`, `finish_reason` and explicit errors; never plaintext text. Go client implements local tokenizer/chat template, forward/inverse mapping, incremental UTF-8 decode, local plaintext stop strings and partial-stream handling. Go Gateway checks auth and binding, forwards/cancels with bounded backpressure. Python backend executes only obfuscated model. EOS/BOS/PAD/forced tokens/logit bias/allowed sets/stop IDs must be transformed into the correct coordinate space. Reject unsupported text-mode features; no silent decode-encode fallback. Retries after partial streams cannot silently duplicate output. Prefix/KV cache scoped by artifact and binding ID.
