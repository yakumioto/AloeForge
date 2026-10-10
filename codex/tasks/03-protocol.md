# Codex task 03 — ClientRuntime and Gateway protocol slice

Prerequisite: 02 merged.

Create Go client library and Go HTTP+SSE gateway with a deterministic **fake token inference backend** only. Define GET model capability/secret binding handshake; POST obfuscated token IDs; SSE sequenced obfuscated output ID events; client-side inverse permute and incremental UTF-8 decoding.

Handle EOS/BOS/PAD, stop tokens, text stop-string local handling, auth, cancellation, bounded buffers, model-binding errors, partial stream and request-id semantics. No plaintext in server logs; test deliberately with forbidden marker prompts to catch accidental logging.

Acceptance: local end-to-end test on CPU with no model; no auto text decode/encode round-trip; recover exact expected token sequence and negative security tests.
