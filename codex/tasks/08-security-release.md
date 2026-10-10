# Codex task 08 — security, performance and production gates

Prerequisites: stable dense model, client/server protocol, GPU backend.

Specify adversary knowledge and visibility (known source model, known plaintext, multi-query access, activations, cache and weights). Build attack harnesses and report recovery metrics with seeds, datasets, query budget and baseline. Audit logs, secrets, path traversal, model isolation, ABI version mismatch and key rotation.

Add GPU benchmarks and scalability tests, separate signed artifacts/secret provisioning, reproducible releases. Encryption-at-rest/TLS are separate controls; token permutation is not encryption. State what is **not proven**, especially malicious server and cryptographic-level security.

No-go: claims of end-to-end secrecy without reproducible empirical evidence and explicit security assumptions.
