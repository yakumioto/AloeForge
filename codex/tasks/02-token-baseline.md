# Codex task 02 — verifiable token-permutation baseline

Prerequisite: 01 merged.

Implement bijective vocab permutation, inverse lookup and tied/untied embedding + LM-head mapping, external versioned client secret, binding commitment and resume semantics. Check original vs obfuscated model logits only after remapping vocab coordinates. Use independent Python oracle and synthetic/tiny model fixtures.

Token-permutation mode is a scaffold and diagnostic baseline **not** complete AloePri V2 or a cryptographic encryption scheme. Never store permutation/key material in server artifact or printed logs.

Acceptance: secret mismatch rejection, corrupted permutation tests, special ID mapping, equality on known synthetic cases, stable resuming, Go/Python independent verification, CPU CI.
