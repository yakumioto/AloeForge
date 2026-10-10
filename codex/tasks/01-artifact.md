# Codex task 01 — secure safetensors/manifest foundation

Prerequisite: 00 merged.

Build a strict Go model inventory: HF config/index discovery; safetensors header parsing with bounded lengths; tensor offset/shape/dtype and shard completeness checks; canonical `ModelInventory` IR. Implement output-driven tensor descriptors, stage-only writer contract, manifest hashing, checkpoint binding and atomic publish rules. Python may read payloads but does not bypass Go verification.

Reject duplicate keys, traversal paths, symlinks escaping model root, overlapping/out-of-bounds tensors, contradictory index entries and ambiguous architecture. Do not load entire model weights into Go memory.

Tests: synthetic shard fixtures including malformed cases; stable descriptor hashes; crash/restart recovery; no output publication after verification failure; verify commands and peak RSS for synthetic large-header/multi-shard tests.
