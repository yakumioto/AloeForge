# Codex task 05 — Qwen2.5 dense full-graph compiler

Prerequisites: 01, 02, 04 merged.

Build architecture registry and Qwen2.5 dense adapter with explicit tensor roles, rotary layout, GQA grouping, norms and tied weights. Go creates `TransformPlan` DAG and controls secrets/atomic staging. Offline Python worker executes V2 transformations with bounded tensor I/O and returns verified output descriptors. Build a separate Python reference Runtime able to load the nonstandard expanded artifact.

Run local synthetic model tests in Codex CPU; schedule Qwen2.5-0.5B actual model tests in an external runner. Preserve numerical provenance and method revision.

Acceptance: no missing/mismatched tensors, no secret in server artifact, full model loads in independent reference backend, teacher-forced logits/greedy and task quality measured (never silently loosen gates); report GPU unavailable separately if applicable.
