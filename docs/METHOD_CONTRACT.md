# AloePri V2 method contract — audit required

**Status:** provisional; no full V2 implementation or security guarantee has been established.

## Evidence labels
- PAPER: exact AloePri V2 assertion, with section/equation/algorithm reference.
- INTERPRETATION: engineering decision not uniquely determined by published text.
- DIAGNOSTIC: temporary ablation not claimable as full V2.

## Mathematical requirements to audit
- KeyMat shapes `P:[d,d+2h]`, `Q:[d+2h,d]`, with `P @ Q ≈ I_d` (not necessarily `Q @ P = I`).
- Validate embedding, residual stream, RMSNorm and attention/FFN projection spaces jointly. Do not blindly apply one generic weight rule irrespective of matrix orientation.
- Token permutation, optional noise, all original and permuted EOS/BOS/PAD IDs, tied vs untied head aliases.
- Q/K/V/O, RoPE pair ordering, group-specific GQA/MQA/MHA transforms, BlockPerm, prefill/decode cache geometry.
- RMSNorm: raw L2 vs RMS correction, expansion d→d+2h, epsilon and kappa scaling.
- FFN gate/up/down, MoE router/expert permutation, MLA low-rank and decoupled RoPE.
- Distinguish exact covariance from floating-point effects, approximate operators and intentional added noise.

## Profiles
`paper-reference`: literal source statements and flagged unknowns.
`diagnostic`: noise-off / ablations for debugging only.
`production`: only after explicit numeric, semantic and attack gates.

Use separate Python FP64 reference and Go/Python production worker implementations; do not reuse the same calculation to assert its own correctness. Compare logits after mapping to original vocabulary coordinate space. Reference: https://arxiv.org/abs/2603.01499v2
