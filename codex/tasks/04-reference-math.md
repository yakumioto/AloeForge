# Codex task 04 — V2 independent mathematical oracle

Prerequisite: paper V2 math audit in `docs/METHOD_CONTRACT.md` reviewed.

Implement standalone FP64/FP32 Python reference for P/Q (d×(d+2h), (d+2h)×d), residual-space transformations, embedding/head, attention Q/K and V/O, group head permutation, RoPE/BlockPerm, RMSNorm scaling and FFN, with separate diagnostic noise-off mode. Do not import production transform worker code into oracle.

Each paper ambiguity must be documented with section/page, exact formula, engineering interpretation, and falsifiable test. Never claim the paper supplied omitted gamma/window/MLA details. Test toy tensors across MHA/GQA/MQA and BF16 conversion.

Acceptance: math proofs/tests pass against independent baseline with documented tolerances; unresolved ambiguous choices are visible and block production profile.
