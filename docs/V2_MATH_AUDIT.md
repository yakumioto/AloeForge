# P4 V2 mathematical audit

PAPER source: [AloePri arXiv:2603.01499v2](https://arxiv.org/html/2603.01499v2),
30 March 2026; [PDF](https://arxiv.org/pdf/2603.01499v2), 28 pages. Page numbers
below are one-based PDF pages. Equations are mathematical facts transcribed for
checking; interpretations and counterexamples are AloeForge's own analysis.

Review status: source/shape/algebra self-audit performed for P4; executable
independent-baseline tests accompany this PR. Human method review and all
model/privacy gates remain pending. This completes a reviewable diagnostic audit,
**not production admission**. `production` and unqualified `paper-reference`
profiles must fail closed. No omitted gamma/window/MLA rule is attributed to the
paper. The independent oracle never imports the production worker.

## Coordinates and cancellation

ENGINEERING INTERPRETATION — all hidden vectors are rows. Logical hidden dimension
is d; physical dimension D=d+2h. Source embedding is V×d, head d×V, input projections
d×head_dim, output projections head_dim×d. Original token i maps to physical p[i];
Pi[p[i],i]=1. Source tied heads are E.T, not a promise that transformed weights
remain tied. Residual branches must have the same output P, or require an explicit
bridge y Q_old P_new; adding different physical coordinate systems is invalid.

For P=[B C E] Z and Q=Z.T vertical_stack(B^-1,F,N), CF=0, EN=0 and Z Z.T=I give
P Q=B B^-1+CF+EN=I. Q P is a rank-d projection on R^D, not I_D when h>0.
Families sharing B,E,F,Z can vary C,N and still satisfy P_i Q_j=I for every pair.
This cross-cancellation is needed when multiple projections see the same input.
Independent matrix entries/random keys cannot be substituted without validation.

## Ambiguity register and falsifiable gates

| ID / source | Printed formula or missing specification | ENGINEERING INTERPRETATION / DIAGNOSTIC choice | Falsifiable test |
|---|---|---|---|
| KEY-01, §5.2.1, Alg.1, p.8 | P=[B C E]Z; C d×h has columns from null(F.T), N h×d has rows from null(E); Q=Z.T[B^-1 F N].T | Literal null-vector directions do not fit the printed shapes. Use rows(C) in null(F.T), columns(N) in null(E), and a vertical stack with **untransposed** B^-1. Do not claim this repairs the publication. | Check CF=0, EN=0, every P_i Q_j=I; nonsymmetric B counterexample rejects transposed B^-1. |
| KEY-02, same | E1 d×h/2, E2 h/2×h, F1 h×h/2, F2 h/2×d; random invertible B/U | Admit positive even h and explicitly bounded well-conditioned toy keys; QR-sign-corrected Gaussian orthogonal sampling, SVD null spaces, PCG64 fixture RNG. Reject singular/ill-conditioned keys and bad identities. Sampling distributions for C/N are project choices. | Shape/rank, singular/near-singular, mismatched family, dtype/nonfinite and condition-bound negatives. |
| EMB-01, §5.2.2, pp.8–9 | E'=Pi(E+alpha_e N_e)P; H'=Q(H+alpha_h N_h)Pi.T | Separate diagnostic noise-off and explicitly supplied additive-noise mode. No noise/security defaults inferred. Restore logits before comparison. Tied source head need not stay tied after expansion. | Exact restored logits in noise-off; explicit-noise result equals independent noisy baseline and differs from clean baseline. |
| QK-01, §5.2.3, Alg.2, p.9 | Q'=Qin Wq R H Z; K'=Qin Wk R H^-1 Z.T | For non-involutive Z, raw dot products contain Z Z instead of Z Z.T. Keep literal formula as a counterexample; diagnostic coherent coordinate relocation uses **the same Z** for Q and K. Identity Z is the default exact ablation. | Three-cycle at position zero breaks literal Q/K score equality; corrected same-Z restores it. |
| ROPE-01, same / Eq.2 §3.4.1 p.4 | 2×2 R blocks and H=diag(s_i I2), but no checkpoint pair layout/partial-RoPE variant contract | Require explicit interleaved or split-half layout and supplied frequency vector; full even head dimension only. Shared R/H per KV group, nonzero bounded scales. Partial/scaled/unknown RoPE fails closed. | Both layouts and unequal positions preserve scores with identity blocks; arbitrary coordinate mixing and wrong layout fail equality. |
| BLOCK-01, Alg.2 BlockPerm, p.9 | t starts at 1; while t<m_blocks; c=min(beta,m_blocks-t); no t update; gamma supplied but unused; zeta_i=zeta^(-2(i-1)/m_blocks) | Literal loop cannot terminate in general and omits the final-block rule. Do not invent gamma or a stochastic window law. Accept an **explicit full partition** of bounded windows and within-window permutation, including terminal singleton. Frequencies are supplied, not guessed from zeta. | Full bijection/partition and window-bound tests; missing tail, cross-window movement and unknown sampler rejected. |
| BLOCK-02, §5.2.3 and §5.4 pp.9–11 | Block shuffling is approximate with fixed RoPE frequencies | Explicitly separate relocating frequencies (exact diagnostic coordinate change) from keeping frequencies fixed (approximate experiment). No accuracy threshold for real models is supplied. | Nontrivial block permutation at unequal positions has measurable fixed-frequency drift; relocating corresponding frequencies restores equality. |
| GROUP-01, §5.2.3 pp.9–10 | tau_kv in S_mkv and tau_group in S_(m/mkv), MLA described without full equations | Preserve contiguous GQA membership: q physical=kv_map[g]*group+within_map[j], K/V move by kv_map, O rows by matching q map. Same within-group map in each group. Support MHA/GQA/MQA only; reject MLA/MoE and incomplete schemas. | All three mechanisms preserve independent attention output; moving shared KV alone breaks it. |
| VO-01, Alg.2 p.9 | V'=Qin Wv U; O'=U^-1 Wo Pout | One invertible U per shared KV group; all queries using that V must cancel with its inverse. Reject singular/unbounded condition numbers; explicit source/output dimensions. | Values/output and full causal attention agree with independent baseline in FP64/FP32. |
| FFN-01, §5.2.4 p.10 / Eq.1 p.4 | G'=Qin G Z; U'=Qin U H Z; D'=Z^-1 H^-1 D Pout | H is an explicitly supplied invertible diagonal, not arbitrary dense scaling. Do not scale the nonlinear gate. Bias/fused layouts/MoE are not admitted. | SiLU baseline equality with positive/negative scales; scaling the gate or wrong inverse order fails. |
| NORM-01, §5.2.5 p.10 / §5.4 p.11 | kappa=E[||xP||/||x||]; kappa*xP/sqrt(sum(y²)/D) approximately xP/sqrt(sum(x²)/d) | Raw-L2 kappa misses sqrt(d/D): even P P.T=a²I yields an extra sqrt(D/d). Exact diagnostic RMS coefficient is a sqrt(d/D); general anisotropic P has no universal exact scalar. Calibration convention (L2/RMS) must be explicit. | Scaled isometry with D>d falsifies raw-L2 kappa; anisotropic directions falsify a universal scalar; calibrated approximation reports drift. |
| NORM-02, same; §3.4.1 p.4 | RMS definition omits epsilon and parameter-folding details | Supply logical/physical epsilon explicitly. Under a scaled isometry, eps_physical=a²(d/D) eps_logical. Fold gamma with Q diag(gamma) P or into the adjacent projection; never attribute this epsilon/folding choice to the paper. | Exact non-unit gamma/isometry/epsilon tests, zero-row handling; unchanged epsilon or omitted fold visibly differs. |
| PREC-01, §5.2.1 p.8 | Half-precision magnitude regulation via lambda, no BF16 backend tolerance contract | NumPy FP64/FP32 oracle and software BF16 round-to-nearest-even conversion only. Conversion/carrier are not BF16 GEMM or GPU evidence. Fix toy tolerances before results; report rounding drift separately from method approximation. | Scalar struct/bit goldens, halfway ties, signs/subnormals/overflow; BF16 toy comparison with declared bounded rounding tolerance. |

## Exact diagnostic proofs

With compatible input P/Q, transformed linear weights Q W P_out satisfy
(x P)(Q W P_out)=(x W)P_out. Embedding/head cancellation additionally places
columns/rows with the inverse token mapping; physical logits at p[i] equal source
logits at i. Residual addition commutes with the **same** right map P.

For Q/K, let A=R H Z, K-map=R H^-1 Z, with block rotations R and paired scalars H.
Without RoPE, A (K-map).T=I. At unequal positions, paired rotations/scalars commute
with their corresponding rotary blocks. A full pair relocation is exact only if
its frequency coordinates move too (or frequencies are equal). Fixed frequencies
are deliberately approximate. GQA transformations must agree across each shared
KV group; summing the correspondingly permuted heads then preserves output.

For FFN, SiLU(a Z)=SiLU(a) Z for a permutation Z. The up-path diagonal scales move
through the Hadamard product and cancel in Z.T H^-1 D. Scaling the **gate** does
not commute with SiLU, so the gate receives permutation only.

For normalization, if P P.T=a²I, RMS_D(xP)=a sqrt(d/D) RMS_d(x). With c=a sqrt(d/D)
and eps_D=c² eps_d, c(xP)/sqrt(mean_D((xP)²)+eps_D) equals
(x/sqrt(mean_d(x²)+eps_d))P. Gamma is separately folded through Q diag(gamma) P.
This is an exact **diagnostic scaled-isometry proof**, not a proof for Algorithm 1's
general non-isometric keys or a privacy result.

## Fixed numerical gates and exclusions

FP64 component/block comparisons: atol=rtol=2e-11; FP32: atol=rtol=3e-4.
P/Q and null/orthogonality checks: 2e-12 FP64, 3e-5 FP32. Toy BF16-converted-input
comparison: atol=rtol=2e-2, explicitly only a rounding-drift check. Condition number
bound 32 and scale magnitude [1/8,8] keep fixtures diagnostic and bounded. No
acceptance threshold is fitted to a failing result. Seeds/shapes and worst observed
errors are recorded by the validation command. Only finite FP64/FP32 arrays, even
full RoPE dimensions, explicit epsilon/frequency/layout and dense bias-free blocks
are admitted. Original/source tensors never become remote runtime artifacts here.

## Oracle API and bounds

Install the pinned `reference` extra (or `dev`, for tests). Entry points are
`keymat.KeyFamily(...).member(seed=...)`, `keymat.isometry(...)`, `v2.Oracle(...)`,
`embedding_head`, `ffn_weights`, `attention_weights`, and the pure `attention`,
`ffn`, `rmsnorm`, `transformed_rmsnorm`, `norm_coefficient`, `rope`, `block_order`
and `bf16_roundtrip` functions. For example:

```python
import numpy as np
from aloeforge_reference.keymat import KeyFamily

family = KeyFamily(6, 2, 0.02, seed=8, dtype=np.float64)
first, second = family.member(seed=9), family.member(seed=10)
first.require_inverse(second)  # Explicit cross-key cancellation check.
```

Arrays must be finite native NumPy FP32/FP64, with each dimension 1–256 and at
most 65,536 elements, checked before transformed attention allocation. Key-family
logical d is 1–64, h is even 2–64; the isometry uses those same bounds. Attention
supports <=8 query heads, contiguous divisible KV groups, full even head dimensions
2–64 and <=64 sequence positions. Positions are explicit strictly increasing
integers in [0,1,000,000]; frequencies are positive supplied floating arrays.
FFN scaling and group R/H/U parameters are explicit; the source matrices are
unbiased/dense, with no model metadata or architecture admission implied.

`Oracle(profile="diagnostic", noise_mode="off")` refuses supplied noises;
`noise_mode="explicit"` requires a finite matching (embedding_delta, head_delta)
tuple. This tuple already includes any desired amplitude; no sigma/alpha population
rule or security guarantee is inferred. BF16 conversion first casts to FP32, then
rounds its bits to nearest-even BF16 and returns FP32 carrier values; it does not
promise direct FP64-to-BF16 single-rounding behavior. Overflow/nonfinite results
fail closed. Toy arrays/PCG64 seeds are not private production keys.

The measurement script is a test harness that reuses only trusted test fixtures
and the independent scalar baseline. Neither oracle code nor a production worker
imports that baseline/harness; the first-party worker remains unimplemented.
