"""Re-run exact diagnostic fixtures and print public scalar numerical evidence.

This is a test harness using the independent scalar baseline in tests, not a
runtime/compiler entry point. It prints no weights, inputs or secret matrices.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Only this repository's trusted, test-only scalar baseline/fixtures are loaded.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from python.tests.reference import test_v2_math as suite  # noqa: E402


def main():
    original_compare = suite.compare
    evidence = {}

    def measured(actual, expected, dtype):
        original_compare(actual, expected, dtype)
        name = np.dtype(dtype).name
        row = evidence.setdefault(name, {"comparisons": 0, "max_absolute_error": 0.0})
        error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
        row["comparisons"] += 1
        row["max_absolute_error"] = max(row["max_absolute_error"], error)

    suite.compare = measured
    for dtype in suite.DTYPES:
        for seed in (0, 11, 23):
            suite.test_key_family_cross_cancellation_and_residual(dtype, seed)
        for tied in (False, True):
            suite.test_embedding_head_exact_logits_and_explicit_noise(dtype, tied)
        suite.test_ffn_independent_baseline_and_nonlinear_gate_counterexample(dtype)
        suite.test_rmsnorm_expansion_epsilon_gamma_and_l2_counterexample(dtype)
        for layout in suite.LAYOUTS:
            suite.test_rope_block_literal_and_fixed_frequency_counterexamples(
                dtype, layout
            )
            for mechanism, heads, groups in suite.MECHANISMS:
                suite.test_attention_independent_baseline(
                    dtype, layout, mechanism, heads, groups
                )
                suite.test_full_toy_decoder_original_logits_and_greedy_ids(
                    dtype, layout, mechanism, heads, groups
                )
    print(
        json.dumps(
            {
                "profile": "diagnostic",
                "numpy": np.__version__,
                "fixture_seeds": [0, 11, 23, 16, 8, 42, 35, 7, 9, 17],
                "exact_comparisons": evidence,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
