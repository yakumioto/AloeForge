import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from aloeforge_runtime.evidence import (
    check_probabilities,
    log_probabilities,
    metric_delta,
    metric_totals,
    read_workload,
)


def test_workload_bounds_and_strict_ids(tmp_path):
    path = tmp_path / "workload.json"
    c = {"vocabulary_size": 11, "max_position_embeddings": 64}
    valid = {"schema_version": 1, "cases": [{"physical_ids": [1, 3], "max_tokens": 4}]}
    path.write_text(json.dumps(valid))
    assert read_workload(path, c) == valid["cases"]
    for case in [
        {"physical_ids": [True, 3], "max_tokens": 4},
        {"physical_ids": [11, 3], "max_tokens": 4},
        {"physical_ids": [1], "max_tokens": 4},
        {"physical_ids": [1, 3], "max_tokens": True},
        {"physical_ids": [1, 3], "max_tokens": 17},
        {"physical_ids": [1] * 64, "max_tokens": 4},
        {"physical_ids": [1, 3], "max_tokens": 4, "prompt": "FORBIDDEN"},
    ]:
        path.write_text(json.dumps(valid | {"cases": [case]}))
        with pytest.raises(ValueError):
            read_workload(path, c)
    for raw in [
        b'{"schema_version":1,"schema_version":1,"cases":[]}',
        b'{"schema_version":true,"cases":[]}',
        b"{}",
        b"x" * 65537,
        json.dumps(valid | {"cases": valid["cases"] * 9}).encode(),
    ]:
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            read_workload(path, c)


def test_probability_gate_is_complete_finite_and_fixed():
    expected = log_probabilities(np.array([1000.0, 1001.0, 999.0]))
    assert np.isclose(np.exp(expected).sum(), 1)
    good = {i: NS(logprob=p) for i, p in enumerate(expected)}
    assert check_probabilities(good, expected) == 0
    for bad in [
        {0: good[0]},
        good | {3: NS(logprob=0)},
        good | {0: NS(logprob=np.nan)},
        good | {0: NS(logprob=1)},
    ]:
        with pytest.raises(ValueError):
            check_probabilities(bad, expected)


def test_timing_evidence_requires_observed_engine_samples():
    before = metric_totals(
        [
            NS(name="vllm:time_to_first_token_seconds", sum=2.0, count=2),
            NS(name="vllm:time_per_output_token_seconds", sum=4.0, count=4),
        ]
    )
    after = {"ttft_seconds": (5.0, 4), "tpot_seconds": (5.0, 8)}
    assert metric_delta(before, after) == {"ttft_seconds": 1.5, "tpot_seconds": 0.25}
    with pytest.raises(ValueError, match="missing"):
        metric_totals([])
    for invalid in [
        before,
        after | {"ttft_seconds": (1.0, 4)},
        after | {"tpot_seconds": (np.nan, 8)},
    ]:
        with pytest.raises(ValueError):
            metric_delta(before, invalid)
