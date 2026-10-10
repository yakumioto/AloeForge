"""Bounded public workload and scalar-only parity/metric gates."""

import math

import numpy as np

from aloeforge_reference.dense_runtime import _json


def read_workload(path, contract):
    with open(path, "rb") as file:
        raw = file.read((64 << 10) + 1)
    if len(raw) > 64 << 10:
        raise ValueError("workload exceeds bounds")
    data = _json(raw)
    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "cases"}
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
        or not isinstance(data["cases"], list)
        or not 1 <= len(data["cases"]) <= 8
    ):
        raise ValueError("unsupported workload schema")
    for case in data["cases"]:
        if (
            not isinstance(case, dict)
            or set(case) != {"physical_ids", "max_tokens"}
            or type(case["max_tokens"]) is not int
            or not 2 <= case["max_tokens"] <= 16
        ):
            raise ValueError("invalid workload case")
        ids = case["physical_ids"]
        if (
            not isinstance(ids, list)
            or not 2 <= len(ids) <= 64
            or any(
                type(i) is not int or not 0 <= i < contract["vocabulary_size"]
                for i in ids
            )
            or len(ids) + case["max_tokens"] > contract["max_position_embeddings"]
        ):
            raise ValueError("invalid physical workload")
    return data["cases"]


def log_probabilities(logits):
    logits = np.asarray(logits, dtype=np.float64)
    peak = logits.max(axis=-1, keepdims=True)
    return logits - peak - np.log(np.exp(logits - peak).sum(-1, keepdims=True))


def check_probabilities(record, expected):
    if not isinstance(record, dict) or set(record) != set(range(len(expected))):
        raise ValueError("complete physical log probabilities required")
    measured = np.asarray([record[i].logprob for i in range(len(expected))])
    if not np.isfinite(measured).all() or not np.allclose(
        measured, expected, atol=3e-4, rtol=3e-4
    ):
        raise ValueError("fixed probability parity gate failed")
    return float(np.max(np.abs(measured - expected)))


METRICS = {
    "ttft_seconds": "vllm:time_to_first_token_seconds",
    "tpot_seconds": "vllm:time_per_output_token_seconds",
}


def metric_totals(metrics):
    values = {}
    for label, name in METRICS.items():
        selected = [m for m in metrics if m.name == name]
        if not selected:
            raise ValueError("required engine timing metric missing")
        values[label] = (sum(m.sum for m in selected), sum(m.count for m in selected))
    return values


def metric_delta(before, after):
    means = {}
    for name in METRICS:
        total = after[name][0] - before[name][0]
        count = after[name][1] - before[name][1]
        if count <= 0 or not math.isfinite(total) or total <= 0:
            raise ValueError("engine timing evidence incomplete")
        means[name] = total / count
    return means
