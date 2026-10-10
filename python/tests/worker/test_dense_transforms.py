"""Independent covariance checks for the admitted worker's diagnostic maps."""

import math

import numpy as np
import pytest

from aloeforge_worker.dense import coordinates, layer_maps

SEED = "23" * 32
CONFIG = {
    "logical_hidden_size": 8,
    "physical_hidden_size": 12,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 2,
    "intermediate_size": 9,
}


def rotate(x, position):
    # Scalar paired rotation; never calls worker or reference runtime rotary.
    result = np.array(x, copy=True)
    for start in range(0, len(x), 2):
        a, b = float(x[start]), float(x[start + 1])
        result[start] = a * math.cos(position) - b * math.sin(position)
        result[start + 1] = a * math.sin(position) + b * math.cos(position)
    return result


@pytest.mark.parametrize("dtype,tolerance", [(np.float64, 2e-11), (np.float32, 3e-4)])
def test_residual_norm_bias_qk_vo_and_ffn_covariance(dtype, tolerance):
    c = CONFIG
    indices, factors = coordinates(c, SEED)
    p = np.zeros((8, 12), dtype=dtype)
    p[indices, np.arange(12)] = factors
    np.testing.assert_allclose(p @ p.T, np.eye(8), atol=tolerance, rtol=tolerance)
    # Rank-d projection, rather than identity, on the physical residual space.
    assert np.linalg.matrix_rank(p.T @ p) == 8
    assert not np.allclose(p.T @ p, np.eye(12))
    rng = np.random.default_rng(71)
    x = rng.normal(size=8).astype(dtype)
    y = x @ p
    gamma = rng.uniform(0.2, 1.8, 8).astype(dtype)
    eps = 0.013
    ratio = 8 / 12
    logical_norm = x / np.sqrt(np.mean(x * x) + eps)
    physical_norm = math.sqrt(ratio) * y / np.sqrt(np.mean(y * y) + ratio * eps)
    w = rng.normal(size=(8, 5)).astype(dtype)
    folded = p.T @ (gamma[:, None] * w)
    np.testing.assert_allclose(
        physical_norm @ folded,
        (logical_norm * gamma) @ w,
        atol=tolerance,
        rtol=tolerance,
    )
    maps = layer_maps(c, SEED, 0)
    q, k, v = [rng.normal(size=n).astype(dtype) for n in [8, 4, 4]]
    # Arbitrary affine projection outputs include nonzero source bias.
    q += rng.normal(size=8).astype(dtype)
    k += rng.normal(size=4).astype(dtype)
    v += rng.normal(size=4).astype(dtype)
    transformed_q = q[maps["q_rows"]] * maps["q_scale"]
    transformed_k = k[maps["kv_rows"]] * maps["k_scale"]
    transformed_v = v[maps["kv_rows"]] * maps["v_scale"]
    source_q, source_k = rotate(q, 1), rotate(k, 3)
    physical_q, physical_k = rotate(transformed_q, 1), rotate(transformed_k, 3)
    for physical_head in range(4):
        source_head = int(maps["q_rows"][2 * physical_head]) // 2
        source_kv = source_head // 2
        source_score = (
            source_q[2 * source_head : 2 * source_head + 2]
            @ source_k[2 * source_kv : 2 * source_kv + 2]
        )
        physical_score = (
            physical_q[2 * physical_head : 2 * physical_head + 2]
            @ physical_k[2 * (physical_head // 2) : 2 * (physical_head // 2) + 2]
        )
        assert abs(source_score - physical_score) <= tolerance * max(
            1, abs(source_score)
        )
    o = rng.normal(size=(8, 8)).astype(dtype)
    source_attention = np.repeat(v.reshape(2, 2), 2, axis=0).reshape(-1)
    physical_attention = np.repeat(transformed_v.reshape(2, 2), 2, axis=0).reshape(-1)
    transformed_o = (
        o[np.ix_(indices, maps["q_rows"])] * factors[:, None] * maps["o_scale"][None, :]
    )
    np.testing.assert_allclose(
        physical_attention @ transformed_o.T,
        (source_attention @ o.T) @ p,
        atol=tolerance,
        rtol=tolerance,
    )
    gate, up, down = [
        rng.normal(size=shape).astype(dtype) for shape in [(8, 9), (8, 9), (9, 8)]
    ]
    order, scales = maps["f_order"], maps["f_scale"]
    tg = p.T @ gate[:, order]
    tu = (p.T @ up[:, order]) * scales
    td = (down[order] @ p) / scales[:, None]
    source_gate = x @ gate
    physical_gate = y @ tg
    source = source_gate / (1 + np.exp(-source_gate)) * (x @ up)
    physical = physical_gate / (1 + np.exp(-physical_gate)) * (y @ tu)
    np.testing.assert_allclose(
        physical @ td, (source @ down) @ p, atol=tolerance, rtol=tolerance
    )
