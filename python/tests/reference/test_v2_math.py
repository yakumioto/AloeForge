"""Fixed-tolerance independent positive/negative P4 mathematical gates."""

import math
import struct
import subprocess
import sys

import numpy as np
import pytest

from aloeforge_reference.keymat import (
    KeyFamily,
    KeyPair,
    generator,
    inverse,
    isometry,
)
from aloeforge_reference.v2 import (
    AttentionWeights,
    Oracle,
    UnresolvedMethod,
    attention,
    bf16_roundtrip,
    block_order,
    ffn,
    norm_coefficient,
    paper_block_sampler,
    permutation_matrix,
    rmsnorm,
    rope,
    transformed_rmsnorm,
)

from . import math_baseline as baseline

DTYPES = [np.float64, np.float32]
LAYOUTS = ["interleaved", "split-half"]
MECHANISMS = [("MHA", 3, 3), ("GQA", 4, 2), ("MQA", 4, 1)]


def compare(actual, expected, dtype):
    tolerance = 2e-11 if dtype == np.float64 else 3e-4
    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=tolerance)


def data(seed=42, dtype=np.float64, *, heads=4, groups=2, d=6, width=6):
    rng = generator(seed)

    def draw(shape):
        return (rng.standard_normal(shape) * 0.2).astype(dtype)

    rows = draw((4, d))
    weights = AttentionWeights(
        draw((heads, d, width)),
        draw((groups, d, width)),
        draw((groups, d, width)),
        draw((heads, width, d)),
    )
    positions = np.array([0, 1, 4, 9])
    frequencies = np.tile(np.array([1.0, 0.13, 0.017], dtype=dtype), (groups, 1))
    for group in range(groups):
        frequencies[group] *= dtype(1 + 0.15 * group)
    return rng, rows, weights, positions, frequencies


def transform_spec(dtype, groups, *, nontrivial=False):
    spec = dict(
        angles=np.tile(np.array([0.2, -0.7, 0.35], dtype=dtype), (groups, 1)),
        scale=np.tile(np.array([1.25, -0.75, 2.0], dtype=dtype), (groups, 1)),
        value_maps=np.tile(
            np.array(
                [
                    [1.1, 0.2, 0, 0, 0, 0],
                    [0, 0.8, 0.1, 0, 0, 0],
                    [0, 0, 1.0, 0.1, 0, 0],
                    [0, 0, 0, 1.2, 0.1, 0],
                    [0, 0, 0, 0, 0.9, 0.1],
                    [0.1, 0, 0, 0, 0, 1.0],
                ],
                dtype=dtype,
            ),
            (groups, 1, 1),
        ),
        block_orders=[[1, 2, 0] if nontrivial else [0, 1, 2] for _ in range(groups)],
        windows=[[3] for _ in range(groups)],
        maximum_window=3,
    )
    for group in range(groups):
        spec["angles"][group] += dtype(0.17 * group)
        spec["scale"][group] *= dtype(1 + 0.1 * group)
        spec["value_maps"][group] += dtype(0.05 * group) * np.eye(6, dtype=dtype)
    return spec


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", [0, 11, 23])
def test_key_family_cross_cancellation_and_residual(dtype, seed):
    family = KeyFamily(6, 4, 0.03, seed=seed, dtype=dtype)
    keys = [family.member(seed=seed + 101 + i) for i in range(3)]
    for key in keys:
        assert key.p.shape == (6, 14) and key.q.shape == (14, 6)
        compare(key.p @ key.q, np.eye(6), dtype)
        assert not np.allclose(key.q @ key.p, np.eye(14))
        for other in keys:
            key.require_inverse(other)
            compare(key.p @ other.q, np.eye(6), dtype)
    x = generator(3).standard_normal((3, 6)).astype(dtype)
    y = generator(5).standard_normal((3, 6)).astype(dtype)
    compare(
        keys[0].encode(x) + keys[0].encode(y),
        np.array(baseline.add(x.tolist(), y.tolist())) @ keys[0].p,
        dtype,
    )
    compare(keys[0].restore(keys[0].encode(x)), x, dtype)
    # Mixing output coordinate systems does not preserve this residual.
    assert (
        np.max(np.abs(keys[0].encode(x) + keys[1].encode(y) - keys[0].encode(x + y)))
        > 0.01
    )


def test_literal_transposed_inverse_is_not_a_right_inverse():
    b = np.array([[1.0, 2.0], [0.0, 1.0]])
    p = np.concatenate((b, np.zeros((2, 4))), axis=1)
    good = np.concatenate((np.linalg.inv(b), np.zeros((4, 2))), axis=0)
    KeyPair(p, good)
    literal = np.concatenate((np.linalg.inv(b).T, np.zeros((4, 2))), axis=0)
    with pytest.raises(ValueError, match="cancellation"):
        KeyPair(p, literal)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("tied", [False, True])
def test_embedding_head_exact_logits_and_explicit_noise(dtype, tied):
    rng = generator(16)
    e = (rng.standard_normal((7, 6)) * 0.2).astype(dtype)
    h = e.T.copy() if tied else (rng.standard_normal((6, 7)) * 0.2).astype(dtype)
    key = KeyFamily(6, 2, 0.02, seed=1, dtype=dtype).member(seed=2)
    order = [3, 0, 6, 1, 5, 2, 4]
    oracle = Oracle(dtype=dtype)
    physical_e, physical_h = oracle.embedding_head(e, h, key, order)
    expected = baseline.matmul(e.tolist(), h.tolist())
    restored = (physical_e[order] @ physical_h)[:, order]
    compare(restored, expected, dtype)
    if tied:
        assert not np.allclose(physical_h, physical_e.T)
    noises = (
        (rng.standard_normal(e.shape) * 0.03).astype(dtype),
        (rng.standard_normal(h.shape) * 0.02).astype(dtype),
    )
    noisy_e, noisy_h = Oracle(dtype=dtype, noise_mode="explicit").embedding_head(
        e, h, key, order, noises=noises
    )
    noisy_result = (noisy_e[order] @ noisy_h)[:, order]
    compare(
        noisy_result,
        baseline.matmul((e + noises[0]).tolist(), (h + noises[1]).tolist()),
        dtype,
    )
    assert np.max(np.abs(noisy_result - expected)) > 0.001
    with pytest.raises(ValueError, match="noise-off"):
        oracle.embedding_head(e, h, key, order, noises=noises)
    with pytest.raises(ValueError, match="requires"):
        Oracle(dtype=dtype, noise_mode="explicit").embedding_head(e, h, key, order)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("mechanism,heads,groups", MECHANISMS)
def test_attention_independent_baseline(dtype, layout, mechanism, heads, groups):
    _, rows, weights, positions, frequencies = data(
        dtype=dtype, heads=heads, groups=groups
    )
    family = KeyFamily(6, 2, 0.01, seed=8, dtype=dtype)
    key_in, key_out = family.member(seed=9), family.member(seed=10)
    oracle = Oracle(dtype=dtype)
    transformed, moved = oracle.attention_weights(
        weights,
        key_in,
        key_out,
        **transform_spec(dtype, groups),
        kv_order=list(reversed(range(groups))),
        within_order=list(reversed(range(heads // groups))),
        frequencies=frequencies,
        layout=layout,
        mechanism=mechanism,
    )
    expected, scores = baseline.attend(
        rows.tolist(),
        weights.q.tolist(),
        weights.k.tolist(),
        weights.v.tolist(),
        weights.o.tolist(),
        positions.tolist(),
        frequencies.tolist(),
        layout,
    )
    output, _ = attention(
        key_in.encode(rows), transformed, positions, moved, layout=layout
    )
    compare(key_out.restore(output), expected, dtype)
    plain_output, plain_scores = attention(
        rows, weights, positions, frequencies, layout=layout
    )
    compare(plain_output, expected, dtype)
    compare(plain_scores, scores, dtype)
    # Causal decode prefixes agree with prefill, including nonconsecutive positions.
    for end in range(1, len(rows) + 1):
        prefix, _ = attention(
            key_in.encode(rows[:end]),
            transformed,
            positions[:end],
            moved,
            layout=layout,
        )
        compare(prefix[-1], output[end - 1], dtype)
    if groups > 1:
        wrong = AttentionWeights(weights.q, weights.k[::-1], weights.v[::-1], weights.o)
        bad_output, _ = attention(rows, wrong, positions, frequencies, layout=layout)
        assert np.max(np.abs(bad_output - expected)) > 0.001


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_rope_block_literal_and_fixed_frequency_counterexamples(dtype, layout):
    _, rows, weights, positions, frequencies = data(dtype=dtype, heads=2, groups=1)
    key = isometry(6, 2, seed=5, dtype=dtype)
    # A deliberately strong counterexample, keeping the declared drift gate.
    rows = rows * 4
    spec = transform_spec(dtype, 1, nontrivial=True)
    oracle = Oracle(dtype=dtype)
    kwargs = dict(
        kv_order=[0],
        within_order=[1, 0],
        frequencies=frequencies,
        layout=layout,
        mechanism="MQA",
    )
    expected, _ = attention(rows, weights, positions, frequencies, layout=layout)
    exact, moved = oracle.attention_weights(
        weights, key, key, **spec, **kwargs, block_mode="relocate-frequencies"
    )
    output, _ = attention(key.encode(rows), exact, positions, moved, layout=layout)
    compare(key.restore(output), expected, dtype)
    fixed, old_freq = oracle.attention_weights(
        weights, key, key, **spec, **kwargs, block_mode="fixed-frequencies"
    )
    approximate, _ = attention(
        key.encode(rows), fixed, positions, old_freq, layout=layout
    )
    assert np.max(np.abs(key.restore(approximate) - expected)) > 0.001
    literal, _ = oracle.attention_weights(
        weights,
        key,
        key,
        **spec,
        **kwargs,
        block_mode="relocate-frequencies",
        variant="paper-literal",
    )
    x = key.encode(rows[:1])
    good_score = (x @ exact.q[0]) @ (x @ exact.k[0]).T
    bad_score = (x @ literal.q[0]) @ (x @ literal.k[0]).T
    assert np.max(np.abs(good_score - bad_score)) > 0.001
    with pytest.raises(ValueError, match="identity"):
        oracle.attention_weights(weights, key, key, **spec, **kwargs)


@pytest.mark.parametrize("dtype", DTYPES)
def test_ffn_independent_baseline_and_nonlinear_gate_counterexample(dtype):
    rng = generator(35)
    x = (rng.standard_normal((4, 6)) * 0.3).astype(dtype)
    gate, up = [(rng.standard_normal((6, 8)) * 0.2).astype(dtype) for _ in range(2)]
    down = (rng.standard_normal((8, 6)) * 0.2).astype(dtype)
    family = KeyFamily(6, 2, 0.02, seed=2, dtype=dtype)
    ki, ko = family.member(seed=3), family.member(seed=4)
    order = [2, 5, 0, 7, 4, 1, 6, 3]
    scale = np.array([2, -1, 0.5, 1.25, -0.75, 1.5, 0.8, -2], dtype=dtype)
    g, u, d = Oracle(dtype=dtype).ffn_weights(gate, up, down, ki, ko, order, scale)
    expected = baseline.feed_forward(
        x.tolist(), gate.tolist(), up.tolist(), down.tolist()
    )
    compare(ko.restore(ffn(ki.encode(x), g, u, d)), expected, dtype)
    z = permutation_matrix(order, dtype)
    wrong_gate = (ki.q @ gate * scale) @ z
    assert (
        np.max(np.abs(ko.restore(ffn(ki.encode(x), wrong_gate, u, d)) - expected))
        > 0.001
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm_expansion_epsilon_gamma_and_l2_counterexample(dtype):
    x = np.array(
        [[0.2, -0.5, 0.9, 0.4, -0.7, 1.1], [0.01, -0.03, 0.02, 0.04, 0.01, -0.01]],
        dtype=dtype,
    )
    gamma = np.array([1.2, 0.8, -0.7, 1.5, 0.9, 1.1], dtype=dtype)
    key = isometry(6, 4, scale=1.4, seed=7, dtype=dtype)
    epsilon = 0.01
    c = 1.4 * math.sqrt(6 / 14)
    expected = baseline.rms(x.tolist(), gamma.tolist(), epsilon)
    result = transformed_rmsnorm(
        key.encode(x), key, gamma, kappa=c, epsilon=c * c * epsilon
    )
    compare(key.restore(result), expected, dtype)
    raw_kappa = norm_coefficient(x, key, convention="l2")
    wrong = transformed_rmsnorm(
        key.encode(x), key, gamma, kappa=raw_kappa, epsilon=c * c * epsilon
    )
    assert np.max(np.abs(key.restore(wrong) - expected)) > 0.1
    wrong_epsilon = transformed_rmsnorm(
        key.encode(x), key, gamma, kappa=c, epsilon=epsilon
    )
    assert np.max(np.abs(key.restore(wrong_epsilon) - expected)) > 0.01
    compare(norm_coefficient(x, key, convention="rms"), c, dtype)
    compare(rmsnorm(x, gamma, epsilon=epsilon), expected, dtype)


def test_anisotropic_norm_has_no_universal_exact_kappa():
    p = np.array([[2.0, 0, 0, 0, 0, 0], [0, 0.5, 0, 0, 0, 0]])
    q = np.array([[0.5, 0], [0, 2.0], [0, 0], [0, 0], [0, 0], [0, 0]])
    key = KeyPair(p, q)
    x = np.eye(2)
    coefficient = norm_coefficient(x, key, convention="rms")
    expected = baseline.rms(x.tolist(), [1.0, 1.0], 0)
    observed = key.restore(
        transformed_rmsnorm(
            key.encode(x), key, np.ones(2), kappa=coefficient, epsilon=0
        )
    )
    assert np.max(np.abs(observed - expected)) > 0.5


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("mechanism,heads,groups", MECHANISMS)
def test_full_toy_decoder_original_logits_and_greedy_ids(
    dtype, layout, mechanism, heads, groups
):
    rng, _, weights, positions, frequencies = data(
        seed=9, dtype=dtype, heads=heads, groups=groups
    )
    e = (rng.standard_normal((7, 6)) * 0.2).astype(dtype)
    h = (rng.standard_normal((6, 7)) * 0.2).astype(dtype)
    gate, up = [(rng.standard_normal((6, 8)) * 0.2).astype(dtype) for _ in range(2)]
    down = (rng.standard_normal((8, 6)) * 0.2).astype(dtype)
    gammas = [(rng.uniform(0.7, 1.3, 6)).astype(dtype) for _ in range(3)]
    tokens = [0, 3, 6, 2]
    order = [2, 5, 1, 6, 0, 4, 3]
    key = isometry(6, 2, scale=1.2, seed=17, dtype=dtype)
    oracle = Oracle(dtype=dtype)
    pe, ph = oracle.embedding_head(e, h, key, order)
    aw, moved = oracle.attention_weights(
        weights,
        key,
        key,
        **transform_spec(dtype, groups),
        kv_order=list(reversed(range(groups))),
        within_order=list(reversed(range(heads // groups))),
        frequencies=frequencies,
        layout=layout,
        mechanism=mechanism,
    )
    g, u, d = oracle.ffn_weights(
        gate,
        up,
        down,
        key,
        key,
        [3, 6, 0, 7, 2, 1, 5, 4],
        np.array([1, 2, -0.5, 0.75, -1, 1.5, -0.8, 1.2], dtype=dtype),
    )
    c = 1.2 * math.sqrt(6 / 10)
    eps = 0.001
    rows = pe[[order[t] for t in tokens]]
    normalized = transformed_rmsnorm(rows, key, gammas[0], kappa=c, epsilon=c * c * eps)
    attended, _ = attention(normalized, aw, positions, moved, layout=layout)
    rows += attended
    normalized = transformed_rmsnorm(rows, key, gammas[1], kappa=c, epsilon=c * c * eps)
    rows += ffn(normalized, g, u, d)
    rows = transformed_rmsnorm(rows, key, gammas[2], kappa=c, epsilon=c * c * eps)
    restored_logits = (rows @ ph)[:, order]
    expected = baseline.decoder(
        tokens,
        e.tolist(),
        h.tolist(),
        weights.q.tolist(),
        weights.k.tolist(),
        weights.v.tolist(),
        weights.o.tolist(),
        gate.tolist(),
        up.tolist(),
        down.tolist(),
        [g.tolist() for g in gammas],
        eps,
        positions.tolist(),
        frequencies.tolist(),
        layout,
    )
    compare(restored_logits, expected, dtype)
    assert (
        np.argmax(restored_logits, axis=1).tolist()
        == np.argmax(expected, axis=1).tolist()
    )


def scalar_bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    upper, lower = bits >> 16, bits & 0xFFFF
    if lower > 0x8000 or lower == 0x8000 and upper % 2:
        upper += 1
    return struct.unpack("<f", struct.pack("<I", (upper & 0xFFFF) << 16))[0]


def test_bf16_ties_subnormals_signs_and_bounded_toy_drift():
    values = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -2.0,
            1.00390625,
            1.01171875,
            -1.00390625,
            2**-133,
            2**-134,
            0.123456,
        ],
        dtype=np.float64,
    )
    result = bf16_roundtrip(values)
    expected = np.array([scalar_bf16(float(v)) for v in values], dtype=np.float32)
    np.testing.assert_array_equal(result.view(np.uint32), expected.view(np.uint32))
    assert result[4] == 1 and result[5] == 1.015625
    assert np.signbit(result[1]) and result[7] == 2**-133 and result[8] == 0
    _, x, weights, positions, frequencies = data(dtype=np.float32)
    quantized = AttentionWeights(
        *(bf16_roundtrip(w) for w in (weights.q, weights.k, weights.v, weights.o))
    )
    output, _ = attention(
        bf16_roundtrip(x), quantized, positions, frequencies, layout="interleaved"
    )
    expected, _ = baseline.attend(
        x.tolist(),
        weights.q.tolist(),
        weights.k.tolist(),
        weights.v.tolist(),
        weights.o.tolist(),
        positions.tolist(),
        frequencies.tolist(),
        "interleaved",
    )
    np.testing.assert_allclose(output, expected, atol=2e-2, rtol=2e-2)
    assert np.max(np.abs(output - expected)) > 1e-7


def test_oracle_import_is_independent_and_blocks_production():
    for profile in ("production", "paper-reference", "unknown"):
        with pytest.raises(UnresolvedMethod):
            Oracle(profile=profile)
    with pytest.raises(UnresolvedMethod, match="gamma"):
        paper_block_sampler(beta=3, gamma=0.5)
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import aloeforge_reference.v2, sys; "
            "assert 'aloeforge_worker' not in sys.modules; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_explicit_windows_tail_and_layout_negatives(dtype):
    assert block_order([1, 0, 2], [2, 1], 2).tolist() == [1, 0, 2]
    for order, windows, maximum in [
        ([1, 2, 0], [2, 1], 2),
        ([1, 0, 2], [2], 2),
        ([1, 0, 2], [3], 2),
        ([1, 1, 2], [3], 3),
        ([True, 0, 2], [3], 3),
        ([0, 1, 2], [0, 3], 3),
    ]:
        with pytest.raises(ValueError):
            block_order(order, windows, maximum)
    rows = np.array(
        [[0.2, 0.5, -0.8, 0.3, 0.7, -0.6], [0.7, 0.1, 0.4, -0.5, 0.9, 0.2]], dtype=dtype
    )
    position = np.array([0, 5])
    freq = np.array([1.0, 0.1, 0.01], dtype=dtype)
    interleaved = rope(rows, position, freq, layout="interleaved")
    split = rope(rows, position, freq, layout="split-half")
    assert np.max(np.abs(interleaved - split)) > 0.1
    with pytest.raises(ValueError, match="layout"):
        rope(rows, position, freq, layout="unknown")
    for bad_position in (
        np.array([1, 1]),
        np.array([-1, 3]),
        np.array([0.0, 1.0]),
        np.array([0, 1000001]),
    ):
        with pytest.raises(ValueError):
            rope(rows, bad_position, freq, layout="interleaved")
    for bad_freq in (np.array([0.0, 0.1, 0.01], dtype=dtype), np.ones(2, dtype=dtype)):
        with pytest.raises(ValueError):
            rope(rows, position, bad_freq, layout="interleaved")


@pytest.mark.parametrize("dtype", [np.float16, np.int64, np.bool_, "object"])
def test_unknown_dtypes_fail_closed(dtype):
    with pytest.raises(ValueError):
        Oracle(dtype=dtype)
    with pytest.raises(ValueError):
        rmsnorm(np.ones((2, 2), dtype=dtype), np.ones(2), epsilon=0.01)


@pytest.mark.parametrize(
    "kind",
    [
        "odd-h",
        "zero-d",
        "bool-d",
        "huge-h",
        "negative-lambda",
        "huge-lambda",
        "nan-lambda",
        "bad-seed",
    ],
)
def test_key_parameters_fail_closed(kind):
    kwargs = dict(d=4, h=2, coefficient=0.03, seed=1)
    if kind == "odd-h":
        kwargs["h"] = 3
    if kind == "zero-d":
        kwargs["d"] = 0
    if kind == "bool-d":
        kwargs["d"] = True
    if kind == "huge-h":
        kwargs["h"] = 66
    if kind == "negative-lambda":
        kwargs["coefficient"] = -0.1
    if kind == "huge-lambda":
        kwargs["coefficient"] = 1
    if kind == "nan-lambda":
        kwargs["coefficient"] = np.nan
    if kind == "bad-seed":
        kwargs["seed"] = True
    with pytest.raises(ValueError):
        KeyFamily(**kwargs)


def test_singular_ill_conditioned_nonfinite_and_cross_family_keys():
    for matrix in (
        np.zeros((2, 2)),
        np.diag([1.0, 1e-9]),
        np.array([[np.nan, 0], [0, 1.0]]),
    ):
        with pytest.raises(ValueError):
            inverse(matrix)
    a = KeyFamily(6, 2, 0.01, seed=1).member(seed=2)
    b = KeyFamily(6, 2, 0.01, seed=3).member(seed=4)
    with pytest.raises(ValueError, match="cross-key"):
        a.require_inverse(b)
    for p, q in (
        (a.p, a.q.T),
        (a.p.astype(np.float32), a.q),
        (a.p, a.q * 2),
        (np.zeros((2, 7)), np.zeros((7, 2))),
    ):
        with pytest.raises(ValueError):
            KeyPair(p, q)
    source = a.p.copy()
    copy = KeyPair(source, a.q)
    source[0] = 0
    assert not np.array_equal(source, copy.p)
    with pytest.raises(ValueError):
        copy.p[0, 0] = 0
    with pytest.raises(ValueError):
        a.encode(np.ones((2, 5)))
    with pytest.raises(ValueError):
        a.restore(np.ones((2, 6)))


def test_attention_unsupported_features_shapes_and_mixed_dtype():
    _, rows, weights, positions, frequencies = data()
    key = isometry(6, 2)
    valid = dict(
        **transform_spec(np.float64, 2),
        kv_order=[1, 0],
        within_order=[1, 0],
        frequencies=frequencies,
        layout="interleaved",
        mechanism="GQA",
    )
    for name in ("MLA", "MoE", "unknown"):
        kwargs = dict(valid, mechanism=name)
        with pytest.raises(UnresolvedMethod):
            Oracle().attention_weights(weights, key, key, **kwargs)
    variants = [
        dict(valid, scale=np.zeros((2, 3))),
        dict(valid, value_maps=np.zeros((2, 6, 6))),
        dict(valid, frequencies=frequencies.astype(np.float32)),
        dict(valid, angles=np.ones((2, 2))),
        dict(valid, kv_order=[0, 0]),
        dict(valid, within_order=[0]),
        dict(valid, windows=[[2], [3]]),
        dict(valid, block_mode="guess-gamma"),
        dict(valid, variant="guess-direction"),
        dict(valid, mechanism="MQA"),
        dict(valid, mechanism="MHA"),
    ]
    for kwargs in variants:
        with pytest.raises(ValueError):
            Oracle().attention_weights(weights, key, key, **kwargs)
    bad = AttentionWeights(weights.q[:3], weights.k, weights.v, weights.o[:3])
    with pytest.raises(ValueError):
        bad.validate()
    with pytest.raises(ValueError):
        attention(
            rows.astype(np.float32),
            weights,
            positions,
            frequencies,
            layout="interleaved",
        )
    with pytest.raises(ValueError):
        attention(
            rows, weights, positions, frequencies, layout="interleaved", causal=False
        )


def test_norm_and_ffn_bad_shapes_scales_and_nonfinite_values():
    key = isometry(2, 2)
    x = np.ones((2, 2))
    for epsilon in (-1, np.nan):
        with pytest.raises(ValueError):
            rmsnorm(x, np.ones(2), epsilon=epsilon)
    with pytest.raises(ValueError):
        rmsnorm(np.zeros((1, 2)), np.ones(2), epsilon=0)
    assert np.array_equal(
        rmsnorm(np.zeros((1, 2)), np.ones(2), epsilon=0.001), np.zeros((1, 2))
    )
    with pytest.raises(ValueError):
        norm_coefficient(np.zeros((1, 2)), key, convention="rms")
    with pytest.raises(ValueError):
        norm_coefficient(x, key, convention="guess")
    for kappa in (0, -1, np.inf):
        with pytest.raises(ValueError):
            transformed_rmsnorm(
                key.encode(x), key, np.ones(2), kappa=kappa, epsilon=0.01
            )
    for values in (np.array([[np.nan, 0]]), np.ones((1, 257)), np.ones((0, 2))):
        with pytest.raises(ValueError):
            key.encode(values)
    gate, up, down = np.ones((2, 3)), np.ones((2, 3)), np.ones((3, 2))
    for scale in (np.array([1.0, 0, 1]), np.array([1.0, 9, 1]), np.ones(2)):
        with pytest.raises(ValueError):
            Oracle().ffn_weights(gate, up, down, key, key, [1, 2, 0], scale)
    with pytest.raises(ValueError):
        ffn(x, gate, up, down.T)
    with pytest.raises(ValueError):
        Oracle().embedding_head(np.ones((3, 2)), np.ones((3, 2)), key, [0, 1, 2])
    with pytest.raises(ValueError):
        Oracle(noise_mode="automatic")
    for values in (
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1e300]),
        np.array([np.finfo(np.float32).max], dtype=np.float32),
    ):
        with pytest.raises(ValueError):
            bf16_roundtrip(values)


@pytest.mark.parametrize("value", [True, np.bool_(False), "0.1", None])
def test_scalar_types_do_not_silently_coerce(value):
    with pytest.raises(ValueError):
        KeyFamily(2, 2, value, seed=1)
    with pytest.raises(ValueError):
        isometry(2, 2, scale=value)
    with pytest.raises(ValueError):
        rmsnorm(np.ones((1, 2)), np.ones(2), epsilon=value)
    with pytest.raises(ValueError):
        transformed_rmsnorm(
            np.ones((1, 6)), isometry(2, 2), np.ones(2), kappa=value, epsilon=0.01
        )


def test_oversized_permutation_fails_before_matrix_allocation():
    with pytest.raises(ValueError):
        permutation_matrix(list(range(257)), np.float64)
    with pytest.raises(ValueError):
        permutation_matrix([], np.float64)


def test_transformed_attention_bound_is_checked_before_allocation():
    p = np.concatenate((np.eye(128), np.zeros((128, 128))), axis=1)
    key = KeyPair(p, p.T)
    weights = AttentionWeights(
        np.zeros((8, 128, 64)),
        np.zeros((1, 128, 64)),
        np.zeros((1, 128, 64)),
        np.zeros((8, 64, 128)),
    )
    with pytest.raises(ValueError, match="exceeds diagnostic bounds"):
        Oracle().attention_weights(
            weights,
            key,
            key,
            angles=np.zeros((1, 32)),
            scale=np.ones((1, 32)),
            value_maps=np.eye(64)[None],
            block_orders=[list(range(32))],
            windows=[[32]],
            maximum_window=32,
            kv_order=[0],
            within_order=list(range(8)),
            frequencies=np.ones((1, 32)),
            layout="interleaved",
            mechanism="MQA",
        )


def test_explicit_noise_requires_finite_matching_geometry():
    oracle = Oracle(noise_mode="explicit")
    key = isometry(2, 2)
    e, h = np.ones((3, 2)), np.ones((2, 3))
    for noise in (
        [np.zeros_like(e), np.zeros_like(h)],
        (np.zeros((3, 1)), np.zeros_like(h)),
        (np.ones_like(e, dtype=np.float32), np.zeros_like(h)),
        (np.full_like(e, np.nan), np.zeros_like(h)),
    ):
        with pytest.raises(ValueError):
            oracle.embedding_head(e, h, key, [2, 0, 1], noises=noise)
