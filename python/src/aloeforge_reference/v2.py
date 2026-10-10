"""Independent, bounded FP64/FP32 V2 mathematical DIAGNOSTIC oracle.

The paper audit explicitly blocks production/paper-reference admission. This
module implements no worker, model adapter, artifact writer or remote runtime.
"""

from dataclasses import dataclass

import numpy as np

from .keymat import (
    MAX_ELEMENTS,
    KeyPair,
    checked,
    dtype_of,
    finite_scalar,
    inverse,
    tensor,
)


class UnresolvedMethod(ValueError):
    """An omitted or ambiguous method specification blocks profile admission."""


def permutation(order, size):
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or not 1 <= size <= 256
        or not isinstance(order, (list, tuple))
        or len(order) != size
        or any(not isinstance(i, int) or isinstance(i, bool) for i in order)
        or sorted(order) != list(range(size))
    ):
        raise ValueError("invalid original-to-physical permutation")
    return np.array(order, dtype=np.int64)


def permutation_matrix(order, dtype):
    p = permutation(order, len(order))
    # Row vectors: original coordinate i goes to physical p[i].
    return np.eye(len(p), dtype=dtype_of(dtype))[:, np.argsort(p)]


def _pairs(width, layout):
    if width < 2 or width > 64 or width % 2:
        raise ValueError("only full even RoPE head dimensions are admitted")
    if layout == "interleaved":
        return [(i, i + 1) for i in range(0, width, 2)]
    if layout == "split-half":
        return [(i, i + width // 2) for i in range(width // 2)]
    raise ValueError("unknown RoPE layout")


def scales(values, size, dtype):
    tensor(values, 1, dtype)
    if (
        values.shape != (size,)
        or (np.abs(values) < 0.125).any()
        or (np.abs(values) > 8).any()
    ):
        raise ValueError("invalid diagonal scale")
    return values


def block_order(order, windows, maximum_window):
    """Explicit full windows; no inferred gamma, sampling law or omitted tail."""
    p = permutation(order, len(order))
    if (
        not isinstance(maximum_window, int)
        or isinstance(maximum_window, bool)
        or not 1 <= maximum_window <= len(p)
        or not isinstance(windows, (list, tuple))
        or any(
            not isinstance(w, int)
            or isinstance(w, bool)
            or not 1 <= w <= maximum_window
            for w in windows
        )
        or sum(windows) != len(p)
    ):
        raise ValueError("invalid explicit block windows")
    start = 0
    for width in windows:
        if set(p[start : start + width]) != set(range(start, start + width)):
            raise ValueError("block permutation crosses declared window")
        start += width
    return p


def paper_block_sampler(*_args, **_kwargs):
    raise UnresolvedMethod("BLOCK-01: gamma, loop progress and tail are unspecified")


def _head_maps(angles, scale, order, windows, maximum_window, layout, dtype):
    tensor(angles, 1, dtype)
    pairs = _pairs(2 * len(angles), layout)
    scales(scale, len(pairs), dtype)
    p = block_order(order, windows, maximum_window)
    r = np.eye(2 * len(pairs), dtype=dtype)
    h = np.eye(2 * len(pairs), dtype=dtype)
    z = np.zeros_like(r)
    for i, (a, b) in enumerate(pairs):
        cosine, sine = np.cos(angles[i]), np.sin(angles[i])
        r[np.ix_([a, b], [a, b])] = [[cosine, -sine], [sine, cosine]]
        h[a, a] = h[b, b] = scale[i]
        dest = pairs[p[i]]
        z[a, dest[0]] = z[b, dest[1]] = 1
    return r, h, z, p


def rope(rows, positions, frequencies, *, layout):
    tensor(rows, 2)
    tensor(frequencies, 1, rows.dtype)
    pairs = _pairs(rows.shape[1], layout)
    if (
        not isinstance(positions, np.ndarray)
        or positions.dtype.kind not in "iu"
        or positions.ndim != 1
        or positions.shape[0] != rows.shape[0]
        or (positions < 0).any()
        or (positions > 1000000).any()
        or (np.diff(positions.astype(np.int64)) <= 0).any()
        or frequencies.shape != (len(pairs),)
        or (frequencies <= 0).any()
    ):
        raise ValueError("invalid explicit RoPE positions/frequencies")
    out = rows.copy()
    # Right multiplication by each row-coordinate rotation block.
    for i, (a, b) in enumerate(pairs):
        theta = positions.astype(rows.dtype) * frequencies[i]
        cosine, sine = np.cos(theta), np.sin(theta)
        out[:, a] = rows[:, a] * cosine + rows[:, b] * sine
        out[:, b] = -rows[:, a] * sine + rows[:, b] * cosine
    return checked(out)


@dataclass(frozen=True)
class AttentionWeights:
    q: np.ndarray  # [query_heads, hidden_in, head_dim]
    k: np.ndarray  # [kv_heads, hidden_in, head_dim]
    v: np.ndarray  # [kv_heads, hidden_in, head_dim]
    o: np.ndarray  # [query_heads, head_dim, hidden_out]

    def validate(self):
        tensor(self.q, 3)
        for value in (self.k, self.v, self.o):
            tensor(value, 3, self.q.dtype)
        heads, hidden, width = self.q.shape
        groups = self.k.shape[0]
        _pairs(width, "interleaved")
        if (
            heads > 8
            or groups > heads
            or heads % groups
            or self.k.shape != (groups, hidden, width)
            or self.v.shape != self.k.shape
            or self.o.shape[:2] != (heads, width)
        ):
            raise ValueError("unsupported MHA/GQA/MQA weight geometry")
        return heads, groups, width


def attention(rows, weights, positions, frequencies, *, layout, causal=True):
    """Bounded causal prefill oracle; no model/cache/backend implementation."""
    heads, groups, width = weights.validate()
    tensor(rows, 2, weights.q.dtype)
    tensor(frequencies, 2, rows.dtype)
    if (
        rows.shape[1] != weights.q.shape[1]
        or rows.shape[0] > 64
        or frequencies.shape != (groups, width // 2)
        or causal is not True
    ):
        raise ValueError("unsupported attention inputs or masking")
    outputs = np.zeros((rows.shape[0], weights.o.shape[2]), dtype=rows.dtype)
    scores = []
    for head in range(heads):
        group = head // (heads // groups)
        query = rope(
            rows @ weights.q[head], positions, frequencies[group], layout=layout
        )
        key = rope(
            rows @ weights.k[group], positions, frequencies[group], layout=layout
        )
        score = checked((query @ key.T) / rows.dtype.type(np.sqrt(width)))
        masked = np.where(np.triu(np.ones(score.shape, dtype=bool), 1), -np.inf, score)
        logits = masked - np.max(masked, axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        outputs += probability @ (rows @ weights.v[group]) @ weights.o[head]
        scores.append(score)
    return checked(outputs), np.stack(scores)


def silu(rows):
    tensor(rows, 2)
    exponential = np.exp(-np.abs(rows))
    result = np.where(
        rows >= 0, rows / (1 + exponential), rows * exponential / (1 + exponential)
    )
    return checked(result)


def ffn(rows, gate, up, down):
    tensor(rows, 2)
    for value in (gate, up, down):
        tensor(value, 2, rows.dtype)
    if (
        gate.shape != up.shape
        or gate.shape[0] != rows.shape[1]
        or down.shape[0] != gate.shape[1]
    ):
        raise ValueError("invalid dense bias-free FFN geometry")
    return checked((silu(rows @ gate) * (rows @ up)) @ down)


def rmsnorm(rows, gamma, *, epsilon):
    epsilon = finite_scalar(epsilon)
    tensor(rows, 2)
    tensor(gamma, 1, rows.dtype)
    if gamma.shape != (rows.shape[1],) or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("invalid explicit RMSNorm parameters")
    square_mean = checked(np.mean(rows * rows, axis=1, keepdims=True))
    denominator = np.sqrt(square_mean + rows.dtype.type(epsilon))
    if (denominator <= 0).any() or not np.isfinite(denominator).all():
        raise ValueError("invalid RMSNorm denominator")
    return checked(rows / denominator * gamma)


def norm_coefficient(samples, key, *, convention):
    """Explicit empirical calibration, not an assumed Gaussian population mean."""
    tensor(samples, 2, key.p.dtype)
    physical = key.encode(samples)
    source = np.linalg.norm(samples, axis=1)
    target = np.linalg.norm(physical, axis=1)
    if (source <= 0).any():
        raise ValueError("zero calibration row")
    checked(source)
    checked(target)
    ratio = checked(target / source)
    if convention == "rms":
        ratio *= np.sqrt(key.p.shape[0] / key.p.shape[1])
    elif convention != "l2":
        raise ValueError("calibration convention must be L2 or RMS")
    return float(np.mean(ratio))


def transformed_rmsnorm(rows, key, gamma, *, kappa, epsilon):
    """Physical RMS + explicit gamma fold; approximate for anisotropic P."""
    kappa = finite_scalar(kappa)
    tensor(rows, 2, key.p.dtype)
    tensor(gamma, 1, key.p.dtype)
    if (
        rows.shape[1] != key.p.shape[1]
        or gamma.shape != (key.p.shape[0],)
        or not np.isfinite(kappa)
        or kappa <= 0
    ):
        raise ValueError("invalid transformed normalization parameters")
    scalar = np.ones(rows.shape[1], dtype=rows.dtype) * rows.dtype.type(kappa)
    normalized = rmsnorm(rows, scalar, epsilon=epsilon)
    folded = (key.q * gamma[None, :]) @ key.p
    return checked(normalized @ folded)


def bf16_roundtrip(values):
    """Software finite BF16 nearest-even conversion, returned in FP32 carrier.

    This is not a BF16 arithmetic kernel. Reject conversion overflow/NaN/Inf.
    """
    if not isinstance(values, np.ndarray) or not 1 <= values.ndim <= 3:
        raise ValueError("invalid BF16 diagnostic tensor")
    tensor(values, values.ndim)
    with np.errstate(over="ignore"):
        rounded_input = values.astype(np.float32)
    checked(rounded_input)
    bits = rounded_input.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return checked(rounded.view(np.float32))


class Oracle:
    """Explicit diagnostic profile; unresolved paper choices block production."""

    def __init__(self, *, dtype=np.float64, profile="diagnostic", noise_mode="off"):
        if profile != "diagnostic":
            raise UnresolvedMethod(
                "V2 audit unresolved: production/paper-reference blocked"
            )
        if noise_mode not in ("off", "explicit"):
            raise ValueError("unknown diagnostic noise mode")
        self.dtype = dtype_of(dtype)
        self.noise_mode = noise_mode

    def _key(self, key):
        if not isinstance(key, KeyPair) or key.p.dtype != self.dtype:
            raise ValueError("invalid oracle key dtype")

    def embedding_head(self, embedding, head, key, token_order, *, noises=None):
        self._key(key)
        tensor(embedding, 2, self.dtype)
        tensor(head, 2, self.dtype)
        if (
            head.shape != (embedding.shape[1], embedding.shape[0])
            or embedding.shape[1] != key.p.shape[0]
        ):
            raise ValueError("invalid embedding/head orientation")
        p = permutation(token_order, embedding.shape[0])
        if self.noise_mode == "off":
            if noises is not None:
                raise ValueError("noise-off mode refuses noise")
        else:
            if not isinstance(noises, tuple) or len(noises) != 2:
                raise ValueError("explicit mode requires two supplied additive noises")
            for value, weight in zip(noises, (embedding, head)):
                tensor(value, 2, self.dtype)
                if value.shape != weight.shape:
                    raise ValueError("noise geometry mismatch")
            embedding = checked(embedding + noises[0])
            head = checked(head + noises[1])
        return checked((embedding @ key.p)[np.argsort(p)]), checked(
            (key.q @ head)[:, np.argsort(p)]
        )

    def ffn_weights(self, gate, up, down, key_in, key_out, order, scale):
        self._key(key_in)
        self._key(key_out)
        for value in (gate, up, down):
            tensor(value, 2, self.dtype)
        d = key_in.p.shape[0]
        if (
            gate.shape != up.shape
            or gate.shape[0] != d
            or down.shape != (gate.shape[1], key_out.p.shape[0])
        ):
            raise ValueError("invalid FFN transformation geometry")
        z = permutation_matrix(order, self.dtype)
        if z.shape[0] != gate.shape[1]:
            raise ValueError("FFN permutation dimension mismatch")
        scales(scale, gate.shape[1], self.dtype)
        return (
            checked(key_in.q @ gate @ z),
            checked((key_in.q @ up * scale) @ z),
            checked(z.T @ (down / scale[:, None]) @ key_out.p),
        )

    def attention_weights(
        self,
        weights,
        key_in,
        key_out,
        *,
        angles,
        scale,
        value_maps,
        block_orders,
        windows,
        maximum_window,
        kv_order,
        within_order,
        frequencies,
        layout,
        block_mode="identity",
        variant="coherent",
        mechanism="GQA",
    ):
        if mechanism not in ("MHA", "GQA", "MQA"):
            raise UnresolvedMethod(
                "MLA/MoE/unknown attention lacks an admitted contract"
            )
        self._key(key_in)
        self._key(key_out)
        heads, groups, width = weights.validate()
        if (
            weights.q.dtype != self.dtype
            or weights.q.shape[1] != key_in.p.shape[0]
            or weights.o.shape[2] != key_out.p.shape[0]
            or mechanism == "MHA"
            and groups != heads
            or mechanism == "MQA"
            and groups != 1
        ):
            raise ValueError("attention logical geometry mismatch")
        for value, rank in ((angles, 2), (scale, 2), (value_maps, 3), (frequencies, 2)):
            tensor(value, rank, self.dtype)
        if (
            angles.shape != (groups, width // 2)
            or scale.shape != angles.shape
            or frequencies.shape != angles.shape
            or (frequencies <= 0).any()
            or value_maps.shape != (groups, width, width)
            or len(block_orders) != groups
            or len(windows) != groups
            or block_mode
            not in ("identity", "relocate-frequencies", "fixed-frequencies")
            or variant not in ("coherent", "paper-literal")
        ):
            raise ValueError("invalid explicit attention transform specification")
        kv_map = permutation(kv_order, groups)
        group_size = heads // groups
        within_map = permutation(within_order, group_size)
        if (
            max(heads * key_in.p.shape[1] * width, heads * width * key_out.p.shape[1])
            > MAX_ELEMENTS
        ):
            raise ValueError("transformed attention tensor exceeds diagnostic bounds")
        q = np.empty((heads, key_in.p.shape[1], width), dtype=self.dtype)
        k = np.empty((groups, key_in.p.shape[1], width), dtype=self.dtype)
        v = np.empty_like(k)
        o = np.empty((heads, width, key_out.p.shape[1]), dtype=self.dtype)
        moved_frequencies = np.empty_like(frequencies)
        for group in range(groups):
            r, h, z, order = _head_maps(
                angles[group],
                scale[group],
                block_orders[group],
                windows[group],
                maximum_window,
                layout,
                self.dtype,
            )
            if block_mode == "identity" and not np.array_equal(
                order, np.arange(width // 2)
            ):
                raise ValueError("exact identity mode refuses a block permutation")
            dest = kv_map[group]
            key_z = z.T if variant == "paper-literal" else z
            k[dest] = key_in.q @ weights.k[group] @ r @ inverse(h) @ key_z
            v[dest] = key_in.q @ weights.v[group] @ value_maps[group]
            u_inverse = inverse(value_maps[group])
            moved_frequencies[dest] = (
                frequencies[group, np.argsort(order)]
                if block_mode == "relocate-frequencies"
                else frequencies[group]
            )
            for j in range(group_size):
                source = group * group_size + j
                target = dest * group_size + within_map[j]
                q[target] = key_in.q @ weights.q[source] @ r @ h @ z
                o[target] = u_inverse @ weights.o[source] @ key_out.p
        for value in (q, k, v, o, moved_frequencies):
            checked(value)
        return AttentionWeights(q, k, v, o), moved_frequencies
