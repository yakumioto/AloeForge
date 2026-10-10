"""Bounded DIAGNOSTIC P/Q construction; see docs/V2_MATH_AUDIT.md KEY-01/02.

NumPy only, no worker/model/runtime imports. Null-space directions and the
vertical Q stack are explicit engineering interpretations, not paper errata.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np

MAX_DIMENSION = 256
MAX_ELEMENTS = 65536
CONDITION_LIMIT = 32.0


def finite_scalar(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("invalid diagnostic scalar type")
    if not np.isfinite(value):
        raise ValueError("nonfinite diagnostic scalar")
    return float(value)


def dtype_of(dtype):
    value = np.dtype(dtype)
    if value not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("only FP32/FP64 diagnostic arrays are admitted")
    return value


def tensor(value, ndim, dtype=None):
    if not isinstance(value, np.ndarray) or value.ndim != ndim:
        raise ValueError("invalid diagnostic tensor rank")
    actual = dtype_of(value.dtype)
    if dtype is not None and actual != dtype_of(dtype):
        raise ValueError("mixed diagnostic dtypes")
    if (
        any(not 1 <= size <= MAX_DIMENSION for size in value.shape)
        or value.size > MAX_ELEMENTS
        or not np.isfinite(value).all()
    ):
        raise ValueError("invalid diagnostic tensor bounds or values")
    return value


def tolerance(dtype):
    return 3e-5 if dtype_of(dtype) == np.dtype("float32") else 2e-12


def checked(value):
    if not np.isfinite(value).all():
        raise ValueError("nonfinite diagnostic calculation")
    return value


def inverse(matrix):
    tensor(matrix, 2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("inverse requires a square matrix")
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > CONDITION_LIMIT:
        raise ValueError("singular or ill-conditioned diagnostic matrix")
    return checked(np.linalg.inv(matrix))


def generator(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("invalid diagnostic seed")
    return np.random.Generator(np.random.PCG64(seed))


def _orthogonal(rng, size, dtype):
    q, r = np.linalg.qr(rng.standard_normal((size, size)).astype(dtype))
    signs = np.where(np.diag(r) < 0, -1, 1).astype(dtype)
    return q * signs


def _null(matrix):
    _, singular, vh = np.linalg.svd(matrix, full_matrices=True)
    cutoff = np.finfo(matrix.dtype).eps * max(matrix.shape) * singular[0]
    rank = np.count_nonzero(singular > cutoff)
    return vh[rank:].T


@dataclass(frozen=True)
class KeyPair:
    """A right-inverse pair, not an encryption key or production Client Secret."""

    p: np.ndarray
    q: np.ndarray

    def __post_init__(self):
        p = tensor(self.p, 2)
        q = tensor(self.q, 2, p.dtype)
        d, physical = p.shape
        if q.shape != (physical, d) or physical < d or (physical - d) % 2:
            raise ValueError("invalid P/Q shapes")
        bound = tolerance(p.dtype)
        if (
            not np.allclose(p @ q, np.eye(d), atol=bound, rtol=bound)
            or np.linalg.cond(p) > CONDITION_LIMIT
            or np.linalg.cond(q) > CONDITION_LIMIT
        ):
            raise ValueError("P/Q cancellation or conditioning failed")
        for name, value in (("p", p), ("q", q)):
            copy = value.copy()
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)

    def require_inverse(self, other):
        if not isinstance(other, KeyPair) or other.q.dtype != self.p.dtype:
            raise ValueError("incompatible key pair")
        if other.q.shape != self.q.shape:
            raise ValueError("incompatible key dimensions")
        bound = tolerance(self.p.dtype)
        if not np.allclose(
            self.p @ other.q, np.eye(self.p.shape[0]), atol=bound, rtol=bound
        ):
            raise ValueError("cross-key cancellation failed")

    def encode(self, rows):
        tensor(rows, 2, self.p.dtype)
        if rows.shape[1] != self.p.shape[0]:
            raise ValueError("logical hidden dimension mismatch")
        return checked(rows @ self.p)

    def restore(self, rows):
        tensor(rows, 2, self.q.dtype)
        if rows.shape[1] != self.q.shape[0]:
            raise ValueError("physical hidden dimension mismatch")
        return checked(rows @ self.q)


class KeyFamily:
    """Shared B/E/F/Z bases; every member P_i cancels every member Q_j.

    Positive even h and explicit seeds are diagnostic choices. SVD/QR sampling
    and PCG64 are numerical fixtures, not cryptographic key generation.
    """

    def __init__(self, d, h, coefficient, *, seed, dtype=np.float64):
        self.dtype = dtype_of(dtype)
        coefficient = finite_scalar(coefficient)
        if (
            not isinstance(d, int)
            or isinstance(d, bool)
            or not 1 <= d <= 64
            or not isinstance(h, int)
            or isinstance(h, bool)
            or not 2 <= h <= 64
            or h % 2
            or not np.isfinite(coefficient)
            or not 0 <= coefficient <= 0.25
        ):
            raise ValueError("invalid diagnostic key parameters")
        self.d, self.h = d, h
        rng = generator(seed)

        def draw(shape):
            return (rng.standard_normal(shape) / np.sqrt(d)).astype(self.dtype)

        self.b = _orthogonal(rng, d, self.dtype) + coefficient * draw((d, d))
        self.b_inverse = inverse(self.b)
        self.e = draw((d, h // 2)) @ draw((h // 2, h))
        self.f = draw((h, h // 2)) @ draw((h // 2, d))
        self.z = _orthogonal(rng, d + 2 * h, self.dtype)
        self.null_f = _null(self.f.T)
        self.null_e = _null(self.e)
        if not self.null_f.shape[1] or not self.null_e.shape[1]:
            raise ValueError("diagnostic null space is empty")

    def member(self, *, seed):
        rng = generator(seed)
        c = rng.standard_normal((self.d, self.null_f.shape[1])).astype(self.dtype)
        c = c @ self.null_f.T / self.dtype.type(np.sqrt(self.d))
        n = self.null_e @ rng.standard_normal((self.null_e.shape[1], self.d)).astype(
            self.dtype
        )
        n = n / self.dtype.type(np.sqrt(self.d))
        bound = tolerance(self.dtype)
        if not np.allclose(c @ self.f, 0, atol=bound) or not np.allclose(
            self.e @ n, 0, atol=bound
        ):
            raise ValueError("null-space residual exceeds fixed tolerance")
        p = np.concatenate((self.b, c, self.e), axis=1) @ self.z
        q = self.z.T @ np.concatenate((self.b_inverse, self.f, n), axis=0)
        return KeyPair(p, q)


def isometry(d, h, *, scale=1.0, seed=0, dtype=np.float64):
    """Noise-off scaled-isometry ablation, NOT Algorithm 1's key distribution."""
    dtype = dtype_of(dtype)
    scale = finite_scalar(scale)
    if (
        not isinstance(d, int)
        or isinstance(d, bool)
        or not 1 <= d <= 64
        or not isinstance(h, int)
        or isinstance(h, bool)
        or not 2 <= h <= 64
        or h % 2
        or not np.isfinite(scale)
        or not 0.125 <= scale <= 8
    ):
        raise ValueError("invalid diagnostic isometry")
    z = _orthogonal(generator(seed), d + 2 * h, dtype)
    return KeyPair(scale * z[:d], z[:d].T / scale)
