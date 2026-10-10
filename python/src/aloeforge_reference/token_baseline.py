"""Independent P2 diagnostic oracle, using Python integers and FP64 dot products.

No Go implementation, worker code, transformer library or tensor runtime is
imported. Matrices are bounded synthetic fixtures, not production model weights.
Token permutation is not encryption or a complete AloePri V2 method.
"""

import hashlib
import json
import math


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def inverse(permutation):
    if not 2 <= len(permutation) <= 65536:
        raise ValueError("invalid permutation size")
    if any(not _integer(i) for i in permutation):
        raise ValueError("invalid permutation values")
    if set(permutation) != set(range(len(permutation))):
        raise ValueError("permutation is not bijective")
    result = [0] * len(permutation)
    for original, physical in enumerate(permutation):
        result[physical] = original
    return result


def _matrix(matrix, vocab):
    if len(matrix) != vocab or not matrix:
        raise ValueError("matrix vocabulary mismatch")
    hidden = len(matrix[0])
    if not 1 <= hidden <= 256 or vocab * hidden > 1048576:
        raise ValueError("matrix size exceeds bounds")
    for row in matrix:
        if len(row) != hidden or any(not math.isfinite(x) for x in row):
            raise ValueError("invalid matrix row")
    return hidden


def permute_rows(matrix, permutation):
    """Return output physical row j = source original row inverse(p)[j]."""
    inv = inverse(permutation)
    _matrix(matrix, len(permutation))
    return [list(matrix[original]) for original in inv]


def _special_ids(ids, vocab):
    if set(ids) != {"bos", "eos", "pad"}:
        raise ValueError("invalid special roles")
    for role, values in ids.items():
        if not isinstance(values, list):
            raise ValueError("invalid special list")
        if role == "eos" and not 1 <= len(values) <= 16:
            raise ValueError("invalid EOS count")
        if role != "eos" and len(values) > 1:
            raise ValueError("invalid BOS/PAD count")
        if any(not _integer(i) or i < 0 or i >= vocab for i in values):
            raise ValueError("special ID out of range")
        if values != sorted(set(values)):
            raise ValueError("special IDs must be sorted and unique")


def map_special_ids(ids, permutation):
    inverse(permutation)
    _special_ids(ids, len(permutation))
    return {
        role: sorted(permutation[i] for i in values) for role, values in ids.items()
    }


def logits(embedding, head, token_id):
    """Toy embedding -> dot-product LM head. None denotes a tied head alias."""
    hidden = _matrix(embedding, len(embedding))
    if head is None:
        head = embedding
    if _matrix(head, len(embedding)) != hidden:
        raise ValueError("head hidden dimension mismatch")
    if not _integer(token_id) or not 0 <= token_id < len(embedding):
        raise ValueError("token ID out of range")
    result = [
        math.fsum(a * b for a, b in zip(embedding[token_id], row)) for row in head
    ]
    if any(not math.isfinite(x) for x in result):
        raise ValueError("nonfinite logits")
    return result


def restore_logits(physical, permutation):
    inverse(permutation)
    if len(physical) != len(permutation) or any(not math.isfinite(x) for x in physical):
        raise ValueError("invalid physical logits")
    return [physical[p] for p in permutation]


def _hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def binding_commitment(material):
    """Reconstruct typed canonical JSON independently of Go's encoder."""
    if set(material) != {"context", "nonce", "permutation"}:
        raise ValueError("invalid private material fields")
    context_fields = (
        "source",
        "tokenizer_sha256",
        "vocabulary_size",
        "hidden_size",
        "tied_embeddings",
        "special_token_ids",
        "method_revision",
        "rng_algorithm",
        "rng_revision",
    )
    context = material["context"]
    if set(context) != set(context_fields):
        raise ValueError("invalid private context fields")
    source = context["source"]
    if set(source) != {"inventory_sha256", "content_sha256"}:
        raise ValueError("invalid source binding fields")
    if not all(
        _hash(x)
        for x in (*source.values(), context["tokenizer_sha256"], material["nonce"])
    ):
        raise ValueError("invalid private fingerprint")
    permutation = material["permutation"]
    inverse(permutation)
    vocab, hidden = context["vocabulary_size"], context["hidden_size"]
    if (
        not _integer(vocab)
        or vocab != len(permutation)
        or not _integer(hidden)
        or not 1 <= hidden <= 256
        or vocab * hidden > 1048576
        or not isinstance(context["tied_embeddings"], bool)
    ):
        raise ValueError("invalid private dimensions")
    _special_ids(context["special_token_ids"], vocab)
    if (
        context["method_revision"] != "token-permutation-v1"
        or context["rng_algorithm"] != "crypto-rand-fisher-yates"
        or context["rng_revision"] != "1"
    ):
        raise ValueError("unsupported private revisions")
    canonical_context = {name: context[name] for name in context_fields}
    canonical_context["source"] = {
        name: source[name] for name in ("inventory_sha256", "content_sha256")
    }
    canonical_context["special_token_ids"] = dict(
        sorted(context["special_token_ids"].items())
    )
    canonical = {
        "context": canonical_context,
        "nonce": material["nonce"],
        "permutation": permutation,
    }
    encoded = json.dumps(canonical, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(b"AloeForge/token-binding/v1\n" + encoded).hexdigest()


def validate_secret(secret):
    if set(secret) != {"schema_version", "binding_id", "material"}:
        raise ValueError("invalid secret fields")
    if not _integer(secret["schema_version"]) or secret["schema_version"] != 1:
        raise ValueError("unsupported secret version")
    if (
        not _hash(secret["binding_id"])
        or binding_commitment(secret["material"]) != secret["binding_id"]
    ):
        raise ValueError("secret commitment mismatch")
