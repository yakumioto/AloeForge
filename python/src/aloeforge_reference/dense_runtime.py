"""Independent artifact-only Qwen2.5 diagnostic runtime on NumPy CPU.

No transform worker, private key, source model or HF remote code is imported.
The loader requires identities obtained from the trusted compiler and verifies
the exact published manifest bytes, tensor schema and streamed payload hashes.
"""

import hashlib
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

KIND = "p5-qwen25-dense-diagnostic-only"
METHOD = "qwen25-isometry-noise-off-v1"
CONTRACT_FIELDS = (
    "schema_version profile method_revision rng_algorithm rng_revision loader_revision "
    "binding_id tokenizer_sha256 vocabulary_size logical_hidden_size "
    "physical_hidden_size intermediate_size num_hidden_layers num_attention_heads "
    "num_key_value_heads head_dim max_position_embeddings logical_norm_epsilon "
    "rope_theta rope_layout cache_layout source_tied_embeddings "
    "mapped_special_token_ids"
).split()


def _hash(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def _int(value):
    return type(value) is int


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json(raw):
    text = raw.decode("utf-8", errors="strict")
    if any(0xD800 <= ord(x) <= 0xDFFF for x in text):
        raise ValueError("invalid Unicode")
    result = json.loads(
        text,
        object_pairs_hook=_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )

    def check(value, depth=0):
        if depth > 32:
            raise ValueError("JSON nesting exceeds bounds")
        if isinstance(value, str):
            value.encode("utf-8", errors="strict")
        elif isinstance(value, dict):
            for key, item in value.items():
                check(key, depth + 1)
                check(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                check(item, depth + 1)

    check(result)
    return result


def _encode(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def validate_contract(c):
    if not isinstance(c, dict) or set(c) != set(CONTRACT_FIELDS):
        raise ValueError("unsupported dense contract fields")
    required = {
        "schema_version": 1,
        "profile": "diagnostic",
        "method_revision": METHOD,
        "rng_algorithm": "crypto-rand-sha256-pcg64",
        "rng_revision": "1",
        "loader_revision": "numpy-qwen25-v1",
        "rope_layout": "split-half",
        "cache_layout": "layer-kv-head-position-dim-v1",
    }
    if any(c[k] != v or type(c[k]) is not type(v) for k, v in required.items()):
        raise ValueError("unsupported dense method or runtime semantics")
    if not _hash(c["binding_id"]) or not _hash(c["tokenizer_sha256"]):
        raise ValueError("invalid public binding")
    bounds = {
        "vocabulary_size": (2, 262144),
        "logical_hidden_size": (2, 8192),
        "physical_hidden_size": (6, 8320),
        "intermediate_size": (2, 32768),
        "num_hidden_layers": (1, 64),
        "num_attention_heads": (1, 64),
        "num_key_value_heads": (1, 64),
        "head_dim": (2, 256),
        "max_position_embeddings": (1, 131072),
    }
    for key, (low, high) in bounds.items():
        if not _int(c[key]) or not low <= c[key] <= high:
            raise ValueError("unsupported dense geometry")
    d, width = c["logical_hidden_size"], c["physical_hidden_size"]
    heads, kv, dim = (
        c["num_attention_heads"],
        c["num_key_value_heads"],
        c["head_dim"],
    )
    if not 4 <= width - d <= 128 or (width - d) % 4 or heads % kv:
        raise ValueError("unsupported residual expansion or GQA")
    if heads * dim != d or dim % 2:
        raise ValueError("unsupported rotary geometry")
    for key, low, high in [
        ("logical_norm_epsilon", 0, 1),
        ("rope_theta", 1, 1e9),
    ]:
        value = c[key]
        if type(value) not in {int, float} or not math.isfinite(value):
            raise ValueError("invalid norm or rotary scalar")
        if not low <= value <= high or key == "logical_norm_epsilon" and value == 0:
            raise ValueError("invalid norm or rotary scalar")
    if type(c["source_tied_embeddings"]) is not bool:
        raise ValueError("invalid tied source declaration")
    ids = c["mapped_special_token_ids"]
    if not isinstance(ids, dict) or set(ids) != {"bos", "eos", "pad"}:
        raise ValueError("invalid mapped special roles")
    for role, values in ids.items():
        if not isinstance(values, list) or len(values) > (16 if role == "eos" else 1):
            raise ValueError("invalid special count")
        if role == "eos" and not values:
            raise ValueError("missing EOS")
        if any(not _int(i) or not 0 <= i < c["vocabulary_size"] for i in values):
            raise ValueError("invalid special ID")
        if sorted(set(values)) != values:
            raise ValueError("noncanonical special IDs")


def _shapes(c):
    width, vocab = c["physical_hidden_size"], c["vocabulary_size"]
    q = c["num_attention_heads"] * c["head_dim"]
    kv = c["num_key_value_heads"] * c["head_dim"]
    f = c["intermediate_size"]
    result = {"embedding.weight": [vocab, width], "head.weight": [vocab, width]}
    for i in range(c["num_hidden_layers"]):
        for role, shape in {
            "q.weight": [q, width],
            "q.bias": [q],
            "k.weight": [kv, width],
            "k.bias": [kv],
            "v.weight": [kv, width],
            "v.bias": [kv],
            "o.weight": [width, q],
            "gate.weight": [f, width],
            "up.weight": [f, width],
            "down.weight": [width, f],
        }.items():
            result[f"layers.{i}.{role}"] = shape
    return result


def _open(root, name):
    if Path(name).name != name or name.startswith("."):
        raise ValueError("invalid artifact basename")
    fd = os.open(root / name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("artifact entry is not regular")
    return os.fdopen(fd, "rb")


def _read(root, name, limit):
    with _open(root, name) as file:
        data = file.read(limit + 1)
    if len(data) > limit:
        raise ValueError("artifact control exceeds bounds")
    return data


@dataclass(frozen=True)
class Cache:
    """Private session state owned by one runtime instance, in physical KV order."""

    owner: object
    length: int
    layers: tuple


class Runtime:
    @classmethod
    def load(
        cls,
        root,
        *,
        artifact_id,
        binding_id,
        tokenizer_sha256,
        compute_dtype=np.float64,
        max_weight_bytes=4 << 30,
        max_cache_bytes=512 << 20,
        max_attention_bytes=64 << 20,
        max_activation_bytes=128 << 20,
        max_logit_bytes=256 << 20,
    ):
        if not all(_hash(x) for x in [artifact_id, binding_id, tokenizer_sha256]):
            raise ValueError("trusted external identities required")
        if compute_dtype not in {np.float32, np.float64}:
            raise ValueError("unsupported compute dtype")
        if np.__version__ != "2.3.3":
            raise ValueError("unsupported numerical runtime toolchain")
        for budget in [
            max_weight_bytes,
            max_cache_bytes,
            max_attention_bytes,
            max_activation_bytes,
            max_logit_bytes,
        ]:
            if not _int(budget) or budget < 1:
                raise ValueError("invalid allocation budget")
        root = Path(root)
        raw = _read(root, "manifest.json", 8 << 20)
        if hashlib.sha256(raw).hexdigest() != artifact_id:
            raise ValueError("artifact identity mismatch")
        manifest = _json(raw)
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "kind",
            "descriptor_sha256",
            "outputs",
            "files",
            "dense_contract",
        }:
            raise ValueError("unsupported manifest fields")
        if type(manifest["schema_version"]) is not int or (
            manifest["schema_version"] != 1 or manifest["kind"] != KIND
        ):
            raise ValueError("unsupported artifact kind")
        c = manifest["dense_contract"]
        validate_contract(c)
        if c["binding_id"] != binding_id or c["tokenizer_sha256"] != tokenizer_sha256:
            raise ValueError("public binding mismatch")
        expected = _shapes(c)
        outputs = manifest["outputs"]
        if not isinstance(outputs, list) or len(outputs) != len(expected):
            raise ValueError("tensor completeness mismatch")
        groups, seen, total = {}, set(), 0
        for t in outputs:
            if not isinstance(t, dict) or list(t) != [
                "name",
                "shard",
                "dtype",
                "shape",
                "payload_sha256",
            ]:
                raise ValueError("unsupported output descriptor")
            name, shard = t["name"], t["shard"]
            if not isinstance(name, str) or name in seen or name not in expected:
                raise ValueError("unknown or duplicated tensor role")
            if (
                t["dtype"] != "F32"
                or t["shape"] != expected[name]
                or any(not _int(i) for i in t["shape"])
            ):
                raise ValueError("tensor dtype or shape mismatch")
            if (
                not isinstance(shard, str)
                or not re.fullmatch(
                    "[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\\.safetensors", shard
                )
                or not _hash(t["payload_sha256"])
            ):
                raise ValueError("invalid shard or payload hash")
            total += 4 * math.prod(t["shape"])
            if total > max_weight_bytes:
                raise ValueError("weight allocation budget exceeded")
            seen.add(name)
            groups.setdefault(shard, []).append(t)
        if len(groups) > 128 or outputs != sorted(
            outputs, key=lambda t: (t["shard"], t["name"])
        ):
            raise ValueError("noncanonical output inventory")
        if (
            hashlib.sha256(_encode(outputs)).hexdigest()
            != manifest["descriptor_sha256"]
        ):
            raise ValueError("descriptor hash mismatch")
        names = sorted(groups)
        files = manifest["files"]
        if not isinstance(files, list) or len(files) != len(names):
            raise ValueError("shard completeness mismatch")
        if set(os.listdir(root)) != {"manifest.json", *names}:
            raise ValueError("unexpected artifact entry")
        tensors = {}
        for record, name in zip(files, names, strict=True):
            if not isinstance(record, dict) or set(record) != {"name", "sha256"}:
                raise ValueError("unsupported file descriptor")
            if record["name"] != name or not _hash(record["sha256"]):
                raise ValueError("invalid file hash")
            entries, offset = {}, 0
            for t in groups[name]:
                size = 4 * math.prod(t["shape"])
                entries[t["name"]] = {
                    "dtype": "F32",
                    "shape": t["shape"],
                    "data_offsets": [offset, offset + size],
                }
                offset += size
            header = _encode(dict(sorted(entries.items())))
            header += b" " * (-len(header) % 8)
            if len(header) > 8 << 20:
                raise ValueError("tensor header exceeds bounds")
            with _open(root, name) as file:
                prefix = file.read(8)
                if (
                    prefix != struct.pack("<Q", len(header))
                    or file.read(len(header)) != header
                ):
                    raise ValueError("noncanonical tensor header")
                if os.fstat(file.fileno()).st_size != 8 + len(header) + offset:
                    raise ValueError("tensor payload length mismatch")
                file.seek(0)
                digest = hashlib.sha256()
                while block := file.read(32 << 10):
                    digest.update(block)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError("shard hash mismatch")
                for t in groups[name]:
                    start, end = entries[t["name"]]["data_offsets"]
                    file.seek(8 + len(header) + start)
                    left, digest = end - start, hashlib.sha256()
                    while left:
                        block = file.read(min(left, 32 << 10))
                        if (
                            not block
                            or not np.isfinite(np.frombuffer(block, "<f4")).all()
                        ):
                            raise ValueError("nonfinite or truncated tensor")
                        digest.update(block)
                        left -= len(block)
                    if digest.hexdigest() != t["payload_sha256"]:
                        raise ValueError("tensor hash mismatch")
                    # Map the same verified open file, not a reopened pathname.
                    array = np.memmap(
                        file,
                        dtype="<f4",
                        mode="r",
                        offset=8 + len(header) + start,
                        shape=tuple(t["shape"]),
                    )
                    tensors[t["name"]] = array
        instance = cls()
        instance.contract, instance.tensors = c, tensors
        instance.dtype = compute_dtype
        instance.max_cache_bytes, instance.max_attention_bytes = (
            max_cache_bytes,
            max_attention_bytes,
        )
        instance._owner = object()
        instance.max_activation_bytes = max_activation_bytes
        instance.max_logit_bytes = max_logit_bytes
        return instance

    def _norm(self, x):
        c = self.contract
        ratio = c["logical_hidden_size"] / c["physical_hidden_size"]
        with np.errstate(over="ignore", invalid="ignore"):
            variance = np.mean(x * x, axis=-1, keepdims=True)
        if not np.isfinite(variance).all():
            raise ValueError("nonfinite residual norm")
        return (
            math.sqrt(ratio) * x / np.sqrt(variance + ratio * c["logical_norm_epsilon"])
        )

    def _rotary(self, x, positions):
        dim = self.contract["head_dim"]
        frequencies = self.contract["rope_theta"] ** (
            -np.arange(0, dim, 2, dtype=self.dtype) / dim
        )
        angles = np.asarray(positions, dtype=self.dtype)[:, None] * frequencies
        cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
        first, second = x[..., : dim // 2], x[..., dim // 2 :]
        return np.concatenate(
            [first * cos - second * sin, second * cos + first * sin], -1
        )

    def forward(self, physical_ids, cache=None):
        """Prefill/decode; only physical IDs are accepted, with contiguous positions."""
        c = self.contract
        if (
            not isinstance(physical_ids, (list, tuple))
            or not physical_ids
            or any(
                not _int(i) or not 0 <= i < c["vocabulary_size"] for i in physical_ids
            )
        ):
            raise ValueError("invalid physical token request")
        if cache is not None:
            if not isinstance(cache, Cache) or cache.owner is not self._owner:
                raise ValueError("cache belongs to another runtime")
            if (
                not _int(cache.length)
                or cache.length < 1
                or len(cache.layers) != (c["num_hidden_layers"])
            ):
                raise ValueError("invalid cache geometry")
            shape = (c["num_key_value_heads"], cache.length, c["head_dim"])
            for pair in cache.layers:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise ValueError("invalid cache layer")
                for value in pair:
                    if not isinstance(value, np.ndarray) or value.shape != shape:
                        raise ValueError("invalid cache tensor geometry")
                    if value.dtype != self.dtype or not np.isfinite(value).all():
                        raise ValueError("invalid cache tensor dtype or value")
        start = 0 if cache is None else cache.length
        length, total = len(physical_ids), start + len(physical_ids)
        if total > c["max_position_embeddings"]:
            raise ValueError("context limit exceeded")
        itemsize = np.dtype(self.dtype).itemsize
        activations = (
            length
            * itemsize
            * (
                4 * c["physical_hidden_size"]
                + 4 * c["intermediate_size"]
                + 4 * c["logical_hidden_size"]
            )
        )
        if (
            activations > self.max_activation_bytes
            or length * c["vocabulary_size"] * itemsize > self.max_logit_bytes
        ):
            raise ValueError("activation or logit allocation budget exceeded")
        cache_size = (
            2
            * c["num_hidden_layers"]
            * c["num_key_value_heads"]
            * (total * c["head_dim"] * itemsize)
        )
        # Two attention buffers and peak replacement cache are budgeted explicitly.
        if 2 * cache_size > self.max_cache_bytes or (
            2 * c["num_attention_heads"] * length * total * itemsize
            > self.max_attention_bytes
        ):
            raise ValueError("cache or attention allocation budget exceeded")
        positions = np.arange(start, total)
        hidden = np.asarray(
            self.tensors["embedding.weight"][physical_ids], dtype=self.dtype
        )
        new_cache = []
        for i in range(c["num_hidden_layers"]):
            prefix = f"layers.{i}."

            def project(x, role):
                return x @ self.tensors[prefix + role + ".weight"].T

            normalized = self._norm(hidden)
            q = project(normalized, "q") + self.tensors[prefix + "q.bias"]
            k = project(normalized, "k") + self.tensors[prefix + "k.bias"]
            v = project(normalized, "v") + self.tensors[prefix + "v.bias"]
            q = self._rotary(
                q.reshape(length, c["num_attention_heads"], c["head_dim"]), positions
            )
            k = self._rotary(
                k.reshape(length, c["num_key_value_heads"], c["head_dim"]), positions
            )
            k = k.transpose(1, 0, 2)
            v = v.reshape(length, c["num_key_value_heads"], c["head_dim"]).transpose(
                1, 0, 2
            )
            if cache is not None:
                k = np.concatenate([cache.layers[i][0], k], axis=1)
                v = np.concatenate([cache.layers[i][1], v], axis=1)
            k.setflags(write=False)
            v.setflags(write=False)
            new_cache.append((k, v))
            group = c["num_attention_heads"] // c["num_key_value_heads"]
            # Compute one group at a time; no full repeated-KV cache is allocated.
            attended = np.empty_like(q)
            for head in range(c["num_attention_heads"]):
                kv_head = head // group
                scores = q[:, head, :] @ k[kv_head].T / math.sqrt(c["head_dim"])
                scores[np.arange(total)[None, :] > positions[:, None]] = -np.inf
                scores -= np.max(scores, axis=-1, keepdims=True)
                probs = np.exp(scores)
                probs /= np.sum(probs, axis=-1, keepdims=True)
                attended[:, head, :] = probs @ v[kv_head]
            hidden = hidden + project(attended.reshape(length, -1), "o")
            normalized = self._norm(hidden)
            gate, up = project(normalized, "gate"), project(normalized, "up")
            # Stable SiLU for extreme finite gate inputs.
            sigmoid = np.exp(-np.logaddexp(0, -gate))
            hidden = hidden + project(gate * sigmoid * up, "down")
        logits = self._norm(hidden) @ self.tensors["head.weight"].T
        if not np.isfinite(logits).all():
            raise ValueError("nonfinite diagnostic logits")
        return logits, Cache(self._owner, total, tuple(new_cache))

    def greedy(self, physical_ids, max_new_tokens):
        if not _int(max_new_tokens) or max_new_tokens < 0:
            raise ValueError("invalid generation bound")
        if (
            not isinstance(physical_ids, (list, tuple))
            or not physical_ids
            or any(
                not _int(i) or not 0 <= i < self.contract["vocabulary_size"]
                for i in physical_ids
            )
        ):
            raise ValueError("invalid physical generation request")
        if max_new_tokens == 0:
            return []
        if (
            len(physical_ids) + max_new_tokens
            > self.contract["max_position_embeddings"]
        ):
            raise ValueError("generation exceeds context")
        logits, cache = self.forward(physical_ids)
        result = []
        for step in range(max_new_tokens):
            token = int(np.argmax(logits[-1]))
            result.append(token)
            if token in self.contract["mapped_special_token_ids"]["eos"]:
                break
            if step + 1 < max_new_tokens:
                logits, cache = self.forward([token], cache)
        return result
