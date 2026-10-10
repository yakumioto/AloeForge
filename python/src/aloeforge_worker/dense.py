"""Trusted offline diagnostic transforms. No reference/oracle imports.

ENGINEERING INTERPRETATION: exact sparse isometry, norm folding, paired Q/K
scales, group permutations, V/O inverse scales and FFN up/down inverse scales.
This is not Algorithm 1 or a production/private model protection scheme.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def _rng(seed, label):
    data = hashlib.sha256(bytes.fromhex(seed) + label.encode("ascii")).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(data[:16], "little")))


def coordinates(c, seed):
    """Private P represented by one nonzero per physical column; Q=P.T."""
    d, width = c["logical_hidden_size"], c["physical_hidden_size"]
    rng = _rng(seed, "residual-v1")
    logical = np.concatenate([np.arange(d), rng.integers(d, size=width - d)])
    logical = logical[rng.permutation(width)]
    counts = np.bincount(logical, minlength=d)
    factors = rng.choice([-1.0, 1.0], width) / np.sqrt(counts[logical])
    return logical, factors


def layer_maps(c, seed, layer):
    rng = _rng(seed, f"layer-{layer}-v1")
    heads, kv, dim = (
        c["num_attention_heads"],
        c["num_key_value_heads"],
        c["head_dim"],
    )
    group = heads // kv
    kv_order = rng.permutation(kv)
    within = rng.permutation(group)
    q_order = (kv_order[:, None] * group + within[None, :]).reshape(-1)
    channels = np.arange(dim)
    q_rows = (q_order[:, None] * dim + channels).reshape(-1)
    kv_rows = (kv_order[:, None] * dim + channels).reshape(-1)
    paired = rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], (kv, dim // 2))
    k_scale = np.concatenate([paired, paired], axis=1)
    q_scale = np.repeat(k_scale, group, axis=0).reshape(-1)
    v_scale = rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], (kv, dim))
    f_order = rng.permutation(c["intermediate_size"])
    f_scale = rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], len(f_order))
    return {
        "q_rows": q_rows,
        "kv_rows": kv_rows,
        "q_scale": q_scale,
        "k_scale": 1 / k_scale.reshape(-1),
        "v_scale": v_scale.reshape(-1),
        "o_scale": 1 / np.repeat(v_scale, group, axis=0).reshape(-1),
        "f_order": f_order,
        "f_scale": f_scale,
    }


def _source(root, inventory):
    shards = {s["name"]: s for s in inventory["shards"]}
    arrays = {}
    for t in inventory["tensors"]:
        path = (root / t["shard"]).resolve(strict=True)
        path.relative_to(root.resolve(strict=True))
        shard = shards[t["shard"]]
        if not path.is_file() or path.stat().st_size != shard["file_bytes"]:
            raise ValueError("source shard changed")
        dtype = {"F32": "<f4", "F16": "<f2", "BF16": "<u2"}[t["dtype"]]
        a = np.memmap(
            path,
            mode="r",
            dtype=dtype,
            offset=shard["data_start"] + t["data_offsets"][0],
            shape=tuple(t["shape"]),
        )
        arrays[t["name"]] = (a, t["dtype"])
    return arrays


def _convert(a, dtype):
    if dtype == "BF16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    result = np.asarray(a, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite source tensor")
    return result


def transform(plan, secret, source_root, scratch):
    """At most 64 output rows per block, with source payloads mapped read-only.

    Scratch holds raw finite F32 payloads; Go validates names/shapes/hashes and
    consumes them into its own stage. Worker never writes a published artifact.
    """
    if plan["schema_version"] != 1:
        raise ValueError("unsupported transform plan")
    if np.__version__ != "2.3.3":
        raise ValueError("unsupported numerical RNG toolchain")
    c = plan["contract"]
    if c["profile"] != "diagnostic" or c["method_revision"] != (
        "qwen25-isometry-noise-off-v1"
    ):
        raise ValueError("unsupported method profile")
    arrays = _source(source_root, plan["inventory"])
    seed = secret["seed"]
    logical, factors = coordinates(c, seed)
    permutation = np.asarray(secret["permutation"])
    inverse = np.argsort(permutation)
    if not np.array_equal(np.sort(permutation), np.arange(c["vocabulary_size"])):
        raise ValueError("invalid token permutation")
    descriptors = []
    cached_layer = None
    maps = None
    for node in plan["nodes"]:
        output = node["output"]
        a, dtype = arrays[node["source"]]
        norm = None
        if "norm" in node:
            norm = _convert(*arrays[node["norm"]])
        rows = np.arange(a.shape[0])
        row_scale = np.ones(len(rows))
        cols = None if a.ndim == 1 else np.arange(a.shape[1])
        col_scale = None if cols is None else np.ones(len(cols))
        if output in {"embedding.weight", "head.weight"}:
            expected_op = {
                "embedding.weight": "embedding-expand-permute-v1",
                "head.weight": "head-fold-expand-permute-v1",
            }[output]
            rows = inverse
            row_scale = np.ones(len(rows))
            cols, col_scale = logical, factors
        else:
            _, layer_text, role, kind = output.split(".")
            layer = int(layer_text)
            expected_op = {
                "q": "attention-input-fold-v1",
                "k": "attention-input-fold-v1",
                "v": "attention-input-fold-v1",
                "o": "attention-output-expand-v1",
                "gate": "ffn-gate-fold-permute-v1",
                "up": "ffn-up-fold-scale-permute-v1",
                "down": "ffn-down-unscale-expand-v1",
            }[role]
            if kind == "bias":
                expected_op = "attention-affine-bias-v1"
            if cached_layer != layer:
                maps = layer_maps(c, seed, layer)
                cached_layer = layer
            if role in {"q", "k", "v"}:
                rows = maps["q_rows" if role == "q" else "kv_rows"]
                row_scale = maps[role + "_scale"]
                if kind == "weight":
                    cols, col_scale = logical, factors
            elif role == "o":
                rows, row_scale = logical, factors
                cols, col_scale = maps["q_rows"], maps["o_scale"]
            elif role in {"gate", "up"}:
                rows = maps["f_order"]
                row_scale = np.ones(len(rows)) if role == "gate" else maps["f_scale"]
                cols, col_scale = logical, factors
            elif role == "down":
                rows, row_scale = logical, factors
                cols, col_scale = maps["f_order"], 1 / maps["f_scale"]
            else:
                raise ValueError("unsupported output role")
        if node["operation"] != expected_op:
            raise ValueError("unsupported graph operation")
        if norm is not None:
            col_scale = col_scale * norm[cols]
        digest = hashlib.sha256()
        name = node["file"]
        if Path(name).name != name or not name.endswith(".tensor"):
            raise ValueError("invalid payload basename")
        fd = os.open(scratch / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as file:
            for start in range(0, len(rows), 64):
                stop = min(start + 64, len(rows))
                if cols is None:
                    values = _convert(a[rows[start:stop]], dtype)
                    values = values * row_scale[start:stop]
                else:
                    values = _convert(a[np.ix_(rows[start:stop], cols)], dtype)
                    values *= row_scale[start:stop, None]
                    values *= col_scale[None, :]
                with np.errstate(over="ignore"):
                    payload = values.astype("<f4")
                if not np.isfinite(payload).all():
                    raise ValueError("nonfinite transformed tensor")
                data = payload.tobytes(order="C")
                file.write(data)
                digest.update(data)
            file.flush()
            os.fsync(file.fileno())
        descriptors.append(
            {
                "name": output,
                "shard": node["shard"],
                "dtype": "F32",
                "shape": node["shape"],
                "payload_sha256": digest.hexdigest(),
            }
        )
    return descriptors


def main():
    parser = argparse.ArgumentParser()
    for flag in ["source-root", "scratch-root", "secret-root"]:
        parser.add_argument("--" + flag, required=True, type=Path)
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read((8 << 20) + 1)
        if len(raw) > 8 << 20:
            raise ValueError("control exceeds bounds")
        plan = json.loads(raw)
        secret = json.loads((args.secret_root / "client-secret.json").read_bytes())
        descriptors = transform(plan, secret, args.source_root, args.scratch_root)
        sys.stdout.write(json.dumps(descriptors, separators=(",", ":")))
    except (ValueError, OSError, KeyError, TypeError, OverflowError):
        # Never echo paths, original tensor names, inputs or private material.
        sys.stderr.write("dense worker rejected input\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
