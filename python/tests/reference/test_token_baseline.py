import copy
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path

import pytest

from aloeforge_reference.token_baseline import (
    binding_commitment,
    inverse,
    logits,
    map_special_ids,
    permute_rows,
    restore_logits,
    validate_secret,
)

REPO = Path(__file__).resolve().parents[3]
EMBEDDING = [[1.0, 2.0], [3.0, -1.0], [0.5, 4.0], [-2.0, 0.25]]
HEAD = [[2.0, -1.0], [0.0, 3.0], [-4.0, 0.5], [1.0, 1.0]]
SPECIAL = {"bos": [0], "eos": [1, 3], "pad": [0]}


def golden_material():
    return {
        "context": {
            "source": {"inventory_sha256": "0" * 64, "content_sha256": "1" * 64},
            "tokenizer_sha256": "2" * 64,
            "vocabulary_size": 4,
            "hidden_size": 2,
            "tied_embeddings": False,
            "special_token_ids": copy.deepcopy(SPECIAL),
            "method_revision": "token-permutation-v1",
            "rng_algorithm": "crypto-rand-fisher-yates",
            "rng_revision": "1",
        },
        "nonce": "0" * 64,
        "permutation": [2, 0, 3, 1],
    }


def test_independent_golden_and_coordinate_direction():
    material = golden_material()
    assert binding_commitment(material) == (
        "f7cd207dc66f23b4e7e160a9327b974f90b2f962c2a6dc6ba2d88ebfc6432d0c"
    )
    permutation = material["permutation"]
    assert inverse(permutation) == [1, 3, 0, 2]
    assert map_special_ids(SPECIAL, permutation) == {
        "bos": [2],
        "eos": [0, 1],
        "pad": [2],
    }
    physical = logits(
        permute_rows(EMBEDDING, permutation),
        permute_rows(HEAD, permutation),
        permutation[1],
    )
    assert physical == [-3, 2, 7, -12.5]
    assert logits(EMBEDDING, HEAD, 1) == [7, -3, -12.5, 2]
    assert physical != logits(EMBEDDING, HEAD, 1)
    assert restore_logits(physical, permutation) == logits(EMBEDDING, HEAD, 1)
    assert logits(EMBEDDING, None, 1) == [1, 10, -2.5, -6.25]


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [0],
        [0, 0],
        [-1, 0],
        [0, 2],
        [False, 1],
        [0.0, 1],
        [0, 1, 3],
        list(range(65537)),
    ],
)
def test_oracle_rejects_corrupt_permutations(bad):
    with pytest.raises(ValueError):
        inverse(bad)


@pytest.mark.parametrize("bad", [[1], [1, 2, math.nan, 4], [1, math.inf, 3, 4]])
def test_oracle_rejects_invalid_logits(bad):
    with pytest.raises(ValueError):
        restore_logits(bad, [2, 0, 3, 1])


@pytest.mark.parametrize(
    "kind",
    ["shape", "nonfinite", "token", "head-hidden", "special-role", "special-duplicate"],
)
def test_oracle_rejects_unsupported_cases(kind):
    with pytest.raises(ValueError):
        if kind == "shape":
            permute_rows([[1], [2, 3]], [1, 0])
        elif kind == "nonfinite":
            permute_rows([[math.nan], [2]], [1, 0])
        elif kind == "token":
            logits(EMBEDDING, HEAD, -1)
        elif kind == "head-hidden":
            logits(EMBEDDING, [[1]] * 4, 0)
        elif kind == "special-role":
            map_special_ids({"bos": [0], "eos": [1], "unknown": []}, [1, 0])
        else:
            map_special_ids({"bos": [], "eos": [1, 1], "pad": []}, [1, 0])


@pytest.mark.parametrize(
    "kind",
    [
        "version",
        "commitment",
        "unknown",
        "nonce",
        "method",
        "rng",
        "tokenizer",
        "dimensions",
    ],
)
def test_independent_secret_rejection(kind):
    material = golden_material()
    secret = {
        "schema_version": 1,
        "binding_id": binding_commitment(material),
        "material": material,
    }
    if kind == "version":
        secret["schema_version"] = 2
    elif kind == "commitment":
        secret["binding_id"] = "3" * 64
    elif kind == "unknown":
        secret["extra"] = "unimplemented"
    elif kind == "nonce":
        material["nonce"] = "invalid"
    elif kind == "method":
        material["context"]["method_revision"] = "full-v2"
    elif kind == "rng":
        material["context"]["rng_revision"] = "2"
    elif kind == "tokenizer":
        material["context"]["tokenizer_sha256"] = "invalid"
    else:
        material["context"]["hidden_size"] = 257
    with pytest.raises(ValueError):
        validate_secret(secret)


def write_json(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


def fixture(tmp_path, tied):
    source, output, secret = (
        tmp_path / name for name in ("source", "output", "secret")
    )
    for directory in (source, output, secret):
        directory.mkdir(mode=0o700)
    write_json(
        source / "config.json",
        {
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "aloeforge_synthetic": True,
        },
    )
    write_json(
        source / "token-baseline.json",
        {
            "schema_version": 1,
            "kind": "synthetic-embedding-head-v1",
            "vocabulary_size": 4,
            "hidden_size": 2,
            "tied_embeddings": tied,
            "special_token_ids": SPECIAL,
        },
    )
    write_json(
        source / "tokenizer.json",
        {
            "schema_version": 1,
            "kind": "synthetic-vocabulary-v1",
            "tokens": ["<bos>", "<eos>", "synthetic", "<eos2>"],
        },
    )
    tensors = {"embedding.weight": EMBEDDING}
    if not tied:
        tensors["lm_head.weight"] = HEAD
    header, payload = {}, bytearray()
    for name, rows in tensors.items():
        start = len(payload)
        for row in rows:
            payload.extend(struct.pack("<2f", *row))
        header[name] = {
            "dtype": "F32",
            "shape": [4, 2],
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    (source / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    return source, output, secret


def read_matrices(path):
    # Independent fixture parser, not an alternate production verification gate.
    # Tests intentionally bound file size before reading this tiny matrix payload.
    assert path.stat().st_size < 16384
    data = path.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    assert 0 < length < 8192
    header = json.loads(data[8 : 8 + length])
    result = {}
    for name, tensor in header.items():
        assert tensor["dtype"] == "F32" and tensor["shape"] == [4, 2]
        start, end = tensor["data_offsets"]
        assert end - start == 32
        result[name] = [
            list(row)
            for row in struct.iter_unpack(
                "<2f", data[8 + length + start : 8 + length + end]
            )
        ]
    return result


def invoke(*args):
    # Trusted repository Go entrypoint; paths are literal argv with no shell.
    return subprocess.run(
        ["go", "run", "./cmd/aloeforge", *args],
        cwd=REPO,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("tied", [False, True], ids=["untied", "tied"])
def test_real_go_cli_with_independent_python_oracle(tmp_path, tied):
    source, output, secret_root = fixture(tmp_path, tied)
    args = (
        "compile-token",
        "--model-root",
        str(source),
        "--output-parent",
        str(output),
        "--secret-root",
        str(secret_root),
        "--stage",
        ".stage-p2",
        "--target",
        "visible",
    )
    compiled = invoke(*args)
    assert compiled.returncode == 0, compiled.stderr
    summary = json.loads(compiled.stdout)
    assert summary["kind"] == "p2-token-permutation-diagnostic-only"
    secret_path = secret_root / "client-secret.json"
    secret_bytes = secret_path.read_bytes()
    assert secret_path.stat().st_mode & 0o777 == 0o600
    secret = json.loads(secret_bytes)
    validate_secret(secret)
    material = secret["material"]
    permutation = material["permutation"]
    assert summary["binding_id"] == binding_commitment(material)
    assert (
        summary["tokenizer_sha256"]
        == hashlib.sha256((source / "tokenizer.json").read_bytes()).hexdigest()
    )
    matrices = read_matrices(output / "visible" / "model.safetensors")
    assert matrices["embedding.weight"] == permute_rows(EMBEDDING, permutation)
    if tied:
        assert set(matrices) == {"embedding.weight"}
    else:
        assert matrices["lm_head.weight"] == permute_rows(HEAD, permutation)
    for original in range(4):
        expected = logits(EMBEDDING, None if tied else HEAD, original)
        physical = logits(
            matrices["embedding.weight"],
            None if tied else matrices["lm_head.weight"],
            permutation[original],
        )
        assert restore_logits(physical, permutation) == expected
    manifest_path = output / "visible" / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert (
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() == summary["artifact_id"]
    )
    assert (
        hashlib.sha256(
            (manifest_path.parent / "model.safetensors").read_bytes()
        ).hexdigest()
        == manifest["files"][0]["sha256"]
    )
    descriptor_bytes = json.dumps(manifest["outputs"], separators=(",", ":")).encode()
    assert hashlib.sha256(descriptor_bytes).hexdigest() == manifest["descriptor_sha256"]
    assert manifest["token_binding"]["mapped_special_token_ids"] == map_special_ids(
        SPECIAL, permutation
    )
    assert {path.name for path in manifest_path.parent.iterdir()} == {
        "manifest.json",
        "model.safetensors",
    }
    for key in ("permutation", "nonce", "special_token_ids", "source"):
        assert f'"{key}":' not in manifest_path.read_text()
    for private in (str(source), str(secret_root), "<bos>", "<eos>"):
        assert private not in compiled.stdout + compiled.stderr
    check = invoke(
        "verify-token",
        "--artifact-root",
        str(output / "visible"),
        "--secret-root",
        str(secret_root),
        "--artifact-id",
        summary["artifact_id"],
        "--tokenizer-sha256",
        summary["tokenizer_sha256"],
    )
    assert check.returncode == 0, check.stderr
    resumed = invoke(*args, "--resume")
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads(resumed.stdout) == summary
    assert secret_path.read_bytes() == secret_bytes
    bad = invoke(
        "verify-token",
        "--artifact-root",
        str(output / "visible"),
        "--secret-root",
        str(secret_root),
        "--artifact-id",
        "0" * 64,
        "--tokenizer-sha256",
        summary["tokenizer_sha256"],
    )
    assert bad.returncode != 0 and bad.stdout == ""
    assert "client artifact identity mismatch" in bad.stderr
