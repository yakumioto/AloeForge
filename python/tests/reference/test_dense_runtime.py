import copy
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aloeforge_reference.dense_runtime import Cache, Runtime, validate_contract
from tests.reference.dense_baseline import forward, greedy

REPO = Path(__file__).resolve().parents[3]
LOGIT_ATOL = LOGIT_RTOL = 3e-4
NLL_GATE = 4e-4


@pytest.fixture(scope="session")
def dense_cli(tmp_path_factory):
    path = tmp_path_factory.mktemp("p5-bin") / "aloeforge"
    subprocess.run(
        ["go", "build", "-o", str(path), "./cmd/aloeforge"], cwd=REPO, check=True
    )
    return path


def source_fixture(
    root,
    *,
    seed=6,
    tied=False,
    kv=1,
    heads=2,
    dtype="F32",
    layers=2,
    vocab=11,
    intermediate=10,
    zero_head=False,
):
    root.mkdir(mode=0o700)
    hidden = heads * 4
    c = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "hidden_act": "silu",
        "tie_word_embeddings": tied,
        "use_sliding_window": False,
        "attention_dropout": 0,
        "bos_token_id": 0,
        "eos_token_id": [2, 10],
        "pad_token_id": 0,
    }
    rng = np.random.default_rng(seed)
    tensors = {}
    tensors["model.embed_tokens.weight"] = rng.normal(0, 0.3, (vocab, hidden))
    tensors["model.norm.weight"] = rng.uniform(0.2, 1.8, hidden)
    if not tied:
        tensors["lm_head.weight"] = rng.normal(0, 0.3, (vocab, hidden))
        if zero_head:
            tensors["lm_head.weight"] *= 0
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        for norm in ["input_layernorm", "post_attention_layernorm"]:
            tensors[prefix + norm + ".weight"] = rng.uniform(0.2, 1.8, hidden)
        for role, rows, cols in [
            ("self_attn.q_proj", hidden, hidden),
            ("self_attn.k_proj", kv * 4, hidden),
            ("self_attn.v_proj", kv * 4, hidden),
            ("self_attn.o_proj", hidden, hidden),
            ("mlp.gate_proj", intermediate, hidden),
            ("mlp.up_proj", intermediate, hidden),
            ("mlp.down_proj", hidden, intermediate),
        ]:
            tensors[prefix + role + ".weight"] = rng.normal(0, 0.15, (rows, cols))
            if role in {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"}:
                tensors[prefix + role + ".bias"] = rng.normal(0, 0.1, rows)
    write_source(root, c, tensors, dtype)
    return c, tensors


def write_source(root, config, tensors, dtype="F32"):
    header, payload = {}, bytearray()
    for name, a in sorted(tensors.items()):
        a = np.asarray(a, dtype="<f4")
        if dtype == "F16":
            stored = a.astype("<f2")
            decoded = stored.astype(np.float32)
        elif dtype == "BF16":
            bits = a.view(np.uint32)
            stored = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")
            decoded = (stored.astype(np.uint32) << 16).view(np.float32)
        else:
            stored, decoded = a, a
        tensors[name] = decoded
        data = stored.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(a.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + payload
    )
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "model": {"type": "BPE", "vocab": {"a": 0}}})
    )


def compile_fixture(cli, base, **fixture_options):
    source, output, secret = (base / name for name in ["source", "output", "secret"])
    config, tensors = source_fixture(source, **fixture_options)
    output.mkdir(mode=0o700)
    secret.mkdir(mode=0o700)
    command = [
        str(cli),
        "compile-dense",
        "--model-root",
        str(source),
        "--output-parent",
        str(output),
        "--secret-root",
        str(secret),
        "--stage",
        ".stage-test",
        "--target",
        "published",
        "--python",
        sys.executable,
        "--expansion",
        "4",
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
    identity = json.loads(process.stdout)
    private = json.loads((secret / "client-secret.json").read_bytes())
    return config, tensors, output / "published", identity, private, command


def load(path, identity, **options):
    return Runtime.load(
        path,
        artifact_id=identity["artifact_id"],
        binding_id=identity["binding_id"],
        tokenizer_sha256=identity["tokenizer_sha256"],
        **options,
    )


@pytest.mark.parametrize(
    "tied,kv,heads,dtype,compute",
    [
        (False, 1, 2, "F32", np.float64),
        (True, 2, 2, "F32", np.float32),
        (True, 1, 2, "BF16", np.float64),
        (False, 2, 2, "F16", np.float32),
        (False, 2, 4, "F32", np.float32),
        (True, 2, 4, "BF16", np.float64),
    ],
)
def test_full_graph_logits_generation_cache_and_isolated_load(
    dense_cli, tmp_path, tied, kv, heads, dtype, compute
):
    c, tensors, path, identity, private, command = compile_fixture(
        dense_cli, tmp_path, tied=tied, kv=kv, heads=heads, dtype=dtype
    )
    runtime = load(path, identity, compute_dtype=compute)
    permutation = private["permutation"]
    ids = [0, 4, 7, 1, 9]
    mapped = [permutation[i] for i in ids]
    baseline = forward(c, tensors, ids)
    physical, cache = runtime.forward(mapped)
    restored = physical[:, permutation]
    np.testing.assert_allclose(restored, baseline, atol=LOGIT_ATOL, rtol=LOGIT_RTOL)
    # Independent full-prefix baseline and runtime split prefill/decode.
    parts, split_cache = [], None
    for chunk in [mapped[:2], mapped[2:4], mapped[4:]]:
        part, split_cache = runtime.forward(chunk, split_cache)
        parts.append(part)
    np.testing.assert_allclose(np.concatenate(parts), physical, atol=2e-6, rtol=2e-6)
    assert cache.length == split_cache.length == len(ids)
    generated = runtime.greedy(mapped, 5)
    inverse = np.argsort(permutation)
    assert [int(inverse[i]) for i in generated] == greedy(c, tensors, ids, 5)
    labels = ids[1:]

    def nll(logits):
        x = logits[:-1]
        peak = np.max(x, axis=-1)
        return float(
            np.mean(
                peak
                + np.log(np.exp(x - peak[:, None]).sum(-1))
                - x[np.arange(len(labels)), labels]
            )
        )

    assert abs(nll(restored) - nll(baseline)) <= NLL_GATE
    manifest = (path / "manifest.json").read_bytes()
    assert b"seed" not in manifest and b"permutation" not in manifest
    assert b"source_inventory" not in manifest and b"original" not in manifest
    assert not any("norm" in name for name in runtime.tensors)
    assert runtime.tensors["head.weight"].shape == (11, c["hidden_size"] + 4)
    for role, original in {"bos": [0], "eos": [2, 10], "pad": [0]}.items():
        assert runtime.contract["mapped_special_token_ids"][role] == sorted(
            permutation[i] for i in original
        )
    resume = subprocess.run([*command, "--resume"], capture_output=True, text=True)
    assert resume.returncode == 0, resume.stderr
    assert json.loads(resume.stdout) == identity
    # No source/secret path remains available under the paths supplied to compiler.
    (tmp_path / "source").rename(tmp_path / "unavailable-source")
    (tmp_path / "secret").rename(tmp_path / "unavailable-secret")
    script = (
        "import json,sys; from aloeforge_reference.dense_runtime import Runtime; "
        "i=json.loads(sys.argv[2]); r=Runtime.load(sys.argv[1],"
        "artifact_id=i['artifact_id'],"
        "binding_id=i['binding_id'],tokenizer_sha256=i['tokenizer_sha256']); "
        "assert 'aloeforge_worker.dense' not in sys.modules; "
        "assert r.forward(json.loads(sys.argv[3]))[0].shape==(5,11)"
    )
    isolated = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(path),
            json.dumps(identity),
            json.dumps(mapped),
        ],
        capture_output=True,
    )
    assert isolated.returncode == 0, isolated.stderr


@pytest.fixture
def compiled(dense_cli, tmp_path):
    return compile_fixture(dense_cli, tmp_path)


@pytest.mark.parametrize(
    "kind",
    [
        "identity",
        "binding",
        "tokenizer",
        "payload",
        "header",
        "extra",
        "symlink",
        "profile",
        "shape",
        "missing",
        "revision",
        "dtype",
        "secret-field",
        "descriptor",
        "duplicate-json",
        "budget",
    ],
)
def test_loader_rejects_corrupt_artifacts(compiled, kind):
    _, _, path, identity, _, _ = compiled
    identity = copy.deepcopy(identity)
    manifest_path = path / "manifest.json"
    document = json.loads(manifest_path.read_bytes())
    if kind in {"identity", "binding", "tokenizer"}:
        key = {
            "identity": "artifact_id",
            "binding": "binding_id",
            "tokenizer": "tokenizer_sha256",
        }[kind]
        identity[key] = "0" * 64
    elif kind in {"payload", "header"}:
        shard = path / document["files"][0]["name"]
        raw = bytearray(shard.read_bytes())
        raw[-1 if kind == "payload" else 8] ^= 1
        shard.write_bytes(raw)
    elif kind == "extra":
        (path / "client-secret.json").write_text("private")
    elif kind == "symlink":
        shard = path / document["files"][0]["name"]
        external = path.parent / "outside.safetensors"
        shard.rename(external)
        shard.symlink_to(external)
    elif kind == "missing":
        (path / document["files"][0]["name"]).unlink()
    elif kind == "duplicate-json":
        raw = manifest_path.read_bytes().replace(
            b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1
        )
        manifest_path.write_bytes(raw)
        identity["artifact_id"] = hashlib.sha256(raw).hexdigest()
    elif kind != "budget":
        if kind == "profile":
            document["dense_contract"]["profile"] = "production"
        elif kind == "revision":
            document["dense_contract"]["rope_layout"] = "interleaved"
        elif kind == "shape":
            document["outputs"][0]["shape"][1] += 1
        elif kind == "dtype":
            document["outputs"][0]["dtype"] = "BF16"
        elif kind == "descriptor":
            document["descriptor_sha256"] = "0" * 64
        elif kind == "secret-field":
            document["dense_contract"]["seed"] = "0" * 64
        raw = json.dumps(document, separators=(",", ":")).encode()
        manifest_path.write_bytes(raw)
        identity["artifact_id"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises((ValueError, OSError)):
        load(path, identity, **({"max_weight_bytes": 1} if kind == "budget" else {}))


@pytest.mark.parametrize(
    "kind",
    [
        "ids",
        "bool-id",
        "empty",
        "context",
        "cache-owner",
        "cache-shape",
        "cache-nonfinite",
        "cache-dtype",
        "allocation",
        "logit-allocation",
        "activation-allocation",
        "empty-generation",
        "negative-generation",
    ],
)
def test_runtime_request_and_cache_fail_closed(compiled, kind):
    _, _, path, identity, _, _ = compiled
    runtime = load(path, identity)
    _, cache = runtime.forward([0, 1])
    with pytest.raises(ValueError):
        if kind == "ids":
            runtime.forward([11])
        elif kind == "bool-id":
            runtime.forward([True])
        elif kind == "empty":
            runtime.forward([])
        elif kind == "context":
            runtime.forward([0] * 65)
        elif kind == "cache-owner":
            load(path, identity).forward([0], cache)
        elif kind == "cache-shape":
            runtime.forward([0], Cache(cache.owner, 3, cache.layers))
        elif kind == "cache-nonfinite":
            layers = list(cache.layers)
            bad = np.array(layers[0][0])
            bad[0, 0, 0] = np.nan
            layers[0] = (bad, layers[0][1])
            runtime.forward([0], Cache(cache.owner, cache.length, tuple(layers)))
        elif kind == "cache-dtype":
            layers = list(cache.layers)
            layers[0] = (layers[0][0].astype(np.float32), layers[0][1])
            runtime.forward([0], Cache(cache.owner, cache.length, tuple(layers)))
        elif kind == "allocation":
            load(path, identity, max_attention_bytes=1).forward([0, 1])
        elif kind == "logit-allocation":
            load(path, identity, max_logit_bytes=1).forward([0, 1])
        elif kind == "activation-allocation":
            load(path, identity, max_activation_bytes=1).forward([0, 1])
        elif kind == "empty-generation":
            runtime.greedy([], 0)
        else:
            runtime.greedy([0], -1)


@pytest.mark.parametrize(
    "kind",
    [
        "source-change",
        "tokenizer-change",
        "secret-corrupt",
        "missing-secret",
        "expansion-change",
    ],
)
def test_resume_rejects_binding_changes(compiled, kind):
    _, _, _, _, _, command = compiled
    root = Path(command[command.index("--model-root") + 1])
    secret = Path(command[command.index("--secret-root") + 1]) / "client-secret.json"
    if kind == "source-change":
        shard = root / "model.safetensors"
        raw = bytearray(shard.read_bytes())
        raw[-1] ^= 1
        shard.write_bytes(raw)
    elif kind == "tokenizer-change":
        (root / "tokenizer.json").write_text('{"other":"tokenizer"}')
    elif kind == "secret-corrupt":
        doc = json.loads(secret.read_bytes())
        doc["permutation"][0] = doc["permutation"][1]
        secret.write_text(json.dumps(doc))
    elif kind == "missing-secret":
        secret.unlink()
    else:
        command[-1] = "8"
    result = subprocess.run([*command, "--resume"], capture_output=True, text=True)
    assert result.returncode == 1 and not result.stdout
    assert str(root) not in result.stderr and "permutation" not in result.stderr


@pytest.mark.parametrize(
    "kind",
    [
        "architecture",
        "missing-qkv-bias",
        "unexpected",
        "shape",
        "sliding",
        "rope-scaling",
        "quantization",
        "unknown-field",
        "tied-alias",
        "nonfinite",
    ],
)
def test_compiler_rejects_unsupported_graphs(dense_cli, tmp_path, kind):
    source = tmp_path / "source"
    c, tensors = source_fixture(source)
    if kind == "architecture":
        c["model_type"], c["architectures"] = "llama", ["LlamaForCausalLM"]
    elif kind == "missing-qkv-bias":
        tensors.pop("model.layers.0.self_attn.q_proj.bias")
    elif kind == "unexpected":
        tensors["model.layers.0.self_attn.q_norm.weight"] = np.ones(4)
    elif kind == "shape":
        tensors["model.layers.0.self_attn.k_proj.weight"] = np.ones((5, 8))
    elif kind == "sliding":
        c["use_sliding_window"] = True
    elif kind == "rope-scaling":
        c["rope_scaling"] = {"rope_type": "yarn"}
    elif kind == "quantization":
        c["quantization_config"] = {}
    elif kind == "unknown-field":
        c["partial_rotary_factor"] = 0.5
    elif kind == "tied-alias":
        c["tie_word_embeddings"] = True
    else:
        tensors["model.layers.0.self_attn.q_proj.weight"][0, 0] = np.nan
    write_source(source, c, tensors)
    for name in ["output", "secret"]:
        (tmp_path / name).mkdir(mode=0o700)
    process = subprocess.run(
        [
            str(dense_cli),
            "compile-dense",
            "--model-root",
            str(source),
            "--output-parent",
            str(tmp_path / "output"),
            "--secret-root",
            str(tmp_path / "secret"),
            "--stage",
            ".stage-test",
            "--target",
            "published",
            "--python",
            sys.executable,
            "--expansion",
            "4",
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1 and not process.stdout
    assert not (tmp_path / "output" / "published").exists()
    assert str(source) not in process.stderr and "nan" not in process.stderr.lower()


def test_contract_scalar_types_and_profile(compiled):
    _, _, path, identity, _, _ = compiled
    c = load(path, identity).contract
    for key, value in [
        ("num_attention_heads", True),
        ("schema_version", True),
        ("logical_norm_epsilon", float("nan")),
        ("profile", "paper-reference"),
        ("physical_hidden_size", 13),
        ("cache_layout", "other"),
    ]:
        altered = copy.deepcopy(c)
        altered[key] = value
        with pytest.raises(ValueError):
            validate_contract(altered)


def test_streamed_block_boundaries(dense_cli, tmp_path):
    c, tensors, path, identity, private, _ = compile_fixture(
        dense_cli, tmp_path, vocab=71, intermediate=70, heads=4, kv=2, layers=3
    )
    ids = [0, 65, 70, 4]
    permutation = private["permutation"]
    physical, _ = load(path, identity).forward([permutation[i] for i in ids])
    np.testing.assert_allclose(
        physical[:, permutation],
        forward(c, tensors, ids),
        atol=LOGIT_ATOL,
        rtol=LOGIT_RTOL,
    )


def test_trusted_client_binding_verification(compiled, dense_cli):
    _, _, path, identity, _, command = compiled
    args = [
        str(dense_cli),
        "verify-dense",
        "--artifact-root",
        str(path),
        "--secret-root",
        command[command.index("--secret-root") + 1],
        "--artifact-id",
        identity["artifact_id"],
        "--tokenizer-sha256",
        identity["tokenizer_sha256"],
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0 and json.loads(result.stdout) == identity
    args[args.index("--artifact-id") + 1] = "0" * 64
    rejected = subprocess.run(args, capture_output=True, text=True)
    assert rejected.returncode == 1 and not rejected.stdout


def test_worker_failure_discards_private_diagnostics(compiled, tmp_path):
    _, _, path, _, _, command = compiled
    private_message = "PRIVATE_SOURCE_AND_SEED_MUST_NOT_ESCAPE"
    fake = tmp_path / "fake-python"
    fake.write_text("#!/bin/sh\nprintf '" + private_message + "' >&2\nprintf '[]'\n")
    fake.chmod(0o700)
    command[command.index("--python") + 1] = str(fake)
    result = subprocess.run([*command, "--resume"], capture_output=True, text=True)
    assert result.returncode == 1 and not result.stdout
    assert private_message not in result.stderr
    assert path.is_dir()


def test_greedy_ties_follow_physical_coordinate_order(dense_cli, tmp_path):
    _, _, path, identity, _, _ = compile_fixture(dense_cli, tmp_path, zero_head=True)
    runtime = load(path, identity)
    logits, _ = runtime.forward([0, 1])
    assert np.all(logits == 0)
    assert runtime.greedy([0, 1], 2)[0] == 0
