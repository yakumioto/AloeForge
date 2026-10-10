import copy
import io
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

from aloeforge_runtime.reference_worker import (
    MAX_FRAME,
    MAX_OPEN,
    read_frame,
    serve,
    validate_open,
    write_frame,
)
from tests.reference import test_dense_runtime as fixtures
from tests.reference.dense_baseline import greedy
from tests.reference.test_dense_runtime import REPO, compile_fixture

dense_cli = fixtures.dense_cli


@pytest.fixture
def opened(dense_cli, tmp_path):
    _, _, path, identity, _, _ = compile_fixture(dense_cli, tmp_path)
    contract = json.loads((path / "manifest.json").read_bytes())["dense_contract"]
    return {
        "schema_version": 1,
        "operation": "open",
        "artifact_root": str(path),
        "runtime": {
            "protocol_version": 1,
            "profile": "p6-local-dense-reference",
            "model_id": "local-reference",
            "artifact_id": identity["artifact_id"],
            "dense_contract": contract,
            "supported_features": ["greedy", "sse", "stop_token_ids"],
            "max_prompt_tokens": 32,
            "max_output_tokens": 16,
            "context_tokens": 64,
        },
        "request": {
            "protocol_version": 1,
            "model_id": "local-reference",
            "artifact_id": identity["artifact_id"],
            "binding_id": identity["binding_id"],
            "request_id": "a" * 32,
            "prompt_token_ids": [0, 1],
            "stop_token_ids": [],
            "sampling": {"temperature": 0, "max_tokens": 3},
            "stream": True,
        },
    }


def framed(value):
    raw = json.dumps(value, separators=(",", ":")).encode()
    return struct.pack(">I", len(raw)) + raw


def test_reference_worker_pull_cache_and_no_prefetch(opened):
    class Fake:
        contract = opened["runtime"]["dense_contract"]
        calls = []

        def forward(self, ids, cache):
            self.calls.append((ids, cache))
            logits = np.zeros((1, self.contract["vocabulary_size"]))
            logits[0, 3] = 1
            return logits, len(self.calls)

    fake = Fake()
    request = opened["request"]
    request["stop_token_ids"] = [3]
    pulls = [
        {
            "schema_version": 1,
            "operation": "pull",
            "request_id": "a" * 32,
            "sequence": sequence,
        }
        for sequence in [1, 2]
    ]
    output = io.BytesIO()
    serve(io.BytesIO(framed(opened)), output, lambda *a, **k: fake)
    assert not fake.calls  # Ready/EOF does not run prefill.
    output = io.BytesIO()
    serve(
        io.BytesIO(framed(opened) + b"".join(framed(x) for x in pulls)),
        output,
        lambda *a, **k: fake,
    )
    assert fake.calls == [([0, 1], None)]  # Physical stop has no extra decode.
    output.seek(0)
    assert read_frame(output)["event"] == "ready"
    assert read_frame(output)["physical_token_id"] == 3
    assert read_frame(output)["event"] == "done"
    assert read_frame(output) is None
    fake.calls.clear()
    request["stop_token_ids"] = []
    # Choose a token outside public special roles to test actual cached decode.
    fake.contract = copy.deepcopy(fake.contract)
    fake.contract["mapped_special_token_ids"]["eos"] = [10]
    opened["runtime"]["dense_contract"] = fake.contract
    pulls.append({**pulls[-1], "sequence": 3})
    serve(
        io.BytesIO(framed(opened) + b"".join(framed(x) for x in pulls)),
        io.BytesIO(),
        lambda *a, **k: fake,
    )
    assert fake.calls == [([0, 1], None), ([3], 1), ([3], 2)]


def test_reference_control_fields_and_types_fail_closed(opened):
    for mutate in [
        lambda x: x.update(artifact_root="relative"),
        lambda x: x.update(secret="private"),
        lambda x: x["runtime"].update(profile="production"),
        lambda x: x["runtime"].update(token_binding={}),
        lambda x: x["runtime"].update(context_tokens=65),
        lambda x: x["request"].update(prompt_token_ids=[True]),
        lambda x: x["request"].update(stop_token_ids=[1, 1]),
        lambda x: x["request"].update(prompt_token_ids=[0] * 32),
        lambda x: x["request"].update(binding_id="0" * 64),
        lambda x: x["request"]["sampling"].update(temperature=False),
        lambda x: x["request"]["sampling"].update(max_tokens=True),
    ]:
        bad = copy.deepcopy(opened)
        mutate(bad)
        if len(bad["request"]["prompt_token_ids"]) == 32:
            bad["request"]["sampling"]["max_tokens"] = 33
        with pytest.raises(ValueError):
            validate_open(bad)


@pytest.mark.parametrize(
    "data",
    [
        b"\0",
        struct.pack(">I", MAX_FRAME + 1),
        struct.pack(">I", 0),
        struct.pack(">I", 4) + b"{",
        framed({"x": 1}).replace(b'{"x":1}', b'{"x":\xff}'),
        struct.pack(">I", 13) + b'{"x":1,"x":2}',
    ],
)
def test_frame_corruption(data):
    with pytest.raises((ValueError, UnicodeError)):
        read_frame(io.BytesIO(data))


def test_frame_bound_and_short_reads():
    class Slow(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 1))

    assert read_frame(Slow(framed({"x": 1}))) == {"x": 1}
    with pytest.raises(ValueError):
        write_frame(io.BytesIO(), {"x": "z" * MAX_FRAME})
    with pytest.raises(ValueError):
        read_frame(io.BytesIO(struct.pack(">I", MAX_OPEN + 1)), MAX_OPEN)


def test_real_reference_process_verified_tls_and_independent_oracle(
    dense_cli, tmp_path
):
    cases = []
    # Includes wide FP64 GQA without bypassing the independent FP32 mock guard.
    for index, options in enumerate(
        [
            {"tied": False, "kv": 1, "dtype": "F32"},
            {"tied": True, "kv": 2, "dtype": "BF16"},
            {"tied": False, "kv": 2, "heads": 4, "dtype": "F16"},
            {"tied": False, "kv": 1, "head_dim": 64, "dtype": "F32"},
        ]
    ):
        base = tmp_path / str(index)
        base.mkdir(mode=0o700)
        c, tensors, path, identity, _, _ = compile_fixture(dense_cli, base, **options)
        # Select an independent source-model workload with >=2 decode steps and
        # a normal first token, so text-stop and real cache tests are substantive.
        for trial in range(256):
            prompt = [trial % c["vocab_size"], trial // c["vocab_size"], 7]
            if prompt[1] >= c["vocab_size"]:
                prompt[1] %= c["vocab_size"]
            expected = greedy(c, tensors, prompt, 5)
            if len(expected) >= 3 and expected[0] not in {0, 2, 10}:
                break
        else:
            pytest.fail("synthetic workload cannot exercise decode/text stop")
        cases.append(
            {
                "Root": str(path),
                "Secret": str(base / "secret"),
                "Python": sys.executable,
                "ArtifactID": identity["artifact_id"],
                "Tokenizer": identity["tokenizer_sha256"],
                "Prompt": prompt,
                "Expected": expected,
                "MaxTokens": 5,
            }
        )
        shutil.rmtree(base / "source")
    specification = tmp_path / "trusted-test-spec.json"
    specification.write_text(json.dumps(cases))
    specification.chmod(0o600)
    result = subprocess.run(
        [
            "go",
            "test",
            "-race",
            "-count=1",
            "./internal/referencebackend",
            "-run",
            "^TestLocalReferenceIntegration$",
        ],
        cwd=REPO,
        env={**os.environ, "ALOE_REFERENCE_TEST_SPEC": str(specification)},
        capture_output=True,
        timeout=120,
    )
    # Go asserts scalar-oracle equality; stdout contains only test status.
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()


def test_worker_failure_is_static_and_source_free():
    bad = subprocess.run(
        [sys.executable, "-I", "-m", "aloeforge_runtime.reference_worker"],
        input=framed({"PRIVATE_SOURCE_AND_SECRET": "must not escape"}),
        capture_output=True,
        timeout=10,
    )
    assert bad.returncode == 1 and not bad.stderr
    message = read_frame(io.BytesIO(bad.stdout))
    assert message == {
        "schema_version": 1,
        "event": "error",
        "error": "reference_error",
    }
