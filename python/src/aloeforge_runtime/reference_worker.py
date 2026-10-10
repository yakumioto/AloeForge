"""Trusted-local, pull-only FP64 diagnostic reference process. No text API.

Only public publication paths/identities and physical IDs cross this pipe.
One process owns one cache; the Go parent kills/reaps it on close/cancellation.
This is independent of the FP32 mock and vLLM candidate, with no torch dependency.
"""

import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

from aloeforge_reference.dense_runtime import Runtime, _json, validate_contract

PROFILE = "p6-local-dense-reference"
MAX_OPEN = 64 << 10
MAX_FRAME = 4096


def _exact(reader, count):
    parts = bytearray()
    while len(parts) < count:
        part = reader.read(count - len(parts))
        if not part:
            raise ValueError("truncated frame")
        parts.extend(part)
    return bytes(parts)


def read_frame(reader, limit=MAX_FRAME):
    prefix = reader.read(1)
    if not prefix:
        return None
    size = struct.unpack(">I", prefix + _exact(reader, 3))[0]
    if not 1 <= size <= limit:
        raise ValueError("frame limit")
    return _json(_exact(reader, size))


def write_frame(writer, message):
    # Explicit encoding keeps stdout exclusively a framed control channel.
    import json

    raw = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()
    if not 1 <= len(raw) <= MAX_FRAME:
        raise ValueError("frame limit")
    writer.write(struct.pack(">I", len(raw)) + raw)
    writer.flush()


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError("unsupported fields")
    if any(v is None for v in value.values()):
        raise ValueError("null fields")


def _integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid integer")


def _ids(value, vocab, low, high):
    if not isinstance(value, list) or not low <= len(value) <= high:
        raise ValueError("invalid physical IDs")
    for token in value:
        _integer(token, 0, vocab - 1)


def validate_open(message):
    _fields(message, "schema_version operation artifact_root runtime request")
    if type(message["schema_version"]) is not int or (
        message["schema_version"] != 1 or message["operation"] != "open"
    ):
        raise ValueError("unsupported IPC")
    root = message["artifact_root"]
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("absolute publication required")
    cap, req = message["runtime"], message["request"]
    _fields(
        cap,
        "protocol_version profile model_id artifact_id dense_contract "
        "supported_features max_prompt_tokens max_output_tokens context_tokens",
    )
    if type(cap["protocol_version"]) is not int or (
        cap["protocol_version"] != 1 or cap["profile"] != PROFILE
    ):
        raise ValueError("unsupported runtime")
    if not isinstance(cap["model_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,64}", cap["model_id"]
    ):
        raise ValueError("invalid model")
    if not isinstance(cap["artifact_id"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", cap["artifact_id"]
    ):
        raise ValueError("invalid artifact identity")
    c = cap["dense_contract"]
    validate_contract(c)
    if cap["supported_features"] != ["greedy", "sse", "stop_token_ids"]:
        raise ValueError("unsupported features")
    _integer(cap["context_tokens"], 2, min(8192, c["max_position_embeddings"]))
    for field in ["max_prompt_tokens", "max_output_tokens"]:
        _integer(cap[field], 1, min(4096, cap["context_tokens"]))
    _fields(
        req,
        "protocol_version model_id artifact_id binding_id request_id "
        "prompt_token_ids stop_token_ids sampling stream",
    )
    if type(req["protocol_version"]) is not int or any(
        req[k] != cap[k] for k in ["protocol_version", "model_id", "artifact_id"]
    ):
        raise ValueError("binding mismatch")
    if req["binding_id"] != c["binding_id"] or req["stream"] is not True:
        raise ValueError("binding mismatch")
    if not isinstance(req["request_id"], str) or not re.fullmatch(
        r"[0-9a-f]{32}", req["request_id"]
    ):
        raise ValueError("invalid request identity")
    s = req["sampling"]
    _fields(s, "temperature max_tokens")
    if type(s["temperature"]) not in {int, float} or s["temperature"] != 0:
        raise ValueError("greedy required")
    _integer(s["max_tokens"], 1, cap["max_output_tokens"])
    _ids(req["prompt_token_ids"], c["vocabulary_size"], 1, cap["max_prompt_tokens"])
    _ids(req["stop_token_ids"], c["vocabulary_size"], 0, 16)
    if len(set(req["stop_token_ids"])) != len(req["stop_token_ids"]) or (
        len(req["prompt_token_ids"]) + s["max_tokens"] > cap["context_tokens"]
    ):
        raise ValueError("invalid request bound")
    return root, cap, req


def serve(reader, writer, loader=Runtime.load):
    root, cap, req = validate_open(read_frame(reader, MAX_OPEN))
    c = cap["dense_contract"]
    runtime = loader(
        root,
        artifact_id=cap["artifact_id"],
        binding_id=c["binding_id"],
        tokenizer_sha256=c["tokenizer_sha256"],
    )
    if runtime.contract != c:
        raise ValueError("publication contract mismatch")
    common = {"schema_version": 1, "request_id": req["request_id"]}
    write_frame(
        writer,
        {
            **common,
            "event": "ready",
            "artifact_id": cap["artifact_id"],
            "binding_id": c["binding_id"],
            "tokenizer_sha256": c["tokenizer_sha256"],
        },
    )
    cache, previous, sequence, terminal = None, None, 0, False
    stops = set(req["stop_token_ids"]) | set(c["mapped_special_token_ids"]["eos"])
    while (pull := read_frame(reader)) is not None:
        _fields(pull, "schema_version operation request_id sequence")
        _integer(pull["schema_version"], 1, 1)
        _integer(pull["sequence"], 1, req["sampling"]["max_tokens"] + 1)
        if (
            pull["operation"] != "pull"
            or pull["request_id"] != req["request_id"]
            or pull["sequence"] != sequence + 1
        ):
            raise ValueError("invalid pull")
        sequence += 1
        record = {**common, "sequence": sequence}
        if terminal:
            write_frame(writer, {**record, "event": "done"})
            return
        ids = req["prompt_token_ids"] if previous is None else [previous]
        logits, cache = runtime.forward(ids, cache)
        previous = int(np.argmax(logits[-1]))
        terminal = previous in stops or sequence == req["sampling"]["max_tokens"]
        write_frame(writer, {**record, "event": "token", "physical_token_id": previous})


def main():
    # No traceback/private path/backend exception can reach the process parent.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 2)
    try:
        serve(sys.stdin.buffer, sys.stdout.buffer)
    except Exception:
        try:
            write_frame(
                sys.stdout.buffer,
                {"schema_version": 1, "event": "error", "error": "reference_error"},
            )
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
