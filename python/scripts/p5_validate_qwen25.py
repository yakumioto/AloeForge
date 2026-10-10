"""Trusted external quality runner. Local pinned checkpoint and dataset only.

Source/Client Secret are confined to this controlled runner, not the independent
artifact runtime. Only scalar aggregate evidence is printed; no IDs or logits.
No checkpoint or evaluation data is automatically downloaded.
"""

import argparse
import importlib.metadata
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from aloeforge_reference.dense_runtime import Runtime, _hash, _int, _json


def dataset(raw, revision):
    if len(raw) > 8 << 20:
        raise ValueError("evaluation dataset exceeds bounds")
    doc = _json(raw)
    if not isinstance(doc, dict) or set(doc) != {
        "schema_version",
        "checkpoint_revision",
        "cases",
    }:
        raise ValueError("unsupported evaluation dataset")
    if (
        type(doc["schema_version"]) is not int
        or doc["schema_version"] != 1
        or doc["checkpoint_revision"] != revision
    ):
        raise ValueError("evaluation revision mismatch")
    cases = doc["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 64:
        raise ValueError("unsupported evaluation case count")
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"input_ids", "target_ids"}:
            raise ValueError("unsupported evaluation case")
        for ids in case.values():
            if (
                not isinstance(ids, list)
                or not 1 <= len(ids) <= 128
                or any(not _int(i) or i < 0 for i in ids)
            ):
                raise ValueError("invalid trusted evaluation IDs")
    return cases


def quality(logits, targets):
    peak = np.max(logits, axis=-1)
    losses = (
        peak
        + np.log(np.exp(logits - peak[:, None]).sum(-1))
        - logits[np.arange(len(targets)), targets]
    )
    return float(np.sum(losses)), int(np.sum(np.argmax(logits, -1) == targets))


def main():
    parser = argparse.ArgumentParser()
    for flag in [
        "model-root",
        "artifact-root",
        "secret-root",
        "identity-file",
        "dataset",
        "cli",
    ]:
        parser.add_argument("--" + flag, required=True, type=Path)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    if (
        not re.fullmatch("[0-9a-f]{40}", args.checkpoint_revision)
        or not 1 <= args.max_new_tokens <= 32
    ):
        raise ValueError("pinned checkpoint and generation bound required")
    for name, expected in {
        "numpy": "2.3.3",
        "torch": "2.8.0",
        "transformers": "4.57.1",
        "tokenizers": "0.22.1",
        "safetensors": "0.6.2",
    }.items():
        if importlib.metadata.version(name).split("+")[0] != expected:
            raise ValueError("external numerical toolchain version mismatch")
    # Imports only after admission; these dependencies are absent from CPU CI.
    import torch
    from transformers import AutoModelForCausalLM

    identity = _json(args.identity_file.read_bytes())
    if set(identity) != {"kind", "artifact_id", "binding_id", "tokenizer_sha256"}:
        raise ValueError("invalid compiler identity record")
    if not all(
        _hash(identity[k]) for k in ["artifact_id", "binding_id", "tokenizer_sha256"]
    ):
        raise ValueError("invalid compiler identities")
    verified = subprocess.run(
        [
            str(args.cli),
            "verify-dense",
            "--artifact-root",
            str(args.artifact_root),
            "--secret-root",
            str(args.secret_root),
            "--artifact-id",
            identity["artifact_id"],
            "--tokenizer-sha256",
            identity["tokenizer_sha256"],
        ],
        capture_output=True,
        check=False,
    )
    if verified.returncode != 0 or _json(verified.stdout) != identity:
        raise ValueError("trusted Client Secret/artifact verification failed")
    secret = _json((args.secret_root / "client-secret.json").read_bytes())
    p = np.asarray(secret["permutation"], dtype=np.int64)
    inv = np.argsort(p)
    runtime = Runtime.load(
        args.artifact_root,
        artifact_id=identity["artifact_id"],
        binding_id=identity["binding_id"],
        tokenizer_sha256=identity["tokenizer_sha256"],
        compute_dtype=np.float32,
    )
    cases = dataset(args.dataset.read_bytes(), args.checkpoint_revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_root,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        .eval()
        .to(args.device)
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested physical GPU unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {
        "profile": "diagnostic",
        "checkpoint_revision": args.checkpoint_revision,
        "cases": len(cases),
        "source_device": args.device,
        "physical_gpu_used": args.device == "cuda",
        "reference_device": "cpu",
        "max_logit_error": 0.0,
        "greedy_exact_cases": 0,
        "target_tokens": 0,
        "source_correct": 0,
        "reference_correct": 0,
        "source_nll_sum": 0.0,
        "reference_nll_sum": 0.0,
        "logit_atol": 3e-4,
        "logit_rtol": 3e-4,
        "nll_delta_gate": 4e-4,
    }
    with torch.inference_mode():
        for case in cases:
            prefix, targets = case["input_ids"], case["target_ids"]
            ids = prefix + targets
            if max(ids) >= len(p):
                raise ValueError("evaluation ID outside vocabulary")
            source = (
                model(torch.tensor([ids], device=args.device), use_cache=False)
                .logits[0]
                .float()
                .cpu()
                .numpy()
            )
            physical, _ = runtime.forward([int(p[i]) for i in ids])
            restored = physical[:, p]
            np.testing.assert_allclose(restored, source, atol=3e-4, rtol=3e-4)
            report["max_logit_error"] = max(
                report["max_logit_error"], float(np.max(np.abs(restored - source)))
            )
            start = len(prefix) - 1
            baseline_nll, baseline_correct = quality(source[start:-1], targets)
            ref_nll, ref_correct = quality(restored[start:-1], targets)
            report["source_nll_sum"] += baseline_nll
            report["reference_nll_sum"] += ref_nll
            report["source_correct"] += baseline_correct
            report["reference_correct"] += ref_correct
            report["target_tokens"] += len(targets)
            baseline_ids = list(prefix)
            original_output = []
            eos = [
                int(inv[i]) for i in runtime.contract["mapped_special_token_ids"]["eos"]
            ]
            for _ in range(args.max_new_tokens):
                logits = model(
                    torch.tensor([baseline_ids], device=args.device), use_cache=False
                ).logits[0, -1]
                token = int(torch.argmax(logits).item())
                original_output.append(token)
                baseline_ids.append(token)
                if token in eos:
                    break
            actual = [
                int(inv[i])
                for i in runtime.greedy(
                    [int(p[i]) for i in prefix], args.max_new_tokens
                )
            ]
            if actual != original_output:
                raise ValueError("greedy generation gate failed")
            report["greedy_exact_cases"] += 1
    count = report["target_tokens"]
    report["source_mean_nll"] = report.pop("source_nll_sum") / count
    report["reference_mean_nll"] = report.pop("reference_nll_sum") / count
    delta = report["reference_mean_nll"] - report["source_mean_nll"]
    report["nll_delta"] = delta
    report["perplexity_ratio"] = math.exp(delta)
    if abs(delta) > 4e-4 or report["source_correct"] != report["reference_correct"]:
        raise ValueError("annotated task quality retention gate failed")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Assertions/HF exceptions can contain plaintext IDs or source paths.
        print('{"status":"failed","diagnostic":"external quality gate failed"}')
        raise SystemExit(1) from None
