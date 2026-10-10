"""Controlled synthetic GPU gate; no source model or Client Secret access.

Requires a separate pinned physical GPU runner. Engine logs are suppressed and
only scalar aggregates are printed. Failure diagnostics never echo tokens/paths.
"""

import argparse
import contextlib
import importlib.metadata
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from aloeforge_reference.dense_runtime import Runtime
from aloeforge_runtime.config import (
    CUDA_VERSION,
    TORCH_VERSION,
    VLLM_VERSION,
    hf_config,
    loader_options,
)
from aloeforge_runtime.evidence import (
    check_probabilities,
    log_probabilities,
    metric_delta,
    metric_totals,
    read_workload,
)


def run(args):
    if not args.allow_diagnostic:
        raise ValueError("controlled diagnostic opt-in required")
    identity = {
        key: getattr(args, key)
        for key in ("artifact_id", "binding_id", "tokenizer_sha256")
    }
    reference = Runtime.load(args.artifact_root, **identity)
    c = reference.contract
    # Full distributions are bounded here. A real-checkpoint GPU suite is still
    # required; this runner intentionally admits only small synthetic publications.
    if c["vocabulary_size"] > 512 or c["head_dim"] < 32:
        raise ValueError("controlled synthetic GPU geometry required")
    cases = read_workload(args.workload, c)
    if importlib.metadata.version("vllm") != VLLM_VERSION:
        raise ValueError("unsupported vLLM revision")
    # All engine/worker imports happen after these settings, including spawning.
    os.environ["VLLM_PLUGINS"] = "aloeforge_dense"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
    os.environ["VLLM_LOGGING_LEVEL"] = "CRITICAL"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    import torch
    from vllm import LLM, SamplingParams

    if (
        torch.__version__.split("+")[0] != TORCH_VERSION
        or torch.version.cuda != CUDA_VERSION
        or not torch.cuda.is_available()
    ):
        raise ValueError("pinned physical CUDA runner required")
    from aloeforge_runtime.plugin import register

    register()
    with tempfile.TemporaryDirectory(prefix="aloe-gpu-config-") as temp:
        config_root = Path(temp)
        (config_root / "config.json").write_text(json.dumps(hf_config(c)))
        llm = LLM(
            model=str(config_root),
            skip_tokenizer_init=True,
            trust_remote_code=False,
            load_format="aloe_dense_diagnostic",
            model_loader_extra_config=loader_options(
                Path(args.artifact_root).resolve(), identity
            ),
            dtype="float32",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            enforce_eager=True,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            disable_log_stats=False,
            max_model_len=c["max_position_embeddings"],
            max_logprobs=c["vocabulary_size"],
            max_num_seqs=8,
            max_num_batched_tokens=c["max_position_embeddings"] * 8,
            gpu_memory_utilization=0.5,
        )
        prompts = [{"prompt_token_ids": x["physical_ids"]} for x in cases]
        sampling = [
            SamplingParams(
                temperature=0,
                max_tokens=x["max_tokens"],
                detokenize=False,
                logprobs=-1,
                prompt_logprobs=-1,
            )
            for x in cases
        ]
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        max_error, generated = 0.0, []
        for case, result in zip(cases, outputs, strict=True):
            ids = case["physical_ids"]
            expected_ids = reference.greedy(ids, case["max_tokens"])
            output = result.outputs[0]
            if list(output.token_ids) != expected_ids:
                raise ValueError("exact physical greedy gate failed")
            logits, cache = reference.forward(ids)
            if result.prompt_logprobs is None or len(result.prompt_logprobs) != len(
                ids
            ):
                raise ValueError("prefill evidence missing")
            for index in range(1, len(ids)):
                max_error = max(
                    max_error,
                    check_probabilities(
                        result.prompt_logprobs[index],
                        log_probabilities(logits[index - 1]),
                    ),
                )
            if output.logprobs is None or len(output.logprobs) != len(expected_ids):
                raise ValueError("decode evidence missing")
            for index, token in enumerate(expected_ids):
                max_error = max(
                    max_error,
                    check_probabilities(
                        output.logprobs[index], log_probabilities(logits[-1])
                    ),
                )
                if index + 1 < len(expected_ids):
                    logits, cache = reference.forward([token], cache)
            generated.append(expected_ids)
        for prompt, tokens, case in zip(prompts, generated, cases, strict=True):
            stopped = llm.generate(
                [prompt],
                SamplingParams(
                    temperature=0,
                    max_tokens=case["max_tokens"],
                    stop_token_ids=[tokens[0]],
                    detokenize=False,
                ),
                use_tqdm=False,
            )
            if list(stopped[0].outputs[0].token_ids) != tokens[:1]:
                raise ValueError("physical stop-token gate failed")
        # Identical physical workload/output bounds, one warmup above and three
        # timed batches. GPU timings come from the engine, not CPU proxies.
        bench_sampling = [
            SamplingParams(temperature=0, max_tokens=x["max_tokens"], detokenize=False)
            for x in cases
        ]
        before = metric_totals(llm.get_metrics())
        started = time.perf_counter()
        for _ in range(3):
            results = llm.generate(prompts, bench_sampling, use_tqdm=False)
            if [list(r.outputs[0].token_ids) for r in results] != generated:
                raise ValueError("repeat/cache isolation gate failed")
        elapsed = time.perf_counter() - started
        timings = metric_delta(before, metric_totals(llm.get_metrics()))
        gpu_bytes = llm.llm_engine.collective_rpc("apply_model", args=(gpu_memory,))
        cpu = measure_reference(reference, cases)
        return {
            "profile": "p6-controlled-synthetic-diagnostic",
            "synthetic_gpu_gate": "passed",
            "gpu_gate": "pending_real_model",
            "vllm": VLLM_VERSION,
            "torch": TORCH_VERSION,
            "cuda": CUDA_VERSION,
            "gpu_device": torch.cuda.get_device_properties(0).name,
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "cases": len(cases),
            "max_log_probability_error": max_error,
            "gpu": {
                **timings,
                "output_tokens_per_second": 3 * sum(map(len, generated)) / elapsed,
                "worker_memory_and_cache": gpu_bytes,
            },
            "reference_cpu": cpu,
            "production_gate": "blocked",
            "real_model_gpu_gate": "pending",
        }


def gpu_memory(model):
    import torch

    caches = []
    for layer in model.graph.layers:
        tensor = layer.attention.attn.kv_cache[0]
        c = model.graph.c
        if (
            tensor.ndim != 5
            or tensor.shape[1] != 2
            or tensor.shape[-2:] != (c["num_key_value_heads"], c["head_dim"])
            or tensor.dtype != torch.float32
            or tensor.device.type != "cuda"
        ):
            raise ValueError("physical paged-cache geometry gate failed")
        caches.append({"shape": list(tensor.shape), "dtype": "float32"})
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "layer_caches": caches,
    }


def measure_reference(reference, cases):
    ttft, decode, counts = [], [], []
    start = time.perf_counter()
    for _ in range(3):
        for case in cases:
            tick = time.perf_counter()
            logits, cache = reference.forward(case["physical_ids"])
            token = int(logits[-1].argmax())
            ttft.append(time.perf_counter() - tick)
            count = 1
            while (
                count < case["max_tokens"]
                and token not in reference.contract["mapped_special_token_ids"]["eos"]
            ):
                tick = time.perf_counter()
                logits, cache = reference.forward([token], cache)
                token = int(logits[-1].argmax())
                decode.append(time.perf_counter() - tick)
                count += 1
            counts.append(count)
    if not decode:
        raise ValueError("workload must exercise decode")
    return {
        "ttft_seconds": float(np.mean(ttft)),
        "tpot_seconds": float(np.mean(decode)),
        "output_tokens_per_second": sum(counts) / (time.perf_counter() - start),
        "gpu_memory_bytes": 0,
        "compute_dtype": "float64",
        "batching": "sequential",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in [
        "artifact-root",
        "artifact-id",
        "binding-id",
        "tokenizer-sha256",
        "workload",
    ]:
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        # Redirect file descriptors as well as Python streams; spawned engine
        # workers inherit the suppression. Exceptions expose only static text.
        with open(os.devnull, "w") as sink:
            saved = (os.dup(1), os.dup(2))
            try:
                os.dup2(sink.fileno(), 1)
                os.dup2(sink.fileno(), 2)
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    report = run(args)
            finally:
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
                os.close(saved[0])
                os.close(saved[1])
    except Exception:
        print('{"error":"controlled_gpu_gate_failed","gpu_gate":"not_passed"}')
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
