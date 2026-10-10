"""vLLM 0.11.0 custom API candidate; controlled GPU gate remains pending."""

import numpy as np
import torch
from torch import nn
from vllm.attention import Attention
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from aloeforge_reference.dense_runtime import Runtime
from aloeforge_runtime.config import (
    CUDA_VERSION,
    validate_execution,
    validate_hf_config,
)
from aloeforge_runtime.graph import DenseGraph


class PagedAttentionAdapter(nn.Module):
    def __init__(self, c, cache_config, prefix):
        super().__init__()
        self.attn = Attention(
            c["num_attention_heads"],
            c["head_dim"],
            c["head_dim"] ** -0.5,
            num_kv_heads=c["num_key_value_heads"],
            cache_config=cache_config,
            prefix=prefix,
        )

    def forward(self, q, k, v, positions):
        # Scheduler-provided forward context owns slots, causal metadata and cache.
        return self.attn(q, k, v)


class AloeForgeDenseForCausalLM(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        vc = vllm_config
        validate_execution(
            dtype=str(vc.model_config.dtype).removeprefix("torch."),
            tp=vc.parallel_config.tensor_parallel_size,
            pp=vc.parallel_config.pipeline_parallel_size,
            quantization=vc.quant_config,
            speculative=vc.speculative_config,
            lora=vc.lora_config,
            prefix_cache=vc.cache_config.enable_prefix_caching,
            chunked_prefill=vc.scheduler_config.enable_chunked_prefill,
            eager=vc.model_config.enforce_eager,
            cache_dtype=vc.cache_config.cache_dtype,
        )
        if torch.version.cuda != CUDA_VERSION or not torch.cuda.is_available():
            raise ValueError("pinned physical CUDA runner required")
        options = vc.load_config.model_loader_extra_config
        if (
            vc.load_config.load_format != "aloe_dense_diagnostic"
            or not isinstance(options, dict)
            or options.get("allow_diagnostic") is not True
        ):
            raise ValueError("controlled diagnostic loader required")
        c = vc.model_config.hf_config.aloe_dense_contract
        validate_hf_config(vc.model_config.hf_config, c)
        index = 0

        def factory(contract):
            nonlocal index
            name = f"{prefix + '.' if prefix else ''}layers.{index}.attn"
            index += 1
            return PagedAttentionAdapter(contract, vc.cache_config, name)

        self.graph = DenseGraph(c, factory)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise ValueError("unsupported embedding or pipeline input")
        return self.graph(input_ids, positions)

    def compute_logits(self, hidden_states):
        return self.graph.compute_logits(hidden_states)


class DenseLoader(BaseModelLoader):
    def __init__(self, load_config):
        super().__init__(load_config)
        options = load_config.model_loader_extra_config
        if (
            not isinstance(options, dict)
            or set(options)
            != {
                "artifact_root",
                "artifact_id",
                "binding_id",
                "tokenizer_sha256",
                "allow_diagnostic",
            }
            or options["allow_diagnostic"] is not True
        ):
            raise ValueError(
                "external artifact identities and diagnostic opt-in required"
            )
        self.options = options.copy()

    def _verified(self, model_config):
        options = self.options.copy()
        root = options.pop("artifact_root")
        options.pop("allow_diagnostic")
        verified = Runtime.load(root, **options, compute_dtype=np.float32)
        if verified.contract != model_config.hf_config.aloe_dense_contract:
            raise ValueError("model configuration does not match verified artifact")
        validate_hf_config(model_config.hf_config, verified.contract)
        return verified

    def download_model(self, model_config):
        # No HF/network fallback. Validation occurs on a local publication only.
        self._verified(model_config)

    def load_weights(self, model, model_config):
        if not isinstance(model, AloeForgeDenseForCausalLM):
            raise ValueError("custom dense model required")
        verified = self._verified(model_config)
        model.graph.load_weights(
            (name, torch.from_numpy(np.array(a, copy=True)))
            for name, a in verified.tensors.items()
        )
