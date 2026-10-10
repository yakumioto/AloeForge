"""Public configuration for a controlled, single-GPU diagnostic experiment."""

import copy

from aloeforge_reference.dense_runtime import validate_contract

ARCHITECTURE = "AloeForgeDenseForCausalLM"
LOAD_FORMAT = "aloe_dense_diagnostic"
VLLM_VERSION = "0.11.0"
TORCH_VERSION = "2.8.0"
CUDA_VERSION = "12.8"


def hf_config(contract):
    validate_contract(contract)
    return {
        "model_type": "qwen2",
        "architectures": [ARCHITECTURE],
        "hidden_size": contract["physical_hidden_size"],
        "head_dim": contract["head_dim"],
        "vocab_size": contract["vocabulary_size"],
        "intermediate_size": contract["intermediate_size"],
        "num_hidden_layers": contract["num_hidden_layers"],
        "num_attention_heads": contract["num_attention_heads"],
        "num_key_value_heads": contract["num_key_value_heads"],
        "max_position_embeddings": contract["max_position_embeddings"],
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "eos_token_id": contract["mapped_special_token_ids"]["eos"].copy(),
        "aloe_dense_contract": copy.deepcopy(contract),
    }


def loader_options(root, identity):
    if not isinstance(identity, dict) or set(identity) != {
        "artifact_id",
        "binding_id",
        "tokenizer_sha256",
    }:
        raise ValueError("external identity fields required")
    if any(
        not isinstance(x, str)
        or len(x) != 64
        or any(ch not in "0123456789abcdef" for ch in x)
        for x in identity.values()
    ):
        raise ValueError("invalid external identity")
    return {
        "artifact_root": str(root),
        **identity,
        "allow_diagnostic": True,
    }


def validate_hf_config(config, contract):
    expected = hf_config(contract)
    if any(
        getattr(config, key, None) != value
        for key, value in expected.items()
        if key != "torch_dtype"
    ):
        raise ValueError("engine geometry does not match dense contract")


def validate_execution(
    *,
    dtype,
    tp,
    pp,
    quantization,
    speculative,
    lora,
    prefix_cache,
    chunked_prefill,
    eager,
    cache_dtype,
):
    if (
        dtype != "float32"
        or type(tp) is not int
        or tp != 1
        or type(pp) is not int
        or pp != 1
        or quantization is not None
        or speculative is not None
        or lora is not None
        or prefix_cache is not False
        or chunked_prefill is not False
        or eager is not True
        or cache_dtype != "auto"
    ):
        raise ValueError("unsupported diagnostic executor configuration")
