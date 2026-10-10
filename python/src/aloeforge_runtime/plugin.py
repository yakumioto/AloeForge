"""Lazy out-of-tree registration. Actual engine integration needs GPU evidence."""

from importlib.metadata import version

from aloeforge_runtime.config import ARCHITECTURE, LOAD_FORMAT, VLLM_VERSION

_registered = False


def register():
    global _registered
    if version("vllm") != VLLM_VERSION:
        raise ValueError("unsupported vLLM revision")
    if _registered:
        return
    from vllm import ModelRegistry
    from vllm.model_executor.model_loader import register_model_loader

    from aloeforge_runtime.vllm_model import DenseLoader

    ModelRegistry.register_model(
        ARCHITECTURE,
        "aloeforge_runtime.vllm_model:AloeForgeDenseForCausalLM",
    )
    register_model_loader(LOAD_FORMAT)(DenseLoader)
    _registered = True
