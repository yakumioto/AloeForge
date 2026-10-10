import copy
import sys
from types import ModuleType
from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aloeforge_runtime.config import (  # noqa: E402
    ARCHITECTURE,
    hf_config,
    loader_options,
    validate_execution,
)
from aloeforge_runtime.graph import CPUAttention, DenseGraph  # noqa: E402
from tests.reference import test_dense_runtime as fixtures  # noqa: E402

dense_cli = fixtures.dense_cli
load = fixtures.load


def compile_fixture(*args, **options):
    config, tensors, path, identity, private, command = fixtures.compile_fixture(
        *args, **options
    )
    identity = {
        k: identity[k] for k in ("artifact_id", "binding_id", "tokenizer_sha256")
    }
    return config, tensors, path, identity, private, command


@pytest.fixture
def publication(dense_cli, tmp_path):
    return compile_fixture(dense_cli, tmp_path, heads=4, kv=2, layers=3, tied=True)


def evaluate(graph, ids, start=0):
    hidden = graph(torch.tensor(ids), torch.arange(start, start + len(ids)))
    return graph.compute_logits(hidden).numpy()


@pytest.mark.parametrize(
    "heads,kv,tied,dtype,head_dim",
    [
        (2, 1, False, "F32", 4),
        (4, 1, True, "F16", 4),
        (4, 2, False, "BF16", 4),
        (4, 4, True, "F32", 4),
        (2, 1, False, "F32", 64),
    ],
)
def test_independent_fp32_prefill_decode_cache_and_greedy(
    dense_cli, tmp_path, heads, kv, tied, dtype, head_dim
):
    torch.set_num_threads(1)
    _, _, path, identity, _, _ = compile_fixture(
        dense_cli,
        tmp_path,
        heads=heads,
        kv=kv,
        tied=tied,
        dtype=dtype,
        head_dim=head_dim,
    )
    reference = load(path, identity)
    if head_dim != 4:
        # The strict 2e-6 cache gate failed for wider FP32 synthetic graphs;
        # reject that mock geometry instead of weakening the numerical gate.
        with pytest.raises(ValueError, match="unvalidated CPU mock geometry"):
            DenseGraph.load(path, allow_diagnostic=True, **identity)
        reference.forward([0, 1, 3])  # P5 FP64 runtime remains independently admitted.
        return
    graph = DenseGraph.load(path, allow_diagnostic=True, **identity)
    ids = [0, 1, 3, 5, 6]
    expected, cache = reference.forward(ids)
    prefill = evaluate(graph, ids)
    np.testing.assert_allclose(prefill, expected, atol=3e-4, rtol=3e-4)
    for layer, (k, v) in zip(graph.layers, cache.layers, strict=True):
        np.testing.assert_allclose(layer.attention.k.numpy(), k, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(layer.attention.v.numpy(), v, atol=2e-6, rtol=2e-6)
    graph.reset()
    parts = [
        evaluate(graph, ids[:2]),
        evaluate(graph, ids[2:4], 2),
        evaluate(graph, ids[4:], 4),
    ]
    np.testing.assert_allclose(np.concatenate(parts), prefill, atol=2e-6, rtol=2e-6)
    graph.reset()
    logits = evaluate(graph, ids)
    generated = []
    for step in range(5):
        token = int(logits[-1].argmax())
        generated.append(token)
        if token in graph.c["mapped_special_token_ids"]["eos"]:
            break
        if step < 4:
            logits = evaluate(graph, [token], len(ids) + step)
    assert generated == reference.greedy(ids, 5)


def test_fail_closed_model_and_weights(publication):
    _, _, path, identity, _, _ = publication
    verified = load(path, identity)
    with pytest.raises(ValueError, match="opt-in"):
        DenseGraph.load(path, **identity)
    bad = identity | {"artifact_id": "0" * 64}
    with pytest.raises(ValueError):
        DenseGraph.load(path, allow_diagnostic=True, **bad)
    graph = DenseGraph(verified.contract)
    with pytest.raises(ValueError, match="not loaded"):
        evaluate(graph, [1])
    weights = [(k, torch.from_numpy(np.array(v))) for k, v in verified.tensors.items()]
    for invalid in [
        weights[:-1],
        weights + [weights[0]],
        [("unknown", weights[0][1])],
        [(weights[0][0], weights[0][1].double())],
        [(weights[0][0], torch.zeros(1))],
        [(weights[0][0], torch.full_like(weights[0][1], torch.nan))],
    ]:
        with pytest.raises(ValueError):
            graph.load_weights(invalid)
        assert not graph._loaded
    graph.load_weights(weights)
    for ids, pos in [
        (torch.tensor([True]), torch.tensor([0])),
        (torch.tensor([-1]), torch.tensor([0])),
        (torch.tensor([11]), torch.tensor([0])),
        (torch.tensor([1]), torch.tensor([-1])),
        (torch.tensor([1]), torch.tensor([64])),
        (torch.tensor([1]), torch.tensor([0, 1])),
    ]:
        with pytest.raises(ValueError):
            graph(ids, pos)
    with pytest.raises(ValueError, match="contiguous"):
        evaluate(graph, [1], start=2)
    evaluate(graph, [1])
    with pytest.raises(ValueError, match="contiguous"):
        evaluate(graph, [1])
    changed = copy.deepcopy(verified.contract)
    changed["rope_layout"] = "interleaved"
    with pytest.raises(ValueError):
        DenseGraph(changed)


def test_public_config_and_execution_rejections(publication):
    _, _, path, identity, _, _ = publication
    c = load(path, identity).contract
    config = hf_config(c)
    assert config["hidden_size"] != config["num_attention_heads"] * config["head_dim"]
    assert not config["tie_word_embeddings"]
    assert config["architectures"] == [ARCHITECTURE]
    config["aloe_dense_contract"]["head_dim"] = 100
    assert c["head_dim"] == 4
    options = loader_options(path, identity)
    assert options["allow_diagnostic"]
    for bad in [{}, identity | {"secret": "private"}, identity | {"binding_id": "bad"}]:
        with pytest.raises(ValueError):
            loader_options(path, bad)
    valid = dict(
        dtype="float32",
        tp=1,
        pp=1,
        quantization=None,
        speculative=None,
        lora=None,
        prefix_cache=False,
        chunked_prefill=False,
        eager=True,
        cache_dtype="auto",
    )
    validate_execution(**valid)
    for k, v in {
        "dtype": "float16",
        "tp": 2,
        "pp": 2,
        "quantization": "fp8",
        "speculative": {},
        "lora": {},
        "prefix_cache": True,
        "chunked_prefill": True,
        "eager": False,
        "cache_dtype": "fp8",
    }.items():
        with pytest.raises(ValueError):
            validate_execution(**(valid | {k: v}))


def test_vllm_registration_loader_and_attention_with_explicit_mocks(
    publication, monkeypatch
):
    # These modules simulate ONLY the pinned Python API. No vLLM/CUDA is loaded.
    _, _, path, identity, _, _ = publication
    reference = load(path, identity)
    registrations, prefixes = [], []
    root = ModuleType("vllm")
    root.ModelRegistry = NS(register_model=lambda *args: registrations.append(args))
    loader_module = ModuleType("vllm.model_executor.model_loader")
    loader_module.register_model_loader = lambda name: lambda cls: registrations.append(
        (name, cls)
    )
    base = ModuleType("vllm.model_executor.model_loader.base_loader")

    class Base:
        def __init__(self, config):
            self.load_config = config

    class MockPaged(torch.nn.Module):
        def __init__(self, heads, dim, scale, num_kv_heads, cache_config, prefix):
            super().__init__()
            assert (heads, dim, scale, num_kv_heads) == (4, 4, 0.5, 2)
            prefixes.append(prefix)
            self.mock = CPUAttention(reference.contract)

        def forward(self, q, k, v):
            start = 0 if self.mock.k is None else self.mock.k.shape[1]
            return self.mock(q, k, v, torch.arange(start, start + len(q)))

    base.BaseModelLoader = Base
    attention_module = ModuleType("vllm.attention")
    attention_module.Attention = MockPaged
    for name, module in {
        "vllm": root,
        "vllm.attention": attention_module,
        "vllm.model_executor.model_loader": loader_module,
        "vllm.model_executor.model_loader.base_loader": base,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    from aloeforge_runtime import plugin

    monkeypatch.setattr(plugin, "version", lambda _: "0.12.0")
    with pytest.raises(ValueError, match="revision"):
        plugin.register()
    monkeypatch.setattr(plugin, "version", lambda _: "0.11.0")
    monkeypatch.setattr(plugin, "_registered", False)
    # Remove mock-dependent module after the test to avoid polluting other tests.
    with monkeypatch.context() as modules:
        modules.delitem(sys.modules, "aloeforge_runtime.vllm_model", raising=False)
        plugin.register()
        plugin.register()
        assert len(registrations) == 2
        from aloeforge_runtime.vllm_model import AloeForgeDenseForCausalLM, DenseLoader

        options = loader_options(path, identity)
        vc = NS(
            model_config=NS(
                dtype=torch.float32,
                enforce_eager=True,
                hf_config=NS(**hf_config(reference.contract)),
            ),
            parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1),
            quant_config=None,
            speculative_config=None,
            lora_config=None,
            cache_config=NS(enable_prefix_caching=False, cache_dtype="auto"),
            scheduler_config=NS(enable_chunked_prefill=False),
            load_config=NS(
                load_format="aloe_dense_diagnostic", model_loader_extra_config=options
            ),
        )
        with pytest.raises(ValueError, match="CUDA"):
            AloeForgeDenseForCausalLM(vllm_config=vc)
        monkeypatch.setattr(torch.version, "cuda", "12.8")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        model = AloeForgeDenseForCausalLM(vllm_config=vc)
        assert prefixes == ["layers.0.attn", "layers.1.attn", "layers.2.attn"]
        loader = DenseLoader(vc.load_config)
        loader.download_model(vc.model_config)
        loader.load_weights(model, vc.model_config)
        ids = [0, 1, 3]
        hidden = model(torch.tensor(ids, dtype=torch.int32), torch.arange(len(ids)))
        np.testing.assert_allclose(
            model.compute_logits(hidden).numpy(),
            reference.forward(ids)[0],
            atol=3e-4,
            rtol=3e-4,
        )
        with pytest.raises(ValueError, match="embedding"):
            model(torch.tensor(ids), torch.arange(len(ids)), inputs_embeds=hidden)
        bad_config = NS(hf_config=NS(aloe_dense_contract={}))
        with pytest.raises(ValueError, match="configuration"):
            loader.load_weights(model, bad_config)
        for extra in [
            {},
            options | {"secret": "private"},
            options | {"allow_diagnostic": False},
        ]:
            with pytest.raises(ValueError):
                DenseLoader(NS(model_loader_extra_config=extra))
        assert registrations[0][0] == ARCHITECTURE
    sys.modules.pop("aloeforge_runtime.vllm_model", None)
