"""Independent FP32 graph; attention is supplied by CPU mock or vLLM.

Artifact validation is shared with the P5 loader. Numerical execution does not
call the NumPy reference forward path or the offline transform worker.
"""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aloeforge_reference.dense_runtime import Runtime, validate_contract
from aloeforge_runtime.config import TORCH_VERSION


def check_torch():
    if torch.__version__.split("+")[0] != TORCH_VERSION:
        raise ValueError("unsupported PyTorch revision")


def check_cpu_geometry(c):
    if c["logical_hidden_size"] > 16 or c["head_dim"] != 4:
        raise ValueError("unvalidated CPU mock geometry")


def norm(x, c):
    ratio = c["logical_hidden_size"] / c["physical_hidden_size"]
    return (
        x
        * torch.rsqrt(
            x.square().mean(-1, keepdim=True) + ratio * c["logical_norm_epsilon"]
        )
        * math.sqrt(ratio)
    )


def rotary(x, positions, c):
    dim = c["head_dim"]
    frequency = c["rope_theta"] ** (
        -torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
    )
    angle = positions[:, None].to(torch.float32) * frequency
    cos, sin = angle.cos()[:, None, :], angle.sin()[:, None, :]
    a, b = x[..., : dim // 2], x[..., dim // 2 :]
    return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)


def row_linear(module, x):
    # Keep the CPU mock's FP32 reduction order independent of chunk size.
    # The actual GPU model uses batched linear operators and has its own gate.
    rows = []
    for row in x:
        parts = []
        for weight in module.weight.split(64, dim=0):
            values = row[None, :] * weight
            while values.shape[-1] > 1:
                if values.shape[-1] % 2:
                    values = F.pad(values, (0, 1))
                values = values[:, ::2] + values[:, 1::2]
            parts.append(values[:, 0])
        result = torch.cat(parts)
        if module.bias is not None:
            result = result + module.bias
        rows.append(result)
    return torch.stack(rows)


class CPUAttention(nn.Module):
    """Single-session causal GQA mock. Never a PagedAttention implementation."""

    def __init__(self, c):
        super().__init__()
        # Wider FP32 graphs have not met the fixed cross-reference cache gate.
        # Keep the mock admission narrower than the controlled GPU candidate.
        check_cpu_geometry(c)
        self.c = c
        self.k = self.v = None

    def reset(self):
        self.k = self.v = None

    def forward(self, q, k, v, positions):
        if q.device.type != "cpu":
            raise ValueError("CPU attention mock requires CPU")
        c = self.c
        count = len(positions)
        start = 0 if self.k is None else self.k.shape[1]
        if not torch.equal(positions, torch.arange(start, start + count)):
            raise ValueError("mock positions must be contiguous")
        heads, kv, dim = (
            c["num_attention_heads"],
            c["num_key_value_heads"],
            c["head_dim"],
        )
        q = q.reshape(count, heads, dim).transpose(0, 1)
        k = k.reshape(count, kv, dim).transpose(0, 1)
        v = v.reshape(count, kv, dim).transpose(0, 1)
        if self.k is not None:
            k, v = torch.cat((self.k, k), 1), torch.cat((self.v, v), 1)
        # One query head at a time, without allocating repeated KV caches.
        output = []
        for h in range(heads):
            rows = []
            kv_head = h // (heads // kv)
            for i in range(count):
                end = start + i + 1
                scores = (q[h, i] @ k[kv_head, :end].T) / math.sqrt(dim)
                rows.append(scores.softmax(-1) @ v[kv_head, :end])
            output.append(torch.stack(rows))
        self.k, self.v = k.detach(), v.detach()
        return torch.stack(output, 1).reshape(count, heads * dim)


class Layer(nn.Module):
    def __init__(self, c, attention):
        super().__init__()
        d, f = c["physical_hidden_size"], c["intermediate_size"]
        q, kv = (
            c["num_attention_heads"] * c["head_dim"],
            c["num_key_value_heads"] * c["head_dim"],
        )
        self.q, self.k, self.v = (
            nn.Linear(d, n, bias=True, dtype=torch.float32) for n in (q, kv, kv)
        )
        self.o = nn.Linear(q, d, bias=False, dtype=torch.float32)
        self.gate, self.up = (
            nn.Linear(d, f, bias=False, dtype=torch.float32) for _ in range(2)
        )
        self.down = nn.Linear(f, d, bias=False, dtype=torch.float32)
        self.attention, self.c = attention, c

    def forward(self, hidden, positions):
        c = self.c
        linear = (
            row_linear
            if isinstance(self.attention, CPUAttention)
            else lambda m, x: m(x)
        )
        x = norm(hidden, c)
        q = rotary(
            linear(self.q, x).reshape(-1, c["num_attention_heads"], c["head_dim"]),
            positions,
            c,
        ).flatten(1)
        k = rotary(
            linear(self.k, x).reshape(-1, c["num_key_value_heads"], c["head_dim"]),
            positions,
            c,
        ).flatten(1)
        hidden = hidden + linear(
            self.o, self.attention(q, k, linear(self.v, x), positions)
        )
        x = norm(hidden, c)
        return hidden + linear(
            self.down, F.silu(linear(self.gate, x)) * linear(self.up, x)
        )


class DenseGraph(nn.Module):
    def __init__(self, contract, attention_factory=CPUAttention):
        check_torch()
        validate_contract(contract)
        if attention_factory is CPUAttention:
            check_cpu_geometry(contract)
        d, v, f = (
            contract["physical_hidden_size"],
            contract["vocabulary_size"],
            contract["intermediate_size"],
        )
        q, kv, n = (
            contract["num_attention_heads"] * contract["head_dim"],
            contract["num_key_value_heads"] * contract["head_dim"],
            contract["num_hidden_layers"],
        )
        elements = 2 * v * d + n * (2 * d * (q + kv) + q + 2 * kv + 3 * d * f)
        if elements * 4 > 4 << 30:
            raise ValueError("diagnostic parameter budget exceeded")
        super().__init__()
        self.c = copy.deepcopy(contract)
        d, v = contract["physical_hidden_size"], contract["vocabulary_size"]
        self.embedding = nn.Embedding(v, d, dtype=torch.float32)
        self.head = nn.Linear(d, v, bias=False, dtype=torch.float32)
        self.layers = nn.ModuleList(
            Layer(self.c, attention_factory(self.c))
            for _ in range(contract["num_hidden_layers"])
        )
        self._loaded = False
        self.requires_grad_(False)

    @torch.no_grad()
    def load_weights(self, weights):
        self._loaded = False
        params, seen = dict(self.named_parameters()), set()
        for name, tensor in weights:
            if name not in params or name in seen:
                raise ValueError("unexpected or duplicate runtime tensor")
            p = params[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != torch.float32
                or tensor.shape != p.shape
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("invalid runtime tensor shape/dtype/value")
            p.copy_(tensor)
            seen.add(name)
        if seen != set(params):
            raise ValueError("incomplete runtime tensor inventory")
        self._loaded = True
        return seen

    @classmethod
    def load(cls, root, *, allow_diagnostic=False, **identity):
        if allow_diagnostic is not True:
            raise ValueError("controlled diagnostic opt-in required")
        verified = Runtime.load(root, **identity, compute_dtype=np.float32)
        graph = cls(verified.contract)
        graph.load_weights(
            (name, torch.from_numpy(np.array(a, copy=True)))
            for name, a in verified.tensors.items()
        )
        return graph.eval()

    def reset(self):
        for layer in self.layers:
            if not isinstance(layer.attention, CPUAttention):
                raise ValueError("engine owns non-mock cache lifecycle")
            layer.attention.reset()

    @torch.no_grad()
    def forward(self, input_ids, positions):
        if not self._loaded:
            raise ValueError("verified weights not loaded")
        if (
            not isinstance(input_ids, torch.Tensor)
            or not isinstance(positions, torch.Tensor)
            or input_ids.dtype not in {torch.int32, torch.int64}
            or positions.dtype != torch.int64
            or input_ids.ndim != 1
            or positions.shape != input_ids.shape
            or input_ids.numel() == 0
            or input_ids.device != positions.device
            or input_ids.device != self.embedding.weight.device
            or self.embedding.weight.dtype != torch.float32
            or (input_ids < 0).any()
            or (input_ids >= self.c["vocabulary_size"]).any()
            or (positions < 0).any()
            or (positions >= self.c["max_position_embeddings"]).any()
        ):
            raise ValueError("invalid physical IDs or positions")
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, positions)
        return norm(hidden, self.c)

    @torch.no_grad()
    def compute_logits(self, hidden):
        if not self._loaded or hidden.dtype != torch.float32:
            raise ValueError("invalid diagnostic logits input")
        logits = (
            row_linear(self.head, hidden)
            if isinstance(self.layers[0].attention, CPUAttention)
            else self.head(hidden)
        )
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite diagnostic logits")
        return logits
