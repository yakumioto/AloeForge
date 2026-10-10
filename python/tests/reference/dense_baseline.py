"""Scalar source-graph baseline. No transformed runtime or worker imports."""

import math

import numpy as np


def forward(config, tensors, ids):
    d, dim = (
        config["hidden_size"],
        config["hidden_size"] // config["num_attention_heads"],
    )
    heads, kv = config["num_attention_heads"], config["num_key_value_heads"]
    group = heads // kv

    def linear(x, name, bias=False):
        w = tensors[name + ".weight"]
        b = tensors[name + ".bias"] if bias else [0] * len(w)
        return [
            math.fsum(float(a) * float(v) for a, v in zip(row, x)) + float(z)
            for row, z in zip(w, b, strict=True)
        ]

    def norm(x, name):
        denominator = math.sqrt(
            math.fsum(v * v for v in x) / d + config["rms_norm_eps"]
        )
        return [
            v * float(g) / denominator for v, g in zip(x, tensors[name], strict=True)
        ]

    def rotate(x, position):
        result = list(x)
        for start in range(0, len(x), dim):
            for pair in range(dim // 2):
                angle = position * config["rope_theta"] ** (-2 * pair / dim)
                a, b = x[start + pair], x[start + pair + dim // 2]
                result[start + pair] = a * math.cos(angle) - b * math.sin(angle)
                result[start + pair + dim // 2] = a * math.sin(angle) + b * math.cos(
                    angle
                )
        return result

    hidden = [[float(v) for v in tensors["model.embed_tokens.weight"][i]] for i in ids]
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}."
        normalized = [norm(x, prefix + "input_layernorm.weight") for x in hidden]
        queries = [
            rotate(linear(x, prefix + "self_attn.q_proj", True), t)
            for t, x in enumerate(normalized)
        ]
        keys = [
            rotate(linear(x, prefix + "self_attn.k_proj", True), t)
            for t, x in enumerate(normalized)
        ]
        values = [linear(x, prefix + "self_attn.v_proj", True) for x in normalized]
        output = []
        for t, query in enumerate(queries):
            attended = []
            for head in range(heads):
                q = query[head * dim : (head + 1) * dim]
                offset = (head // group) * dim
                scores = [
                    math.fsum(a * b for a, b in zip(q, key[offset : offset + dim]))
                    / math.sqrt(dim)
                    for key in keys[: t + 1]
                ]
                probabilities = [math.exp(s - max(scores)) for s in scores]
                denominator = math.fsum(probabilities)
                attended.extend(
                    math.fsum(
                        probabilities[j] * values[j][offset + k] for j in range(t + 1)
                    )
                    / denominator
                    for k in range(dim)
                )
            attention = linear(attended, prefix + "self_attn.o_proj")
            residual = [a + b for a, b in zip(hidden[t], attention, strict=True)]
            normalized = norm(residual, prefix + "post_attention_layernorm.weight")
            gate, up = (
                linear(normalized, prefix + "mlp.gate_proj"),
                linear(normalized, prefix + "mlp.up_proj"),
            )
            activated = [
                a / (1 + math.exp(-a)) * b for a, b in zip(gate, up, strict=True)
            ]
            down = linear(activated, prefix + "mlp.down_proj")
            output.append([a + b for a, b in zip(residual, down, strict=True)])
        hidden = output
    head = "model.embed_tokens" if config["tie_word_embeddings"] else "lm_head"
    return np.asarray([linear(norm(x, "model.norm.weight"), head) for x in hidden])


def greedy(config, tensors, ids, count):
    ids, output = list(ids), []
    eos = config["eos_token_id"]
    eos = eos if isinstance(eos, list) else [eos]
    for _ in range(count):
        token = int(np.argmax(forward(config, tensors, ids)[-1]))
        output.append(token)
        ids.append(token)
        if token in eos:
            break
    return output
