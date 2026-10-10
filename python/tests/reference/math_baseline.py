"""Independent scalar FP64 toy decoder; no NumPy/worker/oracle imports."""

import math


def matmul(left, right):
    return [
        [
            math.fsum(row[k] * right[k][j] for k in range(len(right)))
            for j in range(len(right[0]))
        ]
        for row in left
    ]


def add(left, right):
    return [[x + y for x, y in zip(a, b)] for a, b in zip(left, right)]


def rms(rows, gamma, epsilon):
    return [
        [
            value
            * gamma[i]
            / math.sqrt(math.fsum(v * v for v in row) / len(row) + epsilon)
            for i, value in enumerate(row)
        ]
        for row in rows
    ]


def feed_forward(rows, gate, up, down):
    a, b = matmul(rows, gate), matmul(rows, up)
    hidden = [
        [(x / (1 + math.exp(-x))) * y for x, y in zip(r1, r2)] for r1, r2 in zip(a, b)
    ]
    return matmul(hidden, down)


def rotate(row, position, frequency, layout):
    size = len(row) // 2
    out = row.copy()
    for i, freq in enumerate(frequency):
        a, b = (2 * i, 2 * i + 1) if layout == "interleaved" else (i, i + size)
        angle = position * freq
        out[a] = row[a] * math.cos(angle) + row[b] * math.sin(angle)
        out[b] = row[b] * math.cos(angle) - row[a] * math.sin(angle)
    return out


def attend(rows, query, key, value, output, positions, frequencies, layout):
    heads, groups = len(query), len(key)
    width = len(query[0][0])
    result = [[0.0] * len(output[0][0]) for _ in rows]
    all_scores = []
    for head in range(heads):
        group = head // (heads // groups)
        q = [
            rotate(r, p, frequencies[group], layout)
            for r, p in zip(matmul(rows, query[head]), positions)
        ]
        k = [
            rotate(r, p, frequencies[group], layout)
            for r, p in zip(matmul(rows, key[group]), positions)
        ]
        v = matmul(rows, value[group])
        scores = [
            [math.fsum(a * b for a, b in zip(qr, kr)) / math.sqrt(width) for kr in k]
            for qr in q
        ]
        all_scores.append(scores)
        for token, score in enumerate(scores):
            maximum = max(score[: token + 1])
            exp = [math.exp(s - maximum) for s in score[: token + 1]]
            total = math.fsum(exp)
            weights = [w / total for w in exp]
            mixed = [
                [
                    math.fsum(weights[j] * v[j][i] for j in range(token + 1))
                    for i in range(width)
                ]
            ]
            row = matmul(mixed, output[head])[0]
            result[token] = [a + b for a, b in zip(result[token], row)]
    return result, all_scores


def decoder(
    tokens,
    embedding,
    head,
    query,
    key,
    value,
    output,
    gate,
    up,
    down,
    gammas,
    epsilon,
    positions,
    frequencies,
    layout,
):
    rows = [embedding[t].copy() for t in tokens]
    normalized = rms(rows, gammas[0], epsilon)
    attended, _ = attend(
        normalized, query, key, value, output, positions, frequencies, layout
    )
    rows = add(rows, attended)
    normalized = rms(rows, gammas[1], epsilon)
    rows = add(rows, feed_forward(normalized, gate, up, down))
    rows = rms(rows, gammas[2], epsilon)
    return matmul(rows, head)
