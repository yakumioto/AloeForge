"""Generate sparse, synthetic safetensors fixtures; never fetch or overwrite models."""

import argparse
import json
import struct
from pathlib import Path


def generate(root, shards, tensors, payload_bytes, padding):
    if not 1 <= shards <= 128 or not 1 <= tensors <= 32768:
        raise ValueError("shard/tensor count out of bounds")
    if shards * tensors > 100000 or not 0 <= padding <= 128:
        raise ValueError("aggregate count/name padding out of bounds")
    if payload_bytes < 1 or payload_bytes % tensors:
        raise ValueError("payload size must be positive and divisible by tensor count")
    root.mkdir(parents=True, exist_ok=False)
    config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    weight_map = {}
    header_bytes = 0
    tensor_bytes = payload_bytes // tensors
    for shard in range(shards):
        filename = f"model-{shard + 1:05d}-of-{shards:05d}.safetensors"
        header = {}
        for tensor in range(tensors):
            name = f"synthetic.{'x' * padding}s{shard:03d}.t{tensor:06d}"
            weight_map[name] = filename
            header[name] = {
                "dtype": "U8",
                "shape": [tensor_bytes],
                "data_offsets": [tensor * tensor_bytes, (tensor + 1) * tensor_bytes],
            }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        if len(encoded) > 8 << 20:
            raise ValueError("synthetic header exceeds admission limit")
        with (root / filename).open("xb") as output:
            output.write(struct.pack("<Q", len(encoded)))
            output.write(encoded)
            output.truncate(8 + len(encoded) + payload_bytes)
        header_bytes += len(encoded)
    index = {
        "metadata": {"total_size": shards * payload_bytes},
        "weight_map": weight_map,
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, separators=(",", ":")), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "shards": shards,
                "tensors": shards * tensors,
                "payload_bytes": shards * payload_bytes,
                "header_bytes": header_bytes,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--tensors-per-shard", type=int, default=4096)
    parser.add_argument("--payload-bytes-per-shard", type=int, default=536870912)
    parser.add_argument("--name-padding", type=int, default=0)
    args = parser.parse_args()
    generate(
        args.root,
        args.shards,
        args.tensors_per_shard,
        args.payload_bytes_per_shard,
        args.name_padding,
    )
