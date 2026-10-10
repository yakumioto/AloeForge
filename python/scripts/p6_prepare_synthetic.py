"""Generate a small controlled GPU fixture; private files stay in trusted scratch.

This test utility downloads no checkpoint and prints no Secret or token values.
Never copy its source/secret directories to the GPU runner.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from tests.reference.test_dense_runtime import source_fixture  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cli", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(mode=0o700)
    source = root / "source"
    source_fixture(source, heads=2, kv=1, head_dim=64, vocab=71, intermediate=70)
    output, secret = root / "output", root / "secret"
    output.mkdir(mode=0o700)
    secret.mkdir(mode=0o700)
    process = subprocess.run(
        [
            str(args.cli.resolve()),
            "compile-dense",
            "--model-root",
            str(source),
            "--output-parent",
            str(output),
            "--secret-root",
            str(secret),
            "--stage",
            ".stage-p6",
            "--target",
            "published",
            "--python",
            sys.executable,
            "--expansion",
            "4",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if process.returncode:
        raise ValueError("synthetic compiler failed")
    metadata = json.loads(process.stdout)
    identity = {
        k: metadata[k] for k in ("artifact_id", "binding_id", "tokenizer_sha256")
    }
    # Workload is selected from public physical coordinates, not original prompts.
    workload = {
        "schema_version": 1,
        "cases": [
            {"physical_ids": [0, 1, 3, 5], "max_tokens": 8},
            {"physical_ids": [6, 7, 9, 11, 13, 15], "max_tokens": 8},
        ],
    }
    for name, value in [("identity.json", identity), ("workload.json", workload)]:
        path = root / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as file:
            json.dump(value, file, separators=(",", ":"))
    print('{"fixture":"p6-synthetic","head_dim":64,"cases":2}')


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print('{"error":"synthetic_fixture_failed"}')
        raise SystemExit(1) from None
