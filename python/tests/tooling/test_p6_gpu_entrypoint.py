import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_gpu_entrypoint_requires_optin_and_redacts_failures():
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "python/scripts/p6_validate_gpu.py"),
            "--artifact-root",
            "FORBIDDEN_PRIVATE_PATH",
            "--artifact-id",
            "FORBIDDEN_PRIVATE_ID",
            "--binding-id",
            "bad",
            "--tokenizer-sha256",
            "bad",
            "--workload",
            "FORBIDDEN_PRIVATE_PATH",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 1
    assert json.loads(process.stdout) == {
        "error": "controlled_gpu_gate_failed",
        "gpu_gate": "not_passed",
    }
    assert process.stderr == ""
    assert "FORBIDDEN" not in process.stdout


def test_gpu_entrypoint_help_does_not_import_vllm():
    process = subprocess.run(
        [sys.executable, str(ROOT / "python/scripts/p6_validate_gpu.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0
    assert "--allow-diagnostic" in process.stdout
